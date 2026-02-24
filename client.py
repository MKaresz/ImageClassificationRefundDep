import sys
import io
import json
import asyncio
import httpx
import pandas as pd
import click
import requests
import logging
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from PIL import Image
from typing import Tuple
import shutil

# -------------------------------------------------------------------
# Globals
# -------------------------------------------------------------------
API_URL = "http://host.docker.internal:8000"
# set current timezone for human readability
TIME_ZONE = timezone(timedelta(hours=1))
# folder structure
RAW_ROOT = Path("data/raw")
PROCESSED_ROOT = Path("data/processed")
PREDICTIONS_ROOT = Path("data/predictions")
QUARANTINE_ROOT = Path("data/quarantine")
LOG_ROOT = Path("E:/ImageClassificationRefundDep")

MICRO_BATCH_SIZE = 200
MODEL_VERSION = "v0.1"
ADMIN_API_KEY = "HubtRueBD65daUt4opFc9ez2t"

# -------------------------------------------------------------------
# Setup logger
# -------------------------------------------------------------------
# Configure the logger to write to a file
logging.basicConfig(
    level=logging.INFO,
    filename=f"{LOG_ROOT}/client_log_{date.today().isoformat()}.log",
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
h = logging.StreamHandler(stream=sys.stdout)
h.setLevel(logging.INFO)
logger.addHandler(h)

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def today_str() -> str:
    """Return YYYY-MM-DD for today's folder."""
    return date.today().isoformat()


def yesterday_str() -> str:
    """Return YYYY-MM-DD for yesterday's analytics."""
    return (date.today() - timedelta(days=1)).isoformat()

def ensure_dirs() -> None:
    """
    Ensure that all required directory roots exist.

    This function is idempotent and safe to call multiple times.
    """
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    QUARANTINE_ROOT.mkdir(parents=True, exist_ok=True)

def load_processed_ids(ledger_file: Path) -> set:
    """Return a set of item_ids already processed for a given day.
    
    Args:
        ledger_file: Path to the JSONL ledger file.

    Returns:
        A set of item_ids that have already been logged
    """
    if not ledger_file.exists():
        return set()

    processed = set()
    with open(ledger_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            processed.add(entry["item_id"])
    return processed

# -------------------------------------------------------------------
# Micro-batch sending to server
# -------------------------------------------------------------------
async def send_batch(
    client: httpx.AsyncClient,
    batch: list[Tuple[Path, Image.Image]]
    ) -> dict:
    """
    Send a micro-batch of images (list of tuples (path, PIL.Image))
    to the prediction API.

    Args:
        client: Reusable HTTP client for performance.
        api_url: Base URL of the prediction API
        batch : list of tuples (Path, Image.Image) List of image paths and loaded PIL images.

    Returns:
        dict: JSON response from the server.

    Raises:
        httpx.HTTPError: If the request fails or the server returns an error.
    """
    multipart = []
    for img_path, pil_img in batch:
        buf = io.BytesIO()
        pil_img.save(buf, "PNG")
        buf.seek(0)
        multipart.append((
            "files",
            (img_path.name, buf.getvalue(), "image/png")
        ))

    response = await client.post(
            f"{API_URL}/predict",
            files=multipart,
            timeout=60.0
        )
    response.raise_for_status()
    return response.json()



# -------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------
async def inference_day(day) -> None:
    """
    Run inference on all unprocessed images for a given day and produce a ledger file.

    - Scans day's folder
    - Filters unprocessed images
    - Sends to server in micro-batches
    - Appends predictions to processed.jsonl

    Args:
        day : Folder name in YYYY-MM-DD format. If empty, defaults to today. 
    """
    #check default folders available
    ensure_dirs()
    if day == "":
        day = today_str()

    # create path
    raw_day = RAW_ROOT / day
    processed_day = PROCESSED_ROOT / day

    # checks folder exists and writable
    processed_day.mkdir(parents=True, exist_ok=True)
    if not raw_day.exists():
        logger.info(f"No raw folder found for today {day}")
        return

    # get already processed items
    ledger_file = processed_day / "processed.jsonl"
    already_done = load_processed_ids(ledger_file)

    # Gather images
    all_imgs = list(raw_day.glob("*.jpg"))
    to_process = [p for p in all_imgs if p.name not in already_done]
    if not to_process:
        logger.info(f"No new images to process for {day}.")
        return
    logger.info(f"Found {len(to_process)} new images to process.")

    # Load images
    pairs = []
    for p in to_process:
        try:
            with open(p, "rb") as fh:
                img = Image.open(p).convert("RGB")
                pairs.append((p, img))

        except Exception as e:
            logger.error(f"Failed to load image {p.name}: {e}")

            # move problematic file to quarantine
            q_day = QUARANTINE_ROOT / day
            q_day.mkdir(parents=True, exist_ok=True)
            shutil.move(p, q_day / p.name)

            # generate JSON error log
            with open(q_day / f"{p.name}.json", "w") as jf:
                json.dump({
                    "timestamp": datetime.now(TIME_ZONE).isoformat(),
                    "item_id": p.name,
                    "error": "Client-side load decode failure:",
                    "code": str(e)
                }, jf, indent=2)
            continue

    # Create micro-batches
    batches = [
        pairs[i:i + MICRO_BATCH_SIZE]
        for i in range(0, len(pairs), MICRO_BATCH_SIZE)
    ]

    # Process batches
    with open(ledger_file, "a", encoding="utf-8") as ledger:
        async with httpx.AsyncClient(timeout=60.0) as client:
            for batch in batches:
                logger.info(f"Sending batch of {len(batch)} images...")
                resp = await send_batch(client, batch)

                for result in resp.get("processed_images", []):
                    entry = {
                        "timestamp": datetime.now(TIME_ZONE).isoformat(),
                        "item_id": result["image_name"],
                        "predicted": result["predicted"],
                        "confidence": result["confidence"],
                        "status": result.get("status", "ok"),
                        "model_version": MODEL_VERSION
                    }
                    ledger.write(json.dumps(entry) + "\n")             
                    
                    # handle quarantine on error
                    if result["status"] == "error":
                        # Destination directory
                        q_day = QUARANTINE_ROOT / day
                        q_day.mkdir(parents=True, exist_ok=True)

                        # Move bad image
                        src = raw_day / result["image_name"]
                        dest = q_day / result["image_name"]
                        if src.exists():
                            shutil.move(src, dest)

                        # Optional: write debugging metadata
                        meta_path = q_day / f"{result["image_name"]}.json"
                        with open(meta_path, "w", encoding="utf-8") as mf:
                            json.dump(entry, mf, indent=2)

    logger.info(f"All images for {day} processed successfully.")

# -------------------------------------------------------------------
# Model analytics
# -------------------------------------------------------------------
def run_daily_analytics(day):
    """
    Run analytics on the processed predictions for a given day.

    Should run only after last batch is finished at end-of-day or next morning first.
    Reads previous day's JSONL ledger, writes Parquet, and outputs drift metrics.
    
    Args:
        day : Folder name in YYYY-MM-DD format. If empty, defaults to yesterday.

    """
    # set working folder
    ensure_dirs()
    if day == "":
        day = yesterday_str()
    processed_day = PROCESSED_ROOT / day

    # get ledger file
    ledger_file = processed_day / "processed.jsonl"
    if not ledger_file.exists():
        logger.info(f"No processed.jsonl found for {day} — nothing to analyze.")
        return

    # Read JSONL into DataFrame
    records = []
    with open(ledger_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Write Parquet output for day
    PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    out_file = PREDICTIONS_ROOT / f"{day}.parquet"
    df.to_parquet(out_file, index=False)
    logger.info(f"Saved Parquet file to: {out_file}")

    # model analysis
    logger.info(f"Running analytics for {day}...")
    logger.info("\n--- MODEL ANALYTICS ---")
    logger.info("Class distribution:")
    
    # class appearance table
    logger.info(df["predicted"].value_counts(dropna=False))
    
    # get mean confidence
    mean_conf = df["confidence"].mean()
    logger.info(f"Mean confidence: {mean_conf}")

    # get most common class and get percentage of balance
    top_class = df["predicted"].value_counts(normalize=True).idxmax()
    top_share = df["predicted"].value_counts(normalize=True).max()

    warning_conf = (mean_conf < 0.4)
    warning_imbalance = (top_share > 0.7)

    if warning_conf or warning_imbalance:
        logger.info("\nWARNING:")
        if warning_conf:
            logger.info(f"- Mean confidence is very low: {mean_conf*100:.2f}% possible model drift!")
        if warning_imbalance:
            logger.info(f"- Class imbalance detected: {top_class} = {top_share:.1%}")
       
    # create json dump from model analytics
    payload = {
        "most_frequent_class": top_class,
        "most_frequent_class_share": top_share,
        "warning_model_drift": bool(warning_conf),
        "warning_imbalance": bool(warning_imbalance),
    }
    json_str = json.dumps(payload, ensure_ascii=False, indent=2)  
    with open(processed_day / "model_analytics.jsonl", "w", encoding="utf-8") as f:
        f.write(json_str)

    logger.info("\nAnalytics file is written.")

# -------------------------------------------------------------------
# CLI interface
# -------------------------------------------------------------------
@click.command(no_args_is_help=True)
@click.option('--health', is_flag=True, help="Get server status, info and version number.")
@click.option('--inference', is_flag=True, help="Inference on today's input images under: ./raw/YYYY-MM-DD/")
@click.option('--analytics', is_flag=True, help="Model analytics on yesterday's inferences: ./predictions/YYYY-MM-DD.parquet")
@click.option('--day', type=click.DateTime(formats=["%Y-%m-%d"]), help="Optional date for analytics (format: YYYY-MM-DD)")
@click.option('--model_reload', is_flag=True, help="Ask server to update model to latest champion or default model.")
def main(health, inference, analytics, day, model_reload):
    """batch-processing client that sends images to  FastAPI 
    inference server and performs daily analytics.

    Tasks: Inference, Analytics, Reload-ML model
        - gets server status
        - reads images from disk
        - batches them
        - sends them to the FastAPI server
        - logs predictions
        - runs analytics
        - calls model re-load on server
    """
    if health:
        response = requests.get(f"{API_URL}/health")
        logger.info(f"Status: {response.status_code}")
        logger.info(f"JSON: {response.json()}")
    if inference:
        if day:
            asyncio.run(inference_day(str(day.date())))
        else:
            asyncio.run(inference_day(today_str()))

    if analytics:
        if day:
            run_daily_analytics(str(day.date()))
        else:
            run_daily_analytics(yesterday_str())
    
    if model_reload:
        response = requests.post(f"{API_URL}/admin/reload", 
                                 headers={"X-API-Key": ADMIN_API_KEY})
        logger.info(f"Status: {response.status_code}")
        logger.info(f"JSON: {response.json()}")


if __name__ == "__main__":
    main()

