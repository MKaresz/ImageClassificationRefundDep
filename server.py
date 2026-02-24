import io, os
from typing import List
import logging
import asyncio
from contextlib import asynccontextmanager

# fast API
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, status, Header
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from fastapi.security import APIKeyHeader

# pytorch
import torch
from PIL import Image
import torchvision.transforms as transforms

# mlflow
import mlflow
import mlflow.pyfunc
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient

# Server started by Docker command

# -------------------------------------------------------------------
# Globals
# -------------------------------------------------------------------
SERVER_VERSION = "0.31"
MIN_CONFIDENCE = 0.6
MODEL_NAME = "CNN_Simple"
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

# output classes for human readable prediction
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

# Fast API logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# protect from multiple calls
_reload_lock = asyncio.Lock()

# -------------------------------------------------------------------
# MLflow setup
# -------------------------------------------------------------------
mlflow_uri = os.environ["MLFLOW_TRACKING_URI"]
mlflow.set_tracking_uri(mlflow_uri)
client = MlflowClient()

# -------------------------------------------------------------------
# MLFLow load model helper
# -------------------------------------------------------------------
def load_ml_model(model_name=MODEL_NAME) -> PyFuncModel:
    """
    Load the machine learning modal aliased as champion from mlFlow storage.
    If loading is not successful fall back to model tagged with "basemodel".
    
    Args:
        model_name: name of the model to load, defaults to CNN_Simple
         
    Returns:
        ml_model: machine learning model instance from mlFlow
    """
    ml_model = None
    try:
        ml_model = mlflow.pyfunc.load_model("models:/CNN_Simple@champion")
        logger.info("Loaded champion model.")
    except Exception:
        for mv in client.search_model_versions(f"name = '{model_name}'"):
            if "basemodel" in mv.tags:
                base_model_version = int(mv.version)
                model_uri = f"models:/{model_name}/{base_model_version}"
                logger.info(f"Loading basemodel version {base_model_version} from MLflow...")
                ml_model = mlflow.pyfunc.load_model(model_uri)
                logger.info("Loaded base model.")
    return ml_model

# -------------------------------------------------------------------
# Very simple security check for admin endpoints
# -------------------------------------------------------------------
async def verify_api_key(x_api_key: str = Header(default=None)) -> bool:
    """
    Require X-API-Key header to access protected endpoints.

    Returns:
        bool: True on success or exception
    """
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return True

# -------------------------------------------------------------------
# Server Lifespan 
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager that is executed once when the FastAPI application starts
    and once again when it shuts down. It loads the machine-learning model.

    Args:
        app: The FastAPI application instance.
    """
    logger.info(f"Server v:{SERVER_VERSION} is starting...")
    # to reach ML-model inside app
    app.state.model = load_ml_model()
    # FastAPI starts serving after this
    yield
    logger.info(f"Server v:{SERVER_VERSION} is shutting down...")


# -------------------------------------------------------------------
# FastAPI Dependencies
# -------------------------------------------------------------------
def get_model(request: Request) -> PyFuncModel:
    """Retrieve the machine-learning model stored in the application's state.
    It allows route handlers to receive the model without relying on global 
    variables or manual imports.

    Args:
        app: The FastAPI application instance injected by FastAPI.

    Returns:
        Any: The loaded machine-learning model.

    """
    return request.app.state.model

# -------------------------------------------------------------------
# Start Server App
# -------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

# -------------------------------------------------------------------
# Server Endpoints
# -------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Image classification API for the refund department version:" + str(SERVER_VERSION) + "."}

@app.get("/health")
def health():
    return JSONResponse({
        "status": "ok",
        "server_version": SERVER_VERSION
    })
  
@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    model: PyFuncModel = Depends(get_model),
) -> JSONResponse:
    """
    Run inference on one or more uploaded image files and return a prediction.

    Args:
        files:
            A list of image files to be processed. Each file is read, decoded,
            preprocessed, and passed through the model.
        model:
            The machine-learning model injected via dependency. Expected to
            expose a `.predict()` method that accepts a NumPy float32 batch
            shaped as (N, 1, 28, 28).

    Returns:
        A JSON object containing a list of entries for each image including:
            - the original filename
            - the predicted class label
            - the confidence score (float)
            - a status flag indicating low or acceptable confidence
    """

    if model is None:
        raise HTTPException(status_code=503, detail="Error: No model is loaded.")

    # collect batch result 
    results = []
    
    # preprocess images
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        except Exception:
            logging.error(f"Invalid image file: {file.filename}.")
            results.append({
                "image_name": file.filename,
                "predicted": 0,
                "probabilities":torch.zeros(10).tolist(),
                "confidence": 0.0,
                "status": "error"
            })
            continue
        
        img_tensor = transform(image).unsqueeze(0)
        # convert tensor to pyfunc to match input schema => numpy float32 shape (1,1,28,28)
        img_np = img_tensor.detach().cpu().numpy().astype("float32")

        with torch.no_grad():
            logits_np = model.predict(img_np)
            logits_t = torch.from_numpy(logits_np)
            # Convert to probabilities (softmax over last dim)
            probs_t = torch.softmax(logits_t, dim=1)
            conf, pred_idx = torch.max(probs_t, dim=1)
            pred_idx = int(pred_idx.item())
            confidence = float(conf.item())
            pred_class = CLASSES[pred_idx]
            print(probs_t.shape)
            status = "low_confidence" if conf < MIN_CONFIDENCE else "ok"
            results.append({
                "image_name": file.filename,
                "predicted": pred_class,
                "probabilities": probs_t.squeeze(0).tolist(),
                "confidence": confidence,
                "status": status
            })
    return JSONResponse({"processed_images": results})

@app.post("/admin/reload", dependencies=[Depends(verify_api_key)])
async def admin_reload(request: Request) -> JSONResponse:
    """ Reload the machine-learning model used by the application.
    
    Returns:
        A JSON object indicating whether the reload succeeded. 
        On success: `{"status": "ok", "message": "Model reloaded."}`
        On failure: `{"status": "error", "message": "<error>"}`
    """
    if _reload_lock.locked():
        # Optional: reject concurrent reloads or just wait for the lock
        return JSONResponse(
            status_code = status.HTTP_409_CONFLICT,
            content={"status": "busy", "message": "Reload already in progress."},
        )

    async with _reload_lock:
        logger.info("Reloading model.")

        try:
            model = await asyncio.to_thread(load_ml_model)
            if model is None:
                raise ValueError("Missing Champion/Base model!")
            
            request.app.state.model = model
            logger.info("Reloaded model: %s", model)
            return JSONResponse(
                status_code = status.HTTP_200_OK,
                content={"status": "ok", "message": "Model reloaded."},
            )

        except Exception as e:
            logger.error("Model reload failed: %s", e)
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model reload failed: {e}",
            )

