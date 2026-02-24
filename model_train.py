import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import mlflow
from mlflow.models import infer_signature
from torch_model import SimpleCNN
import numpy as np

# visualize
import matplotlib.pyplot as plt

# params
DO_TRAIN = True
MODEL_NAME = "Product_Classifier"
MODEL_VERSION = "0.0.1"
NUM_EPOCHS = 200
BATCH_SIZE = 64
DATASET_NAME = "Fashion_MNIST"
DATASET_VERSION = "1.0"
DEVICE = "cpu"


# set random seed
manual_seed = 42
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)

### Loading Training Data
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Load training and testing data
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

### Creating Model
device = DEVICE

# detect device
if (True):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

print(f"Using {device} device")

def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        mlflow.log_metric("train_loss", loss, step=epoch)


def test(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    mlflow.log_metric("train_accuracy", correct, step=epoch)
    mlflow.log_metric("val_loss", test_loss, step=epoch)

model = SimpleCNN().to(device)

### Optimizing the Model Parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# collected hyperparams, metrics, model data for MLflow
mlf_params = {
    "name": MODEL_NAME,
    "version": MODEL_VERSION,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "dataset": DATASET_NAME,
    "dataset_version": DATASET_VERSION,
    "optimizer": optimizer.__class__.__name__,
    "model_architecture": "CNN",
    "random_seed": manual_seed,
    "loss_function": loss_fn.__class__.__name__,
}

mlf_metrics = {
    "train_loss": 0.0,
    "train_accuracy": 0.0,
    "val_loss": 0.0,
    "epoch": 0
}

from datetime import date
mlf_model_meta = {
    "training_date": date.today().strftime("%Y-%m-%d"),
}

# Infer signature from input/output
# Input Schema: float32, (-1,1,28,28), name "images"
input_example = torch.randn(2, 1, 28, 28)
# Output Schema: float32, (-1,10), name "logits"
predictions = torch.randn(1, 10)
# Infer signature from input/output
signature = infer_signature(input_example.numpy(), predictions.detach().numpy())

# setup mlflow experiment
experiment_id = None
try:
    # new experiment force S3 location
    experiment_id = mlflow.create_experiment('RefundClassify', artifact_location='s3://mlstore-wg0ti2bljtkghtvp9n')
except Exception:
    # update experiment
    mlflow.set_experiment("RefundClassify")

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
with mlflow.start_run(experiment_id=experiment_id) as run:
    # log params into MLflow
    mlflow.log_params(mlf_params)

    # train model
    epochs = NUM_EPOCHS
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, epoch+1)
        test(test_dataloader, model, loss_fn, epoch+1)
        # logs metrics into MLflow
        mlflow.log_metric("epoch", epoch+1, step=epoch+1)

    print("Training is done!")

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    # Log all files in a directory
    mlflow.log_artifacts("artifacts/", artifact_path="artifacts")

    # log model into MLflow with metrics and input example
    model_cpu = model.to("cpu") # set to CPU as current docker only supports cpu
    mlflow.pytorch.log_model(
        pytorch_model=model_cpu,
        name="model",
        signature=signature,
        code_paths=["./torch_model.py"],
        #pip_requirements=["-r", "requirements.txt"],
        input_example=input_example.numpy(),
        registered_model_name="CNN_Simple"
        )



