# Run:  .\start-mlflow.ps1

$env:AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
$env:AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
$env:AWS_DEFAULT_REGION = "us-east-1"

# docker is set to: ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination s3://mlstore-wg0ti2bljtkghtvp9n --host 0.0.0.0 --port 5000 --allowed-hosts="127.0.0.1:5000,localhost,host.docker.internal:5000"