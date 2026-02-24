# start image with ps script with env variables included to S3 bucket

# base image
FROM python:3.12.8

#create work directory
WORKDIR /code

# copy server_requirements.txt and run
COPY ./server_requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#copy app code
COPY ./server.py /code/app/server.py

# Expose fast API port
EXPOSE 8000

# set MLflow communication uri as an env variable
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

# start server
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]