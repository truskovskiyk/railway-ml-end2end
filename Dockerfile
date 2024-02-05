FROM python:3.11.3 as pipeline

WORKDIR /app


RUN pip install --upgrade pip
COPY requirements-pipeline.txt requirements-pipeline.txt
RUN pip install -r requirements-pipeline.txt

ENV DAGSTER_HOME /data/pipelines
RUN mkdir -p $DAGSTER_HOME

COPY . .

CMD touch $DAGSTER_HOME/dagster.yaml && dagster dev -p 3000 -h 0.0.0.0 -f /app/pipelines/main.py


FROM nvcr.io/nvidia/tritonserver:24.01-vllm-python-py3 as serving

WORKDIR /app

COPY requirements-serving.txt requirements-serving.txt
RUN pip install -r requirements-serving.txt

RUN mkdir -p models
COPY serving/ /app/models

CMD tritonserver --model-repository /app/models


FROM python:3.11.3 as monitoring

WORKDIR /app

COPY requirements-monitoring.txt requirements-monitoring.txt
RUN pip install -r requirements-monitoring.txt

COPY . .

CMD streamlit run --server.port 8080 --server.address 0.0.0.0 monitoring/main.py