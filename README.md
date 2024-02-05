# Railway ML

## Setup 

```
python -m venv ~/env/
source ~/env/bin/activate
pip install -r requirements.txt
```

## Datasets


```
python data/main.py load-text-to-sql-dataset
python data/main.py map-dataset-from-sql-to-surreal-sql --num-samples 1000
python data/main.py check-if-surreal-can-execute
python data/main.py load-data-for-labeling
```

- https://huggingface.co/datasets/b-mc2/sql-create-context
- https://huggingface.co/datasets/Clinton/Text-to-sql-v1
- https://railway.app/template/Axgpqb


## Experiments 

```
python experiments/main.py --model-name-or-path google/flan-t5-small --epochs 5.0
```


- https://huggingface.co/cssupport/t5-small-awesome-text-to-sql
- https://huggingface.co/machinists/Mistral-7B-SQL
- https://huggingface.co/machinists/Mistral-7B-Instruct-SQL


## Pipelines



```
dagster dev -p 3000 -h 0.0.0.0 -f pipelines/main.py
```

```
docker build -t kyrylprojector/pipeline:latest --target pipeline .
docker run -it -p 3000:3000 kyrylprojector/pipeline:latest
docker push kyrylprojector/pipeline:latest
```

## Serving

```
docker build -t kyrylprojector/serving:latest --target serving .
docker run -it -p 8000:8000 kyrylprojector/serving:latest
docker push kyrylprojector/serving:latest
```

## Monitoring

```
docker build -t kyrylprojector/monitoring:latest --target monitoring .
docker run -it -p 8080:8080 kyrylprojector/monitoring:latest
```
