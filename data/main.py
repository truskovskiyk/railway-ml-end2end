from datasets import load_dataset
import typer
from tqdm import tqdm
import json
from openai import OpenAI
from surrealdb import Surreal
from surrealdb.ws import SurrealException
import asyncio
from datasets import Dataset
import pandas as pd
import argilla as rg
from diskcache import Cache
from functools import wraps
from pydantic_settings import BaseSettings, SettingsConfigDict

SQL_TO_SURREAL_QL = """
SQL to SurrealDB mapping
Quickly learn how to map your SQL knowledge to corresponding SurrealDB concepts and syntax.

Introduction
As a multi-model database, SurrealDB offers a lot of flexibility. Our SQL-like query language SurrealQL is a good example of this, where we often have more than one way to achieve the same result, depending on developer preference. In this mapping guide, we will focus on the syntax that most closely resembles the ANSI Structured Query Language (SQL).

Concepts mapping
For more in-depth explanations of SurrealDB concepts, see the concepts page.

Relational databases	SurrealDB
database	database
table	table
row	record
column	field
index	index
primary key	record id
transaction	transaction
join	record links, embedding and graph relations
Syntax mapping
Let's get you up to speed with SurrealQL syntax with some CRUD examples.

Create
As relational databases are schemafull, only the SurrealQL schemafull approach is shown below. For a schemafull option see the define table page.

For more SurrealQL examples, see the create and insert pages.

SQL	SurrealQL
CREATE TABLE person ( person_id SERIAL PRIMARY KEY, name varchar(255) ) // SERIAL is PosgresSQL syntax	DEFINE TABLE person SCHEMAFULL; DEFINE FIELD name ON TABLE person TYPE string; // record id field is defined by default
INSERT INTO person (name) VALUES ('John'), ('Jane')	INSERT INTO person (name) VALUES ('John'), ('Jane')
CREATE INDEX idx_name ON person (name)	DEFINE INDEX idx_name ON TABLE person COLUMNS name
column	field
index	index
primary key	record id
transaction	transaction
join	record links, embedding and graph relations
Read
For more SurrealQL examples, see the select, live select and return pages.

SQL	SurrealQL
SELECT * FROM person	SELECT * FROM person
SELECT name FROM person	SELECT name FROM person
SELECT name FROM person WHERE name = "Jane"	SELECT name FROM person WHERE name = "Jane"
EXPLAIN SELECT name FROM person WHERE name = "Jane"	SELECT name FROM person WHERE name = "Jane" EXPLAIN
SELECT count(*) AS person_count FROM person	SELECT count() AS person_count FROM person GROUP ALL
SELECT DISTINCT name FROM person	SELECT array::distinct(name) FROM person GROUP ALL
SELECT * FROM person LIMIT 10	SELECT * FROM person LIMIT 10
SELECT review.*, person.name as reviewer FROM review INNER JOIN person on review.person_id = person.id	SELECT *, person.name as reviewer FROM review
Update
For more SurrealQL examples, see the update page.

SQL	SurrealQL
ALTER TABLE person ADD last_name varchar(255)	DEFINE FIELD last_name ON TABLE person TYPE string
UPDATE person SET last_name = "Doe" WHERE name = "Jane"	UPDATE person SET last_name = "Doe" WHERE name = "Jane"
ALTER TABLE person DROP COLUMN last_name	REMOVE FIELD last_name ON TABLE person
Delete
For more SurrealQL examples, see the delete and remove pages.

SQL	SurrealQL
DELETE FROM person WHERE name = "Jane"	DELETE person WHERE name = "Jane"
DELETE FROM person	DELETE person
DROP TABLE person	REMOVE TABLE person
"""


from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    SURREAL_URI: str
    SURREAL_USER: str
    SURREAL_PASS: str
    OPEN_AI_KEY: str
    ARGILLA_URI: str
    ARGILLA_KEY: str
    ARGILLA_NAMESPACE: str


SURREAL_URI = Settings().SURREAL_URI
SURREAL_USER = Settings().SURREAL_USER
SURREAL_PASS = Settings().SURREAL_PASS
OPEN_AI_KEY = Settings().OPEN_AI_KEY
ARGILLA_URI = Settings().ARGILLA_URI
ARGILLA_KEY = Settings().ARGILLA_KEY
ARGILLA_NAMESPACE = Settings().ARGILLA_NAMESPACE


def cache_diskcache(timeout: int = 86400 * 10, cache_path: str = ".diskcache"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Cache(cache_path) as cache:
                key = str((func.__name__, args, kwargs))
                if key in cache:
                    return cache[key]
                result = func(*args, **kwargs)
                cache.set(key, result, timeout)
                return result

        return wrapper

    return decorator


def load_text_to_sql_dataset():
    dataset = load_dataset("b-mc2/sql-create-context")["train"]
    print(f"Loaded dataset {dataset}")
    return dataset


@cache_diskcache()
def run_surreal_query(sq_schema: str, sq_query: str) -> (bool, bool):
    async def main():
        db = Surreal(SURREAL_URI)
        await db.connect()
        await db.signin({"user": SURREAL_USER, "pass": SURREAL_PASS})
        await db.use("test", "test")
        try:
            res = await db.query(sq_schema)
            sq_schema_valid = res[0]["status"] == "OK"
        except SurrealException:
            sq_schema_valid = False

        try:
            res = await db.query(sq_query)
            sq_query_valid = res[0]["status"] == "OK"
        except SurrealException:
            sq_query_valid = False

        await db.close()
        return sq_schema_valid, sq_query_valid

    sq_schema_valid, sq_query_valid = asyncio.run(main())
    return sq_schema_valid, sq_query_valid


@cache_diskcache()
def map_one_sample(sql_query: str, sql_schema: str) -> (str, str):
    client = OpenAI(api_key=OPEN_AI_KEY)
    prompt = f"""
    Based on this tutorial, which describe how to map SQL queries to SurrealDB queries or (SurrealQL). 

    ###
    {SQL_TO_SURREAL_QL}
    ###

    Map next 2 SQL statments into SurrealQL: 

    "sql_schema": {sql_schema}
    "sql_query": {sql_query}

    Provide response in JSON format, please map everything into 1 query: 
    {{"sq_schema": "OUTPUT", "sq_query", "OUTPUT"}}
    """

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a expert in SQL and SurrealDB databases."},
            {"role": "user", "content": prompt},
        ],
    )
    output_text = response.choices[0].message.content
    try:
        output_text_dict = json.loads(output_text)
    except json.JSONDecodeError as ex:
        print(ex)
        output_text_dict = {}
    
    sq_query = output_text_dict.get("sq_query")
    sq_schema = output_text_dict.get("sq_schema")

    if isinstance(sq_schema, list):
        sq_schema = " ".join(sq_schema)

    return sq_query, sq_schema


def map_dataset_from_sql_to_surreal_sql(num_samples: int = None, dataset=None):
    if dataset is None:
        dataset = load_dataset("b-mc2/sql-create-context")["train"]

    if num_samples is not None:
        dataset = dataset.select(range(num_samples))

    data = []
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]

        sql_query = sample["answer"]
        sql_schema = sample["context"]

        sq_query, sq_schema = map_one_sample(sql_query=sql_query, sql_schema=sql_schema)

        data.append(
            {
                "sql_query": sql_query,
                "sql_schema": sq_schema,
                "sq_query": sq_query if isinstance(sq_query, str) else None,
                "sq_schema": sq_schema if isinstance(sq_schema, str) else None,
                "user_query": sample["question"],
            }
        )

    df = pd.DataFrame(data)
    surreal_db = Dataset.from_pandas(df=df)
    surreal_db.save_to_disk(dataset_path="surreal_db_stage_1")
    return surreal_db


def check_if_surreal_can_execute(surreal_db=None):
    if surreal_db is None:
        surreal_db = Dataset.load_from_disk(dataset_path="surreal_db_stage_1")

    data = []
    for idx in tqdm(range(len(surreal_db))):
        sample = surreal_db[idx]

        sq_schema = sample["sq_schema"]
        sq_query = sample["sq_query"]

        sq_schema_valid, sq_query_valid = run_surreal_query(sq_schema=sq_schema, sq_query=sq_query)
        data.append({"sq_schema_valid": sq_schema_valid, "sq_query_valid": sq_query_valid})

    surreal_db = surreal_db.add_column(name="sq_schema_valid", column=[x["sq_schema_valid"] for x in data])
    surreal_db = surreal_db.add_column(name="sq_query_valid", column=[x["sq_query_valid"] for x in data])
    surreal_db.save_to_disk(dataset_path="surreal_db_stage_2")
    return surreal_db


def load_data_for_labeling(dataset_name: str = "surrealdb-dataset-from-sql-create-context", surreal_db=None):
    if surreal_db is None:
        surreal_db = Dataset.load_from_disk(dataset_path="surreal_db_stage_2")

    rg.init(api_url=ARGILLA_URI, api_key=ARGILLA_KEY, workspace=ARGILLA_NAMESPACE)

    dataset = rg.FeedbackDataset(
        guidelines="Text to SurrealQL.",
        fields=[
            rg.TextField(name="user_query", title="Text from user"),
            rg.TextField(name="sql_query", title="Original SQL query"),
            rg.TextField(name="sql_schema", title="Original SQL schema"),
            rg.TextField(name="sq_query", title="SurrealQL query", required=False),
            rg.TextField(name="sq_schema", title="SurrealQL table schema", required=False),
            rg.TextField(name="sq_schema_valid", title="is SurrealQL table schema valid?", required=False),
            rg.TextField(name="sq_query_valid", title="is SurrealQL query valid?", required=False),
        ],
        questions=[
            rg.TextQuestion(
                name="corrected_sq_query",
                title="Provide a correction to the SurrealQL query:",
                required=True,
                use_markdown=True,
            ),
            rg.TextQuestion(
                name="corrected_sq_table_schema",
                title="Provide a correction to the SurrealQL table schema:",
                required=True,
                use_markdown=True,
            ),
        ],
    )

    records = []
    for idx in tqdm(range(len(surreal_db))):
        sample = surreal_db[idx]
        fields = sample
        record = rg.FeedbackRecord(fields=fields)
        records.append(record)

    dataset.add_records(records)
    res = dataset.push_to_argilla(name=dataset_name)
    return res.url


def cli():
    app = typer.Typer()
    app.command()(load_text_to_sql_dataset)
    app.command()(map_dataset_from_sql_to_surreal_sql)
    app.command()(check_if_surreal_can_execute)
    app.command()(load_data_for_labeling)
    app()


if __name__ == "__main__":
    cli()
