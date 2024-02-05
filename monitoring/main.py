from tqdm import tqdm
import argilla as rg
import requests
from pydantic_settings import BaseSettings, SettingsConfigDict
import streamlit as st


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    ARGILLA_URI: str
    ARGILLA_KEY: str
    ARGILLA_NAMESPACE: str
    SERVING_URL: str
    FEEDBACK_DATASET_NAME: str

ARGILLA_URI = Settings().ARGILLA_URI
ARGILLA_KEY = Settings().ARGILLA_KEY
ARGILLA_NAMESPACE = Settings().ARGILLA_NAMESPACE
SERVING_URL = Settings().SERVING_URL
FEEDBACK_DATASET_NAME = Settings().FEEDBACK_DATASET_NAME

TEMPLATE = "schema: {schema}; user_query: {user_query}; result:"



class FeedbackLoopSubmitter:
    def __init__(self) -> None:
        rg.init(api_url=ARGILLA_URI, api_key=ARGILLA_KEY, workspace=ARGILLA_NAMESPACE)

        name = FEEDBACK_DATASET_NAME
        dataset = rg.FeedbackDataset(
            guidelines="Text to SurrealQL.",
            fields=[
                rg.TextField(name="user_query", title="Text from user"),
                rg.TextField(name="table_schema", title="Schema", required=False),
                rg.TextField(name="llm_text_input", title="Full LLM input", required=False),
                rg.TextField(name="generated_text", title="LLM output", required=False),
                rg.TextField(name="serving_url", title="serving_url", required=False),
            ],
            questions=[
                rg.TextQuestion(
                    name="corrected_sq_query",
                    title="Provide a correction to the SurrealQL query:",
                    required=True,
                    use_markdown=True,
                ),
                rg.LabelQuestion(
                    name="is_ql_query_correct",
                    title="Is QL query correct?",
                    required=True,
                    labels=["correct", "not correct"],
                ),
            ],
        )
        if name in [x.name for x in rg.list_datasets()]:
            feedback_dataset = rg.FeedbackDataset.from_argilla(name=name)
        else:
            feedback_dataset = dataset.push_to_argilla(name=name)
        self.feedback_dataset = feedback_dataset

    def submit_feedback(
        self, table_schema: str, user_query: str, llm_text_input: str, generated_text: str, serving_url: str
    ):
        record = rg.FeedbackRecord(
            fields={
                "user_query": user_query,
                "table_schema": table_schema,
                "llm_text_input": llm_text_input,
                "generated_text": generated_text,
                "serving_url": serving_url,
            }
        )
        self.feedback_dataset.add_records([record])

    def generate(self, schema: str, user_query: str) -> str:
        llm_text_input = TEMPLATE.format(schema=schema, user_query=user_query)

        headers = {"Content-Type": "application/json"}
        data = {"inputs": llm_text_input, "parameters": {"max_new_tokens": 512}}

        response = requests.post(SERVING_URL, headers=headers, json=data)
        generated_text = response.json()[0]["generated_text"]

        self.submit_feedback(
            table_schema=schema,
            user_query=user_query,
            llm_text_input=llm_text_input,
            generated_text=generated_text,
            serving_url=SERVING_URL,
        )

        return generated_text


@st.cache_resource
def get_model():
    return FeedbackLoopSubmitter()


def ui():
    model = get_model()
    schema = st.text_input("schema")
    user_query = st.text_input("user_query")

    st.write("The current schema is", schema)
    st.write("The current user_query", user_query)

    generate_ql = st.button("Generate QL")
    if generate_ql:
        generated_ql = model.generate(schema=schema, user_query=user_query)
        st.write("Result:", generated_ql)


if __name__ == "__main__":
    ui()
