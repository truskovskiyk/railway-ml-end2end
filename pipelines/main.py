from dagster import asset, Config, AssetExecutionContext, Output, MetadataValue

from data.main import load_text_to_sql_dataset
from data.main import map_dataset_from_sql_to_surreal_sql
from data.main import check_if_surreal_can_execute
from data.main import load_data_for_labeling


@asset(group_name="data")
def origin_text2sql_dataset():
    dataset = load_text_to_sql_dataset()
    return Output(
        dataset,
        metadata={
            "preview": MetadataValue.md(dataset.to_pandas().head().to_markdown()),
        },
    )


class MappedDataConfig(Config):
    num_samples: int = 10


@asset(group_name="data")
def mapped_text2sql_dataset_to_surreal(origin_text2sql_dataset, config: MappedDataConfig):
    dataset = map_dataset_from_sql_to_surreal_sql(num_samples=config.num_samples, dataset=origin_text2sql_dataset)
    return Output(
        dataset,
        metadata={
            "preview": MetadataValue.md(dataset.to_pandas().head().to_markdown()),
        },
    )


@asset(group_name="data")
def surreal_db_dataset_with_execution(mapped_text2sql_dataset_to_surreal):
    dataset = check_if_surreal_can_execute(surreal_db=mapped_text2sql_dataset_to_surreal)
    return Output(
        dataset,
        metadata={
            "preview": MetadataValue.md(dataset.to_pandas().head().to_markdown()),
        },
    )


class LabelingDataConfig(Config):
    dataset_name: str


@asset(group_name="data")
def labeling_data(context: AssetExecutionContext, surreal_db_dataset_with_execution):
    dataset_name = f"surrealdb-{context.run_id}"
    url = load_data_for_labeling(dataset_name=dataset_name, surreal_db=surreal_db_dataset_with_execution)
    return Output(
        url,
        metadata={
            "dataset_url": MetadataValue.url(url),
        },
    )


@asset(group_name="model_tbd")
def pre_trained_llm():
    pass


@asset(group_name="model_tbd")
def feedback_dataset():
    pass


@asset(group_name="model_tbd")
def test_dataset():
    pass


@asset(group_name="model_tbd")
def updated_model(pre_trained_llm, feedback_dataset, test_dataset):
    pass
