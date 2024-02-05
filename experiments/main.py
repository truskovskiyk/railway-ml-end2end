import typer


from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import (
    get_peft_model,
    TaskType,
    LoraConfig,
    PrefixTuningConfig,
)

import pandas as pd
import datasets
from dataclasses import dataclass
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from typing import List, Union


import mlflow

mlflow.set_tracking_uri(uri="https://mlflow-tracking-production.up.railway.app")

TEMPLATE = "schema: {schema}; text: {user_query}; result:"


def get_data_sql_create_context(tokenizer: AutoTokenizer) -> List[Union[DatasetDict, int, int]]:
    dataset = load_dataset("b-mc2/sql-create-context")
    print(f"Dataset size: {len(dataset['train'])}")

    tokenized_inputs = dataset["train"].map(lambda x: tokenizer(TEMPLATE.format(schema=x["context"], text=x["question"]), truncation=True),)

    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    tokenized_targets = dataset["train"].map(
        lambda x: tokenizer(f'{x["answer"]}', truncation=True),
    )

    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")

    train, val = train_test_split(dataset["train"].to_pandas(), random_state=42, test_size=0.01)
    _dataset = DatasetDict({"train": Dataset.from_pandas(train), "eval": Dataset.from_pandas(val)})
    return _dataset, max_source_length, max_target_length

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


def main(model_name_or_path: str = "google/flan-t5-large", epochs: float = 5.0):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # loading dataset
    dataset, max_source_length, max_target_length = get_data(tokenizer)

    def preprocess_function(sample, padding="max_length"):
        inputs = [
            f"schema: {context}; text: {question}" for (context, question) in zip(sample["context"], sample["question"])
        ]

        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        labels = tokenizer(
            text_target=sample["answer"],
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )

        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, batch_size=1000)
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    print("Getting PEFT method")
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM
    )
    results_dir = "output"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, lora_config)
    print(model.print_trainable_parameters())

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        save_total_limit=5,
        logging_first_step=True,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        output_dir=results_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        report_to="mlflow",
    )

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    print(f"training_args = {training_args}")
    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        data_collator=data_collator,
    )
    model.config.use_cache = False

    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    eval_stats = trainer.evaluate()
    eval_loss = eval_stats["eval_loss"]
    print(f"Training loss:{train_loss}|Val loss:{eval_loss}")

    peft_model_id = f"{results_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    print("Experiment over")


if __name__ == "__main__":
    typer.run(main)
