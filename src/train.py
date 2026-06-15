import os
import random
import numpy as np
import torch
import yaml
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType

from preprocess import load_and_prepare
from evaluate import compute_rouge


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QADataset(Dataset):
    def __init__(self, df, tokenizer, max_input, max_target, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_target = max_target
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        model_inputs = self.tokenizer(
            row["prompt"],
            max_length=self.max_input,
            truncation=True,
            padding="max_length",
        )
        if not self.is_test:
            labels = self.tokenizer(
                row["output"],
                max_length=self.max_target,
                truncation=True,
                padding="max_length",
            )
            label_ids = [
                -100 if t == self.tokenizer.pad_token_id else t
                for t in labels["input_ids"]
            ]
            model_inputs["labels"] = label_ids
        return {k: torch.tensor(v) for k, v in model_inputs.items()}


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_and_tokenizer(cfg):
    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if cfg["peft"]["use_lora"]:
        peft_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=cfg["peft"]["r"],
            lora_alpha=cfg["peft"]["lora_alpha"],
            lora_dropout=cfg["peft"]["lora_dropout"],
            target_modules=cfg["peft"]["target_modules"],
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()
    return tokenizer, model


def train(cfg_path="configs/config.yaml", output_dir="outputs/checkpoint"):
    cfg = load_config(cfg_path)
    set_seed(cfg["training"]["seed"])

    tokenizer, model = build_model_and_tokenizer(cfg)

    train_df = load_and_prepare(cfg["data"]["train_path"])
    val_df = load_and_prepare(cfg["data"]["val_path"])

    train_dataset = QADataset(
        train_df, tokenizer,
        cfg["model"]["max_input_length"],
        cfg["model"]["max_target_length"],
    )
    val_dataset = QADataset(
        val_df, tokenizer,
        cfg["model"]["max_input_length"],
        cfg["model"]["max_target_length"],
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        scores = compute_rouge(decoded_preds, decoded_labels)
        return scores

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["training"]["num_epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        warmup_steps=cfg["training"]["warmup_steps"],
        weight_decay=cfg["training"]["weight_decay"],
        fp16=cfg["training"]["fp16"],
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL_f1",
        logging_steps=50,
        seed=cfg["training"]["seed"],
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    return trainer


if __name__ == "__main__":
    train()
