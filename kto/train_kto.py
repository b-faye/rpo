import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    get_scheduler,
)
from evaluate import load
from tqdm import tqdm
import pandas as pd
import os

# Evaluation metrics
bertscore = load("bertscore")
rouge = load("rouge")
bleu = load("bleu")
sacrebleu = load("sacrebleu")
meteor = load("meteor")
toxicity = load("toxicity", module_type="measurement")


def tokenize_dataset(dataset, tokenizer, max_len):
    def preprocess(example):
        input_enc = tokenizer(example["instruction"], padding="max_length", truncation=True, max_length=max_len)
        target_enc = tokenizer(example["response"], padding="max_length", truncation=True, max_length=max_len)

        labels = [(tid if tid != tokenizer.pad_token_id else -100) for tid in target_enc["input_ids"]]

        return {
            "input_ids": input_enc["input_ids"],
            "attention_mask": input_enc["attention_mask"],
            "labels": labels,
            "reward": example.get("reward", 0),
            "instruction": example["instruction"],
            "response": example["response"],
        }

    return dataset.map(preprocess)


def custom_collate_fn(batch):
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
        "labels": torch.tensor([item["labels"] for item in batch]),
        "reward": torch.tensor([item["reward"] for item in batch], dtype=torch.float),
        "instruction": [item["instruction"] for item in batch],
        "response": [item["response"] for item in batch],
    }


def compute_all_metrics(preds, refs, device):
    length_ratios = [len(p.split()) / max(1, len(r.split())) for p, r in zip(preds, refs)]
    return {
        "bertscore_f1": np.mean(bertscore.compute(predictions=preds, references=refs, lang="en", device=device)["f1"]),
        "rougeL": rouge.compute(predictions=preds, references=refs)["rougeL"],
        "bleu": bleu.compute(predictions=preds, references=refs)["bleu"],
        "sacrebleu": sacrebleu.compute(predictions=preds, references=refs)["score"],
        "meteor": meteor.compute(predictions=preds, references=refs)["meteor"],
        "toxicity_avg": np.mean(toxicity.compute(predictions=preds)["toxicity"]),
        "length_ratio": np.mean(length_ratios),
    }


class KTOTrainer:
    def __init__(self, model, ref_model, train_loader, val_loader, tokenizer, device, config):
        self.model = model
        self.ref_model = ref_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.config = config

        self.optimizer = AdamW(self.model.parameters(), lr=config["lr"])
        steps_per_epoch = len(train_loader) // config["gradient_accumulation_steps"]
        total_steps = steps_per_epoch * config["epochs"]

        self.scheduler = get_scheduler(
            "linear", optimizer=self.optimizer, num_warmup_steps=150, num_training_steps=total_steps
        )

        self.beta = config["beta"]
        self.lambda_d = config["lambda_d"]
        self.lambda_u = config["lambda_u"]

    @staticmethod
    def compute_log_probs(logits, labels):
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        vocab_size = logits.size(-1)
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1)).view(labels.size())
        mask = labels != -100
        return -(loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def kto_loss(self, r_theta, z0, rewards):
        desirable = rewards > 0
        undesirable = ~desirable

        beta_r = torch.clamp(self.beta * (r_theta - z0), -20, 20)
        beta_z = torch.clamp(self.beta * (z0 - r_theta), -20, 20)

        v = torch.zeros_like(r_theta)
        v[desirable] = self.lambda_d * torch.sigmoid(beta_r[desirable])
        v[undesirable] = self.lambda_u * torch.sigmoid(beta_z[undesirable])

        lambdas = torch.where(desirable, self.lambda_d, self.lambda_u)
        return (lambdas - v).mean()

    def train(self):
        for epoch in range(self.config["epochs"]):
            self.model.train()
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            total_loss = total_samples = 0

            for step, batch in enumerate(progress):
                inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = self.model(**inputs)
                log_probs = self.compute_log_probs(outputs.logits, inputs["labels"])

                with torch.no_grad():
                    ref_outputs = self.ref_model(**inputs)
                    ref_log_probs = self.compute_log_probs(ref_outputs.logits, inputs["labels"])

                r_theta = log_probs - ref_log_probs
                z0 = r_theta.mean().detach()
                loss = self.kto_loss(r_theta, z0, inputs["reward"]) / self.config["gradient_accumulation_steps"]
                loss.backward()

                total_loss += loss.item() * len(batch["input_ids"])
                total_samples += len(batch["input_ids"])

                if (step + 1) % self.config["gradient_accumulation_steps"] == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if (step + 1) % self.config["save_per_steps"] == 0:
                    self.save_model(step + 1)
                    self.evaluate(step + 1)

    def evaluate(self, step):
        self.model.eval()
        preds, refs, instrs = [], [], []
        total_loss = total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Evaluating step {step}"):
                inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = self.model(**inputs)
                ref_outputs = self.ref_model(**inputs)

                r_theta = self.compute_log_probs(outputs.logits, inputs["labels"]) - \
                          self.compute_log_probs(ref_outputs.logits, inputs["labels"])
                z0 = r_theta.mean()
                loss = self.kto_loss(r_theta, z0, inputs["reward"])

                total_loss += loss.item() * len(batch["input_ids"])
                total_samples += len(batch["input_ids"])

                generated = self.model.generate(inputs["input_ids"], max_new_tokens=self.config["max_new_tokens"])
                decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

                preds.extend(decoded)
                refs.extend(batch["response"])
                instrs.extend(batch["instruction"])

        avg_loss = total_loss / total_samples
        print(f"Validation Loss @ Step {step}: {avg_loss:.4f}")

        results_df = pd.DataFrame({
            "Instruction": instrs,
            self.config["csv_name"]: preds
        })
        results_df.to_csv(f"{self.config['saved_model_name']}-step-{step}.csv", index=False)

        metrics = compute_all_metrics(preds, refs, self.device)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    def save_model(self, step):
        path = f"checkpoints/{self.config['saved_model_name']}-step-{step}"
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def train_kto(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(config["base_model"])
    model = T5ForConditionalGeneration.from_pretrained(config["base_model"]).to(device)
    ref_model = T5ForConditionalGeneration.from_pretrained(config["base_model"]).to(device)

    train_data = load_dataset(config["dataset"], split="train")
    val_data = load_dataset(config["dataset"], split="validation")

    reward_thresh = np.mean(train_data["reward"])

    def binarize_reward(ex):
        ex["reward"] = 1 if ex["reward"] >= reward_thresh else -1
        return ex

    train_data = train_data.map(binarize_reward)
    val_data = val_data.map(binarize_reward)

    train_data = tokenize_dataset(train_data, tokenizer, config["max_len"])
    val_data = tokenize_dataset(val_data, tokenizer, config["max_len"])

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, collate_fn=custom_collate_fn)

    trainer = KTOTrainer(model, ref_model, train_loader, val_loader, tokenizer, device, config)
    trainer.train()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    configs = {
        "small": {
            "base_model": "google/flan-t5-small",
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "max_len": 512,
            "save_per_steps": 2500,
            "csv_name": "KTO SMALL",
            "saved_model_name": "flan-t5-small-kto"
        },
        "large": {
            "base_model": "google/flan-t5-large",
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "max_len": 512,
            "save_per_steps": 10000,
            "csv_name": "KTO LARGE",
            "saved_model_name": "flan-t5-large-kto"
        },
        "xl": {
            "base_model": "google/flan-t5-xl",
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "max_len": 256,
            "save_per_steps": 20000,
            "csv_name": "KTO XL",
            "saved_model_name": "flan-t5-xl-kto"
        },
    }

    # Choose one of: "small", "large", or "xl"
    model_size = "small"
    cfg = {
        **configs[model_size],
        "lr": 1e-4,
        "epochs": 1,
        "max_new_tokens": 768,
        "beta": 1.0,
        "lambda_d": 1.0,
        "lambda_u": 1.0,
        "dataset": "bilalfaye/ultrafeedback-rpo-split"
    }

    train_kto(cfg)
