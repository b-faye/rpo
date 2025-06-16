import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, AdamW, get_scheduler
)
from torch.utils.data import DataLoader
from evaluate import load

# Metrics
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
            "response": example["response"]
        }
    return dataset.map(preprocess)


def collate_batch(batch):
    return {
        "input_ids": torch.tensor([b["input_ids"] for b in batch]),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch]),
        "reward": torch.tensor([b["reward"] for b in batch], dtype=torch.float),
        "instruction": [b["instruction"] for b in batch],
        "response": [b["response"] for b in batch],
    }


def compute_metrics(preds, refs, device):
    length_ratios = [len(p.split()) / max(1, len(r.split())) for p, r in zip(preds, refs)]
    avg_ratio = sum(length_ratios) / len(length_ratios)
    return {
        "bertscore_f1": np.mean(bertscore.compute(predictions=preds, references=refs, lang="en", device=device)["f1"]),
        "rougeL": rouge.compute(predictions=preds, references=refs)["rougeL"].item(),
        "bleu": bleu.compute(predictions=preds, references=refs)["bleu"],
        "sacrebleu": sacrebleu.compute(predictions=preds, references=refs)["score"],
        "meteor": meteor.compute(predictions=preds, references=refs)["meteor"].item(),
        "toxicity_avg": np.mean(toxicity.compute(predictions=preds)["toxicity"]),
        "length_ratio": avg_ratio
    }


class ValueNetwork(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, encoder_hidden):
        pooled = encoder_hidden.mean(dim=1)
        return self.model(pooled).squeeze(-1)


class DROTrainer:
    def __init__(self, model, ref_model, value_net, train_loader, val_loader, tokenizer, cfg, device):
        self.model = model
        self.ref_model = ref_model
        self.value_net = value_net
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.optimizer = AdamW(list(self.model.parameters()) + list(self.value_net.parameters()), lr=cfg["lr"])

        total_steps = len(train_loader) // cfg["gradient_accumulation_steps"] * cfg["epochs"]
        self.scheduler = get_scheduler(
            "linear", optimizer=self.optimizer,
            num_warmup_steps=150, num_training_steps=total_steps
        )

    def compute_log_probs(self, logits, labels):
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        loss = loss_fn(flat_logits, flat_labels).view(labels.shape)
        mask = (labels != -100)
        token_count = mask.sum(dim=1).clamp(min=1)
        return - (loss * mask).sum(dim=1) / token_count

    def train(self):
        self.model.train()
        self.value_net.train()

        for epoch in range(self.cfg["epochs"]):
            total_loss = 0
            total_samples = 0
            for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")):
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                rewards = batch["reward"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                log_probs = self.compute_log_probs(outputs.logits, labels)

                with torch.no_grad():
                    ref_out = self.ref_model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                    log_probs_ref = self.compute_log_probs(ref_out.logits, labels)

                enc_hidden = self.model.encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
                value_pred = self.value_net(enc_hidden)

                residual = rewards - value_pred - self.cfg["tau"] * (log_probs - log_probs_ref)
                loss = 0.5 * (residual ** 2).mean() / self.cfg["gradient_accumulation_steps"]
                loss.backward()

                total_loss += loss.item()
                total_samples += len(batch["input_ids"])

                if (step + 1) % self.cfg["gradient_accumulation_steps"] == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            print(f"Train Loss: {total_loss / total_samples:.4f}")
            self.evaluate()

    def evaluate(self):
        self.model.eval()
        self.value_net.eval()
        predictions, references, instructions = [], [], []

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)

            with torch.no_grad():
                gen_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.cfg["max_new_tokens"],
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )
                decoded = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                predictions.extend(decoded)
                references.extend(batch["response"])
                instructions.extend(batch["instruction"])

        metrics = compute_metrics(predictions, references, self.device)
        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        df = pd.DataFrame({"Instruction": instructions, self.cfg["output_column"]: predictions})
        df.to_csv(f"{self.cfg['saved_model_name']}_predictions.csv", index=False)


def train_dro(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(cfg["base_model"])
    model = T5ForConditionalGeneration.from_pretrained(cfg["base_model"]).to(device)
    ref_model = T5ForConditionalGeneration.from_pretrained(cfg["base_model"]).to(device)
    for param in ref_model.parameters():
        param.requires_grad = False

    value_net = ValueNetwork(cfg["hidden_dim"]).to(device)

    train_data = load_dataset(cfg["dataset"], split="train")
    val_data = load_dataset(cfg["dataset"], split="validation")

    rewards = np.array(train_data["reward"])
    mean_r, std_r = rewards.mean(), rewards.std()
    normalize = lambda ex: {**ex, "reward": (ex["reward"] - mean_r) / (std_r + 1e-8)}
    train_data = train_data.map(normalize)
    val_data = val_data.map(normalize)

    train_data = tokenize_dataset(train_data, tokenizer, cfg["max_len"])
    val_data = tokenize_dataset(val_data, tokenizer, cfg["max_len"])

    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_data, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_batch)

    trainer = DROTrainer(model, ref_model, value_net, train_loader, val_loader, tokenizer, cfg, device)
    trainer.train()


if __name__ == "__main__":
    import torch
    import numpy as np

    torch.manual_seed(42)
    np.random.seed(42)

    dro_configs = {
        "small": {
            "base_model": "google/flan-t5-small",
            "hidden_dim": 512,
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "max_len": 512,
            "save_per_steps": 2500,
            "output_column": "DRO SMALL",
            "saved_model_name": "flan-t5-small-dro"
        },
        "large": {
            "base_model": "google/flan-t5-large",
            "hidden_dim": 1024,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "max_len": 512,
            "save_per_steps": 10000,
            "output_column": "DRO LARGE",
            "saved_model_name": "flan-t5-large-dro"
        },
        "xl": {
            "base_model": "google/flan-t5-xl",
            "hidden_dim": 2048,
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "max_len": 256,
            "save_per_steps": 20000,
            "output_column": "DRO XL",
            "saved_model_name": "flan-t5-xl-dro"
        },
    }

    # Choose model size: "small", "large", or "xl"
    model_size = "small"

    cfg = {
        **dro_configs[model_size],
        "lr": 1e-4,
        "epochs": 1,
        "tau": 1.0,
        "max_new_tokens": 768,
        "dataset": "bilalfaye/ultrafeedback-rpo-split"
    }

    train_dro(cfg)

