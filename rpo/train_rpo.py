import torch
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader
from collections import defaultdict
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_scheduler
from evaluate import load
from tqdm import tqdm
import pandas as pd
import os

# Load metrics
bertscore = load("bertscore")
rouge = load("rouge")
bleu = load("bleu")
sacrebleu = load("sacrebleu")
meteor = load("meteor")
toxicity = load("toxicity", module_type="measurement")


def custom_collate_fn(batch):
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long),
        "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long),
        "reward": torch.tensor([item["reward"] for item in batch], dtype=torch.float),
        "instruction": [item["instruction"] for item in batch],
        "response": [item["response"] for item in batch],
    }


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


def build_prompt_index(dataset):
    prompt_to_indices = defaultdict(list)
    for i, item in enumerate(dataset):
        prompt_to_indices[item["instruction"]].append(i)
    return prompt_to_indices


def compute_log_probs(model, input_ids, attention_mask, labels, requires_grad=False):
    with torch.set_grad_enabled(requires_grad):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits
    vocab_size = logits.size(-1)
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    logits = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    loss_per_token = loss_fct(logits, labels_flat).view(labels.shape)
    mask = (labels != -100)
    log_probs = - (loss_per_token * mask).sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1)
    return log_probs / token_counts


def compute_all_metrics(preds, refs, device):
    length_ratios = [len(p.split()) / max(1, len(r.split())) for p, r in zip(preds, refs)]
    avg_length_ratio = sum(length_ratios) / len(length_ratios)
    return {
        "bertscore_f1": sum(bertscore.compute(predictions=preds, references=refs, lang="en", device=device)["f1"]) / len(preds),
        "rougeL": rouge.compute(predictions=preds, references=refs)["rougeL"].item(),
        "bleu": bleu.compute(predictions=preds, references=refs)["bleu"],
        "sacrebleu": sacrebleu.compute(predictions=preds, references=refs)["score"],
        "meteor": meteor.compute(predictions=preds, references=refs)["meteor"].item(),
        "toxicity_avg": sum(toxicity.compute(predictions=preds)["toxicity"]) / len(preds),
        "length_ratio": avg_length_ratio
    }


class RPOLoss:
    def __init__(self, ref_model, dataset, prompt_to_indices, tau=0.7, alpha=0.05, device="cuda"):
        self.ref_model = ref_model
        self.dataset = dataset
        self.prompt_to_indices = prompt_to_indices
        self.tau = tau
        self.device = device
        self.v_hat_by_prompt = self._precompute_v_hat()
        self.alpha = alpha

    def _precompute_v_hat(self):
        v_hat = {}
        for prompt in tqdm(self.prompt_to_indices.keys(), desc="Computing V_hat"):
            z_sum = 0.0
            for idx in self.prompt_to_indices[prompt]:
                ex = self.dataset[idx]
                inp = torch.tensor(ex["input_ids"]).unsqueeze(0).to(self.device)
                att = torch.tensor(ex["attention_mask"]).unsqueeze(0).to(self.device)
                tgt = torch.tensor(ex["labels"]).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    log_prob = self._compute_log_prob(self.ref_model, inp, att, tgt)
                    pi_ref = torch.exp(log_prob)
                r = torch.tensor(ex["reward"]).to(self.device)
                z_sum += pi_ref * torch.exp(r / self.tau)
            v = self.tau * torch.log(z_sum)
            v_hat[prompt] = v.detach()
        return v_hat

    def compute_loss(self, rewards, prompts, log_pi_theta, log_pi_ref):
        V_hat = torch.stack([self.v_hat_by_prompt[prompt] for prompt in prompts])
        reg_target = (rewards - V_hat) / self.tau
        return 0.5 * ((log_pi_theta - log_pi_ref - reg_target) ** 2).mean()

    def _compute_log_prob(self, model, input_ids, attention_mask, labels):
        return compute_log_probs(model, input_ids, attention_mask, labels)


class RPOTrainer:
    def __init__(self, model, ref_model, rpo_loss, train_dl, val_dl, tokenizer, cfg, device):
        self.model = model
        self.ref_model = ref_model
        self.loss_fn = rpo_loss
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=cfg["lr"])
        steps_per_epoch = len(train_dl) // cfg["gradient_accumulation_steps"]
        total_steps = steps_per_epoch * cfg["epochs"]
        self.lr_scheduler = get_scheduler("linear", self.optimizer, 150, total_steps)

    def train(self):
        for epoch in range(self.cfg["epochs"]):
            self.model.train()
            for step, batch in enumerate(tqdm(self.train_dl, desc=f"Epoch {epoch + 1}")):
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                log_pi_theta = compute_log_probs(self.model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"], requires_grad=True)
                log_pi_ref = compute_log_probs(self.ref_model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"])
                loss = self.loss_fn.compute_loss(inputs["reward"], inputs["instruction"], log_pi_theta, log_pi_ref)
                loss = loss / self.cfg["gradient_accumulation_steps"]
                loss.backward()
                if (step + 1) % self.cfg["gradient_accumulation_steps"] == 0:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                if (step + 1) % self.cfg["save_per_steps"] == 0:
                    self.save_model(step + 1)
                    self.evaluate(step + 1)

    def evaluate(self, step):
        self.model.eval()
        preds, refs, prompts = [], [], []
        with torch.no_grad():
            for batch in tqdm(self.val_dl, desc="Evaluating"):
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output_ids = self.model.generate(inputs["input_ids"], max_new_tokens=self.cfg["max_new_tokens"],
                                                 do_sample=True, temperature=0.7, top_k=50, top_p=0.95,
                                                 repetition_penalty=1.2, no_repeat_ngram_size=3)
                decoded_preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                preds.extend(decoded_preds)
                refs.extend(inputs["response"])
                prompts.extend(inputs["instruction"])
        metrics = compute_all_metrics(preds, refs, self.device)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        df = pd.DataFrame({"Instruction": prompts, self.cfg["csv_name"]: preds})
        df.to_csv(f"{self.cfg['saved_model_name']}-step-{step}.csv", index=False)

    def save_model(self, step):
        path = f"checkpoints/{self.cfg['saved_model_name']}-step-{step}"
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def train_rpo(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(cfg["base_model"])
    model = T5ForConditionalGeneration.from_pretrained(cfg["base_model"]).to(device)
    ref_model = T5ForConditionalGeneration.from_pretrained(cfg["base_model"]).to(device)

    train_ds = load_dataset(cfg["dataset"], split="train")
    val_ds = load_dataset(cfg["dataset"], split="validation")

    r_mean, r_std = np.mean(train_ds["reward"]), np.std(train_ds["reward"])
    normalize_reward = lambda x: {**x, "reward": (x["reward"] - r_mean) / (r_std + 1e-8)}

    train_ds = train_ds.map(normalize_reward)
    val_ds = val_ds.map(normalize_reward)

    train_ds = tokenize_dataset(train_ds, tokenizer, cfg["max_len"])
    val_ds = tokenize_dataset(val_ds, tokenizer, cfg["max_len"])

    prompt_index = build_prompt_index(train_ds)
    rpo_loss = RPOLoss(ref_model, train_ds, prompt_index, tau=cfg["tau"], device=device)

    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=custom_collate_fn)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=custom_collate_fn)

    trainer = RPOTrainer(model, ref_model, rpo_loss, train_dl, val_dl, tokenizer, cfg, device)
    trainer.train()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    model_configs = {
        "small": {
            "base_model": "google/flan-t5-small",
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "max_len": 512,
            "save_per_steps": 2500,
            "csv_name": "RPO Small",
            "saved_model_name": "flan-t5-small-rpo"
        },
        "large": {
            "base_model": "google/flan-t5-large",
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "max_len": 512,
            "save_per_steps": 10000,
            "csv_name": "RPO Large",
            "saved_model_name": "flan-t5-large-rpo"
        },
        "xl": {
            "base_model": "google/flan-t5-xl",
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "max_len": 256,
            "save_per_steps": 20000,
            "csv_name": "RPO XL",
            "saved_model_name": "flan-t5-xl-rpo"
        },
    }

    selected_model = "large"  # Change to "small" or "xl" to run those configurations
    cfg = {
        **model_configs[selected_model],
        "lr": 1e-4,
        "epochs": 1,
        "max_new_tokens": 768,
        "tau": 5.0,
        "dataset": "bilalfaye/ultrafeedback-rpo-split"
    }

    train_rpo(cfg)