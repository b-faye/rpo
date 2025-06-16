# 🧠 Reward Partitioning Optimization (RPO)

**Reward Partitioning Optimization (RPO)** is a simple and robust method for **value-free policy optimization** in **scalar-feedback RLHF**. Unlike methods like Direct Preference Optimization (DPO) that require preference pairs, RPO only needs triplets of *(prompt, response, reward)* — making it practical for real-world human feedback (e.g., thumbs-up/down).

RPO avoids value function modeling by directly partitioning and normalizing scalar rewards to supervise the policy. It's stable, efficient, and easy to implement.

> 🚀 RPO **outperforms DRO** and **KTO** across automatic metrics and LLM-based preference evaluation.

---

## 📦 Installation

```bash
# Clone the repository
git clone git@github.com:b-faye/rpo.git
cd rpo

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> ✅ Training was conducted on a single NVIDIA A100 GPU (80GB).

---

## 📊 Dataset

RPO requires a dataset of scalar-feedback triplets:

```json
[
  {
    "prompt": "Instruction goes here",
    "responses": [
      {"text": "Response A", "reward": 3.2},
      {"text": "Response B", "reward": -1.0},
      {"text": "Response C", "reward": 7.5}
    ]
  },
  ...
]
```

Each prompt is paired with multiple *(response, reward)* entries.
We use the **UltraFeedback** dataset from [Cui et al., 2023](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized).

---

## 🏋️ Training

We provide training scripts for three methods:

```bash
# Train RPO
python3 rpo/train_rpo.py

# Train DRO
python3 dro/train_dro.py

# Train KTO
python3 kto/train_kto.py
```

All models are initialized from **Flan-T5** variants (Small, Large, XL) with instruction tuning.

---

## 🔮 Inference

Use the following snippet to generate responses with your trained RPO model:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained("bilalfaye/flan-t5-large-rpo").to(device)
tokenizer = AutoTokenizer.from_pretrained("bilalfaye/flan-t5-large-rpo")

prompt = "Give an example of a situation where honesty is not the best policy."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    input_ids=inputs["input_ids"],
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
)

response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(response)
```

Available models:

* [`bilalfaye/flan-t5-small-rpo`](https://huggingface.co/bilalfaye/flan-t5-small-rpo)
* [`bilalfaye/flan-t5-large-rpo`](https://huggingface.co/bilalfaye/flan-t5-large-rpo)
* [`bilalfaye/flan-t5-xl-rpo`](https://huggingface.co/bilalfaye/flan-t5-xl-rpo)

---

## ✨ Qualitative Results

**Prompt**: *"Explain why democracy is important in one paragraph."*

| Method     | Output                                                                                                                                                                      |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RPO** ✅  | Democracy empowers citizens, ensures accountability, and fosters equal representation by allowing individuals to participate in governance and voice their opinions freely. |
| **DRO** ⚠️ | Democracy allows people to vote and participate, which is helpful, though sometimes inefficient in large societies.                                                         |
| **KTO** ❌  | Democracy is good. People vote. Leaders get chosen.                                                                                                                         |

RPO outputs are **more fluent, aligned, and helpful**, showing better semantic richness and preference satisfaction.

---

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{faye2024rpo,
  title={Value-Free Policy Optimization via Reward Partitioning},
  author={Bilal Faye and Hanane Azzag and Mustapha Lebbah},
  year={2024},
  journal={arXiv preprint},
}
```


