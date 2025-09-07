# GPT2ARTMap

Perfect 🚀 — let’s wrap the **ART-regularized GPT-2 training** into a Hugging Face `Trainer` subclass so you can use it like a normal fine-tuning job but with ART feedback built in.

---

# 🔹 Step 1: ART module (same as before)

```python
import torch
import torch.nn.functional as F

class SimpleART:
    def __init__(self, vigilance=0.75, device="cpu"):
        self.vigilance = vigilance
        self.prototypes = []
        self.device = device

    def _similarity(self, x, y):
        return F.cosine_similarity(x, y, dim=-1)

    def update(self, embedding):
        if not self.prototypes:
            self.prototypes.append(embedding.detach().clone())
            return 0, "new"

        sims = torch.cat([self._similarity(embedding, p).unsqueeze(0) for p in self.prototypes])
        best_idx = torch.argmax(sims).item()
        best_sim = sims[best_idx].item()

        if best_sim >= self.vigilance:
            # resonance → update prototype
            self.prototypes[best_idx] = 0.5 * (self.prototypes[best_idx] + embedding)
            return best_idx, "resonated"
        else:
            # create new cluster
            self.prototypes.append(embedding.detach().clone())
            return len(self.prototypes)-1, "new"

def art_loss_fn(embedding, art, device="cpu"):
    if not art.prototypes:
        art.prototypes.append(embedding.detach().clone())
        return torch.tensor(0.0, device=device)

    sims = torch.cat([F.cosine_similarity(embedding, p).unsqueeze(0) for p in art.prototypes])
    best_idx = torch.argmax(sims).item()
    best_sim = sims[best_idx].item()

    if best_sim >= art.vigilance:
        proto = art.prototypes[best_idx].detach()
        return 1 - F.cosine_similarity(embedding, proto).mean()
    else:
        return torch.tensor(0.1, device=device)
```

---

# 🔹 Step 2: Custom Trainer with ART Loss

```python
from transformers import Trainer

class ARTTrainer(Trainer):
    def __init__(self, *args, art_module=None, lambda_art=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.art = art_module
        self.lambda_art = lambda_art

    def compute_loss(self, model, inputs, return_outputs=False):
        # Hugging Face Trainer passes `inputs` with labels
        labels = inputs.get("labels")
        outputs = model(**inputs, output_hidden_states=True)
        loss_ce = outputs.loss

        # mean hidden states
        emb = outputs.hidden_states[-1].mean(dim=1)
        loss_art = art_loss_fn(emb, self.art, device=emb.device)

        loss = loss_ce + self.lambda_art * loss_art

        # update ART clusters (no gradient)
        self.art.update(emb.detach())

        return (loss, outputs) if return_outputs else loss
```

---

# 🔹 Step 3: Usage Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset

# Load data
dataset = load_dataset("imdb", split="train[:1%]")  # small subset for demo
dataset = dataset.map(lambda e: {"labels": e["text"]})  # labels placeholder

# Model + tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize, batched=True)

# Init ART
art = SimpleART(vigilance=0.8, device="cuda")

# Training args
args = TrainingArguments(
    output_dir="./art-gpt2",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    learning_rate=5e-5,
)

# Trainer
trainer = ARTTrainer(
    model=AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to("cuda"),
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    art_module=art,
    lambda_art=0.1,
)

trainer.train()
```

