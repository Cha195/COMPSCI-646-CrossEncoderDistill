import json
import math
import os
import random
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# Config
# =========================


@dataclass
class TrainConfig:
    train_path: str = "/Users/aneeshsathe/Desktop/tiny-experiments/qwendistill/kd_multihop_dev_set.jsonl"
    val_path: str | None = None

    output_dir: str = "/Users/aneeshsathe/Desktop/tiny-experiments/qwendistill/checkpoints/bge_student"

    student_name: str = "BAAI/bge-m3"
    teacher_name: str = "Qwen/Qwen3-Reranker-0.6B"

    batch_size: int = 8
    max_negs: int = 8
    max_q_len: int = 64
    max_d_len: int = 256

    lr: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    temperature: float = 0.07
    lambda_align: float = 0.5

    eval_every_steps: int = 500
    save_every_steps: int = 2000

    max_teacher_length: int = 2048

    seed: int = 42


# =========================
# Dataset
# =========================


class MultiHopBGETrainDataset(Dataset):
    """
    Expects JSONL with:
    {
      "query": str,
      "positive": str,
      "negatives": [str, ...]
    }
    """

    def __init__(self, jsonl_path: str, max_negs: int | None = None):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)

                # query
                query = ex["query"]

                # positive
                pos_raw = ex.get("positive")
                if isinstance(pos_raw, dict):
                    pos = pos_raw.get("paragraph", "").strip()
                else:
                    pos = str(pos_raw).strip()

                # collect negatives into a list[str] called negs
                negs: list[str] = []
                if "negatives" in ex:
                    raw_negs = ex["negatives"]
                    if isinstance(raw_negs, dict):
                        par = raw_negs.get("paragraph")
                        if isinstance(par, str):
                            negs.append(par.strip())
                    elif isinstance(raw_negs, list):
                        for n in raw_negs:
                            if isinstance(n, dict):
                                par = n.get("paragraph")
                                if isinstance(par, str):
                                    negs.append(par.strip())
                            else:
                                negs.append(str(n).strip())
                elif "hard_negatives" in ex:
                    raw_negs = ex["hard_negatives"]
                    for n in raw_negs:
                        if isinstance(n, dict):
                            par = n.get("paragraph")
                            if isinstance(par, str):
                                negs.append(par.strip())
                        else:
                            negs.append(str(n).strip())
                else:
                    continue

                negs = [n for n in negs if n]

                # Filter based on max_negs
                if max_negs is not None:
                    if len(negs) < max_negs:
                        # skip examples that don't have enough negatives
                        continue
                    negs = negs[:max_negs]

                if not pos or len(negs) == 0:
                    continue

                self.samples.append(
                    {"query": query, "positive": pos, "negatives": negs}
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class BGETrainCollator:
    def __init__(self, tokenizer, max_q_len: int = 64, max_d_len: int = 256):
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_d_len = max_d_len

    def __call__(self, batch):
        # raw texts
        queries_raw = [b["query"] for b in batch]
        positives_raw = [b["positive"] for b in batch]
        negatives_raw = [b["negatives"] for b in batch]

        docs_raw = [[b["positive"]] + b["negatives"] for b in batch]  # [B][1+M]

        # flatten docs
        flat_docs_raw = [d for docs in docs_raw for d in docs]

        # tokenize queries
        q_tok = self.tokenizer(
            queries_raw,
            padding=True,
            truncation=True,
            max_length=self.max_q_len,
            return_tensors="pt",
        )

        # tokenize docs
        d_tok = self.tokenizer(
            flat_docs_raw,
            padding=True,
            truncation=True,
            max_length=self.max_d_len,
            return_tensors="pt",
        )

        B = len(batch)
        num_docs = len(docs_raw[0])
        assert all(len(d) == num_docs for d in docs_raw), (
            "variable num_docs per example"
        )

        d_input_ids = d_tok["input_ids"].view(B, num_docs, -1)
        d_attention_mask = d_tok["attention_mask"].view(B, num_docs, -1)

        return {
            "q_input_ids": q_tok["input_ids"],
            "q_attention_mask": q_tok["attention_mask"],
            "d_input_ids": d_input_ids,
            "d_attention_mask": d_attention_mask,
            "queries_raw": queries_raw,
            "positives_raw": positives_raw,
            "negatives_raw": negatives_raw,
            "docs_raw": docs_raw,
        }


# =========================
# Student model (BGE)
# =========================


class BGEStudent(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        # SentenceTransformer wrapper for BGE-M3
        self.model = SentenceTransformer(model_name)
        # underlying HF transformer module
        self.encoder = self.model._first_module().auto_model

    def encode(self, input_ids, attention_mask):
        """
        input_ids: [B, L]
        attention_mask: [B, L]
        returns: [B, H] CLS-based, L2-normalized embeddings
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # CLS pooling
        emb = outputs.last_hidden_state[:, 0, :]  # [B, H]
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save(output_dir)


# =========================
# Teacher wrapper (Qwen reranker)
# =========================


class QwenReranker:
    def __init__(self, model_name: str, max_length: int = 2048, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.max_length = max_length

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_messages_batch(self, queries, docs):
        system_msg = (
            "Judge if the Document satisfies the Query given the Instruct. "
            'Answer with "yes" if it is relevant, otherwise "no".'
        )
        instruct = (
            "Given a web search query, judge whether the document answers the query."
        )
        messages_batch = []
        for q, d in zip(queries, docs):
            messages = [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": f"<Instruct>: {instruct}\n\n<Query>: {q}\n\n<Document>: {d}",
                },
            ]
            messages_batch.append(messages)
        return messages_batch

    @torch.no_grad()
    def score(self, queries, docs) -> torch.Tensor:
        """
        queries: list[str], docs: list[str], same length
        returns scores in [0,1] as prob("yes")
        """
        messages_batch = self._build_messages_batch(queries, docs)
        input_ids = self.tokenizer.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits  # [B, T, V]
        last_logits = logits[:, -1, :]  # [B, V]

        yes_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        no_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]

        yes_logit = last_logits[:, yes_id]
        no_logit = last_logits[:, no_id]

        stacked = torch.stack([no_logit, yes_logit], dim=-1)  # [B, 2]
        probs = torch.softmax(stacked, dim=-1)[:, 1]  # prob of yes
        return probs  # [B]


# =========================
# Loss functions
# =========================


def contrastive_infonce(
    scores: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """
    scores: [B, num_docs], doc 0 is positive
    """
    logits = scores / temperature
    targets = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    return nn.functional.cross_entropy(logits, targets)


def listwise_kl(
    scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    tau_T: float = 1.0,
    tau_S: float = 1.0,
) -> torch.Tensor:
    """
    scores: [B, num_docs] student
    teacher_scores: [B, num_docs] teacher
    """
    teacher_logit = teacher_scores / tau_T
    teacher_prob = nn.functional.softmax(teacher_logit, dim=-1)

    student_logit = scores / tau_S
    student_logprob = nn.functional.log_softmax(student_logit, dim=-1)

    return nn.functional.kl_div(student_logprob, teacher_prob, reduction="batchmean")


# =========================
# Trainer
# =========================


class DistillationTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._set_seed(cfg.seed)

        # student
        self.student = BGEStudent(cfg.student_name)
        # Move to device if needed
        if hasattr(self.student.model, "to"):
            self.student.model = self.student.model.to(self.device)

        # Get tokenizer from student model
        self.student_tokenizer = self.student.model.tokenizer

        # teacher
        self.teacher = QwenReranker(
            cfg.teacher_name,
            max_length=cfg.max_teacher_length,
            device=self.device,
        )

        # collator
        self.collator = BGETrainCollator(
            self.student_tokenizer,
            max_q_len=cfg.max_q_len,
            max_d_len=cfg.max_d_len,
        )

        # data
        self.train_dataset = MultiHopBGETrainDataset(
            cfg.train_path, max_negs=cfg.max_negs
        )
        self.val_dataset = (
            MultiHopBGETrainDataset(cfg.val_path, max_negs=cfg.max_negs)
            if cfg.val_path is not None
            else None
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=0,  # Changed from 4 to 0 to avoid fork issues
            pin_memory=False,  # Changed to False for MPS
        )

        self.val_loader = (
            DataLoader(
                self.val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=self.collator,
                num_workers=0,  # Changed from 4 to 0
                pin_memory=False,  # Changed to False for MPS
            )
            if self.val_dataset is not None
            else None
        )

        # optim and scheduler
        self.optimizer = AdamW(
            self.student.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        num_training_steps = len(self.train_loader) * cfg.num_epochs
        self.scheduler = self._build_scheduler(
            self.optimizer, num_training_steps, cfg.warmup_ratio
        )

        os.makedirs(cfg.output_dir, exist_ok=True)
        self.global_step = 0

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _build_scheduler(optimizer, num_training_steps: int, warmup_ratio: float):
        warmup_steps = int(num_training_steps * warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps and warmup_steps > 0:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(
                max(1, num_training_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _compute_teacher_scores(self, queries, positives, negatives) -> torch.Tensor:
        """
        queries: list[str] length B
        positives: list[str] length B
        negatives: list[list[str]] length B, each same length
        returns [B, num_docs] where num_docs = 1 + len(negatives per example)
        """
        B = len(queries)

        # Build all query-doc pairs
        all_queries = []
        all_docs = []

        for q, pos, negs in zip(queries, positives, negatives):
            # Add positive
            all_queries.append(q)
            all_docs.append(pos)
            # Add negatives
            for neg in negs:
                all_queries.append(q)
                all_docs.append(neg)

        # Get teacher scores
        scores_flat = self.teacher.score(all_queries, all_docs)  # [total_pairs]

        # Reshape back to [B, num_docs]
        num_docs = 1 + len(
            negatives[0]
        )  # assuming same number of negatives per example
        return scores_flat.view(B, num_docs)

    def training_step(self, batch):
        self.student.train()

        # token ids & masks
        q_ids = batch["q_input_ids"].to(self.device)
        q_attn = batch["q_attention_mask"].to(self.device)

        d_ids = batch["d_input_ids"].to(self.device)
        d_attn = batch["d_attention_mask"].to(self.device)

        # raw strings for teacher
        queries_raw = batch["queries_raw"]
        positives_raw = batch["positives_raw"]
        negatives_raw = batch["negatives_raw"]

        B, num_docs, L = d_ids.shape

        # ===== student forward =====
        # queries
        q_emb = self.student.encode(q_ids, q_attn)  # [B, H]

        # docs
        d_ids_flat = d_ids.view(B * num_docs, L)
        d_attn_flat = d_attn.view(B * num_docs, L)
        d_emb_flat = self.student.encode(d_ids_flat, d_attn_flat)  # [B*num_docs, H]
        d_emb = d_emb_flat.view(B, num_docs, -1)  # [B, num_docs, H]

        # student scores
        scores = torch.sum(q_emb.unsqueeze(1) * d_emb, dim=-1)  # [B, num_docs]

        # ===== teacher scores =====
        with torch.no_grad():
            teacher_scores = self._compute_teacher_scores(
                queries_raw, positives_raw, negatives_raw
            ).to(self.device)  # [B, num_docs]

        # ===== loss =====
        loss_infonce = contrastive_infonce(scores, temperature=self.cfg.temperature)
        loss_kl = listwise_kl(scores, teacher_scores)

        loss = loss_infonce + self.cfg.lambda_align * loss_kl

        return loss, loss_infonce.item(), loss_kl.item()

    @torch.no_grad()
    def evaluate(self):
        if self.val_loader is None:
            return None

        self.student.eval()
        total_loss = 0.0
        total_batches = 0

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            q_ids = batch["q_input_ids"].to(self.device)
            q_attn = batch["q_attention_mask"].to(self.device)

            d_ids = batch["d_input_ids"].to(self.device)
            d_attn = batch["d_attention_mask"].to(self.device)

            queries_raw = batch["queries_raw"]
            positives_raw = batch["positives_raw"]
            negatives_raw = batch["negatives_raw"]

            B, num_docs, L = d_ids.shape

            # student
            q_emb = self.student.encode(q_ids, q_attn)
            d_ids_flat = d_ids.view(B * num_docs, L)
            d_attn_flat = d_attn.view(B * num_docs, L)
            d_emb_flat = self.student.encode(d_ids_flat, d_attn_flat)
            d_emb = d_emb_flat.view(B, num_docs, -1)

            scores = torch.sum(q_emb.unsqueeze(1) * d_emb, dim=-1)

            # teacher
            teacher_scores = self._compute_teacher_scores(
                queries_raw, positives_raw, negatives_raw
            ).to(self.device)

            loss_infonce = contrastive_infonce(scores, temperature=self.cfg.temperature)
            loss_kl = listwise_kl(scores, teacher_scores)
            loss = loss_infonce + self.cfg.lambda_align * loss_kl

            total_loss += loss.item()
            total_batches += 1

        return total_loss / max(1, total_batches)

    def _save_checkpoint(self, tag: str):
        ckpt_dir = os.path.join(self.cfg.output_dir, tag)
        os.makedirs(ckpt_dir, exist_ok=True)

        # save student encoder in SentenceTransformer format
        self.student.save(os.path.join(ckpt_dir, "student"))

        state = {
            "global_step": self.global_step,
            "config": asdict(self.cfg),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(state, os.path.join(ckpt_dir, "trainer_state.pt"))
        print(f"Saved checkpoint to {ckpt_dir}")

    def train(self):
        best_val = None

        num_training_steps = len(self.train_loader) * self.cfg.num_epochs
        progress = tqdm(total=num_training_steps, desc="Training")

        for epoch in range(self.cfg.num_epochs):
            for batch in self.train_loader:
                loss, loss_infonce, loss_kl = self.training_step(batch)

                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.cfg.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()

                self.global_step += 1
                progress.update(1)
                progress.set_postfix(
                    epoch=epoch + 1,
                    step=self.global_step,
                    loss=f"{loss.item():.4f}",
                    infonce=f"{loss_infonce:.4f}",
                    kl=f"{loss_kl:.4f}",
                )

                # eval
                if (
                    self.cfg.eval_every_steps > 0
                    and self.global_step % self.cfg.eval_every_steps == 0
                ):
                    val_loss = self.evaluate()
                    if val_loss is not None:
                        print(f"\nStep {self.global_step} - val_loss: {val_loss:.4f}")
                        if best_val is None or val_loss < best_val:
                            best_val = val_loss
                            self._save_checkpoint("best")

                # periodic save
                if (
                    self.cfg.save_every_steps > 0
                    and self.global_step % self.cfg.save_every_steps == 0
                ):
                    self._save_checkpoint(f"step_{self.global_step}")

        # final save
        self._save_checkpoint("final")
        progress.close()
        print(
            f"\nTraining complete! Best val loss: {best_val:.4f}"
            if best_val
            else "\nTraining complete!"
        )


# =========================
# Main
# =========================


def main():
    cfg = TrainConfig()
    trainer = DistillationTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
