import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import numpy as np


class CodeBERTEncoder(nn.Module):
    """CodeBERT encoder that produces normalized embeddings. Supports gradient flow."""

    def __init__(self, model_name: str = "microsoft/codebert-base", max_seq_len: int = 512):
        super().__init__()
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts: List[str], device: torch.device = None, no_grad: bool = False) -> torch.Tensor:
        """Encode texts to normalized embeddings. Shape: [N, 768]."""
        if device is None:
            device = next(self.model.parameters()).device

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        ).to(device)

        if no_grad:
            with torch.no_grad():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)

        # CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]  # [N, 768]
        embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def encode_query(self, query: str, device: torch.device = None) -> torch.Tensor:
        """Encode single query WITH gradient. Shape: [768]."""
        emb = self.encode([query], device=device, no_grad=False)
        return emb[0]  # [768]

    def encode_corpus_batch(self, texts: List[str], device: torch.device = None) -> np.ndarray:
        """Encode corpus texts WITHOUT gradient. Returns numpy array."""
        emb = self.encode(texts, device=device, no_grad=True)
        return emb.cpu().numpy()
