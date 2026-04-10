#!/usr/bin/env python3
"""
Code Search Web Server — side-by-side comparison of Pretrained vs RL-Trained CodeBERT.
Supports adding / soft-deleting corpus snippets, persisted across restarts.

Persistent files (in corpus_dir):
  user_snippets.json   — snippets added via the web UI
  deleted_ids.json     — snippet_ids that have been soft-deleted

Usage:
    python scripts/search_server.py --checkpoint outputs/checkpoints/step_3000.pt
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import json
import uuid
import argparse
import threading
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.config import load_config
from src.data.corpus_builder import load_corpus_metadata
from src.data.schema import CodeSnippet
from src.retriever.encoder import CodeBERTEncoder
from src.retriever.faiss_index import FaissIndex
from src.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SnippetResult(BaseModel):
    rank: int
    score: float
    docstring: str
    code: str
    snippet_id: str
    is_user: bool = False


class SearchResponse(BaseModel):
    pretrained: List[SnippetResult]
    rl_trained: List[SnippetResult]


class StatusResponse(BaseModel):
    pretrained_ready: bool
    rl_ready: bool
    corpus_size: int        # active (non-deleted) snippets
    rl_step: int
    user_count: int         # user-added active snippets


class AddSnippetRequest(BaseModel):
    code: str
    label: str = ""         # optional display label


class UserSnippetInfo(BaseModel):
    snippet_id: str
    label: str
    code_preview: str


# ---------------------------------------------------------------------------
# Model / corpus state
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.encoder_pre: Optional[CodeBERTEncoder] = None
        self.encoder_rl:  Optional[CodeBERTEncoder] = None
        self.index_pre:   Optional[FaissIndex] = None
        self.index_rl:    Optional[FaissIndex] = None

        # Ordered list — FAISS vector position i == corpus_snippets[i]
        self.corpus_snippets: List[CodeSnippet] = []

        self.deleted_ids:     Set[str] = set()
        self.user_snippet_ids: Set[str] = set()

        self.rl_step:   int = 0
        self.device = torch.device("cpu")
        self.corpus_dir: str = ""
        self.embedding_dim: int = 768

        self.pretrained_ready = False
        self.rl_ready = False

        self.lock = threading.Lock()   # guards corpus mutations


state = AppState()
app   = FastAPI(title="Adaptive Code RAG — Search Server")


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _deleted_path() -> Path:
    return Path(state.corpus_dir) / "deleted_ids.json"

def _user_path() -> Path:
    return Path(state.corpus_dir) / "user_snippets.json"

def _load_deleted_ids() -> Set[str]:
    p = _deleted_path()
    return set(json.loads(p.read_text())) if p.exists() else set()

def _save_deleted_ids() -> None:
    _deleted_path().write_text(json.dumps(sorted(state.deleted_ids)))

def _load_user_snippets() -> List[CodeSnippet]:
    p = _user_path()
    if not p.exists():
        return []
    return [CodeSnippet(**d) for d in json.loads(p.read_text())]

def _save_user_snippets() -> None:
    snippets = [s for s in state.corpus_snippets
                if s.snippet_id in state.user_snippet_ids]
    data = [{"snippet_id": s.snippet_id, "code": s.code,
              "docstring": s.docstring, "language": s.language,
              "corpus_idx": s.corpus_idx}
            for s in snippets]
    _user_path().write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Index build helpers
# ---------------------------------------------------------------------------

def _snippet_text(s: CodeSnippet) -> str:
    """Text fed to encoder for corpus indexing."""
    prefix = f"{s.docstring} " if s.docstring else ""
    return (prefix + s.code)[:512]

def _build_index(encoder: CodeBERTEncoder, snippets: List[CodeSnippet]) -> FaissIndex:
    texts = [_snippet_text(s) for s in snippets]
    batch_size = 64
    all_embs = []
    for i in range(0, len(texts), batch_size):
        embs = encoder.encode_corpus_batch(texts[i:i+batch_size], device=state.device)
        all_embs.append(embs)
    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)
    idx = FaissIndex(embedding_dim=state.embedding_dim)
    idx.build(embeddings)
    return idx


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def load_models(config_path: str, checkpoint_path: str) -> None:
    """Runs in a background thread. Populates state.*."""
    config = load_config(config_path)
    state.corpus_dir   = config.data.corpus_dir
    state.embedding_dim = config.retriever.embedding_dim

    if torch.cuda.is_available():
        state.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        state.device = torch.device("mps")
    else:
        state.device = torch.device("cpu")
    print(f"[startup] Device: {state.device}")

    # --- Corpus ---
    print("[startup] Loading corpus metadata...")
    original = load_corpus_metadata(config.data.corpus_dir)

    state.deleted_ids = _load_deleted_ids()
    user_snippets     = _load_user_snippets()

    state.corpus_snippets  = original + user_snippets
    state.user_snippet_ids = {s.snippet_id for s in user_snippets}

    active = sum(1 for s in state.corpus_snippets
                 if s.snippet_id not in state.deleted_ids)
    print(f"[startup] Corpus: {len(original)} original + {len(user_snippets)} user "
          f"= {len(state.corpus_snippets)} total, {active} active")

    # --- Pretrained encoder ---
    print("[startup] Loading pretrained CodeBERT...")
    state.encoder_pre = CodeBERTEncoder(
        model_name=config.retriever.model_name,
        max_seq_len=config.retriever.max_seq_len,
    ).to(state.device)
    state.encoder_pre.eval()
    print("[startup] Building pretrained FAISS index...")
    state.index_pre = _build_index(state.encoder_pre, state.corpus_snippets)
    state.pretrained_ready = True
    print("[startup] Pretrained index ready.")

    # --- RL-trained encoder ---
    print(f"[startup] Loading RL checkpoint: {checkpoint_path}")
    state.encoder_rl = CodeBERTEncoder(
        model_name=config.retriever.model_name,
        max_seq_len=config.retriever.max_seq_len,
    ).to(state.device)
    state.rl_step = load_checkpoint(checkpoint_path, state.encoder_rl)
    state.encoder_rl.eval()
    print(f"[startup] Building RL-trained FAISS index (step={state.rl_step})...")
    state.index_rl = _build_index(state.encoder_rl, state.corpus_snippets)
    state.rl_ready = True
    print("[startup] All indices ready. Server fully operational.")


# ---------------------------------------------------------------------------
# Search helper
# ---------------------------------------------------------------------------

def _search(encoder: CodeBERTEncoder, index: FaissIndex,
            query: str, top_k: int) -> List[SnippetResult]:
    with torch.no_grad():
        q_emb = encoder.encode_query(query, device=state.device)
        q_np  = q_emb.detach().cpu().numpy().astype(np.float32)

    # Fetch extra candidates to absorb soft-deleted slots
    search_k = min(top_k + len(state.deleted_ids) + 20, index.ntotal)
    scores, indices = index.search(q_np, top_k=search_k)

    results, rank = [], 1
    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        snippet = state.corpus_snippets[idx]
        if snippet.snippet_id in state.deleted_ids:
            continue
        results.append(SnippetResult(
            rank=rank,
            score=float(score),
            docstring=snippet.docstring,
            code=snippet.code,
            snippet_id=snippet.snippet_id,
            is_user=(snippet.snippet_id in state.user_snippet_ids),
        ))
        rank += 1
        if rank > top_k:
            break

    return results


# ---------------------------------------------------------------------------
# Routes — search
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_index():
    html_path = Path(__file__).parent.parent / "web" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>web/index.html not found</h1>", status_code=404)


@app.get("/status", response_model=StatusResponse)
def get_status():
    active = sum(1 for s in state.corpus_snippets
                 if s.snippet_id not in state.deleted_ids)
    user_active = sum(1 for sid in state.user_snippet_ids
                      if sid not in state.deleted_ids)
    return StatusResponse(
        pretrained_ready=state.pretrained_ready,
        rl_ready=state.rl_ready,
        corpus_size=active,
        rl_step=state.rl_step,
        user_count=user_active,
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    top_k = max(1, min(req.top_k, 10))
    pre_results, rl_results = [], []
    if state.pretrained_ready:
        pre_results = _search(state.encoder_pre, state.index_pre, req.query, top_k)
    if state.rl_ready:
        rl_results  = _search(state.encoder_rl,  state.index_rl,  req.query, top_k)
    return SearchResponse(pretrained=pre_results, rl_trained=rl_results)


# ---------------------------------------------------------------------------
# Routes — corpus management
# ---------------------------------------------------------------------------

@app.get("/corpus/list")
def list_user_snippets():
    """Return all user-added snippets (including deleted ones, with flag)."""
    with state.lock:
        items = []
        for s in state.corpus_snippets:
            if s.snippet_id not in state.user_snippet_ids:
                continue
            items.append({
                "snippet_id": s.snippet_id,
                "label":      s.docstring,
                "code_preview": s.code[:120],
                "deleted":    s.snippet_id in state.deleted_ids,
            })
    return {"snippets": items}


@app.post("/corpus/add")
def add_snippet(req: AddSnippetRequest):
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="code must not be empty")
    if not state.pretrained_ready or not state.rl_ready:
        raise HTTPException(status_code=503, detail="Models still loading")

    with state.lock:
        snippet_id  = f"user_{uuid.uuid4().hex[:10]}"
        corpus_idx  = len(state.corpus_snippets)

        snippet = CodeSnippet(
            snippet_id=snippet_id,
            code=req.code,
            docstring=req.label,
            language="python",
            corpus_idx=corpus_idx,
        )

        text = _snippet_text(snippet)

        with torch.no_grad():
            emb_pre = state.encoder_pre.encode_corpus_batch([text], device=state.device).astype(np.float32)
            emb_rl  = state.encoder_rl.encode_corpus_batch([text],  device=state.device).astype(np.float32)

        state.index_pre.index.add(emb_pre)
        state.index_rl.index.add(emb_rl)

        state.corpus_snippets.append(snippet)
        state.user_snippet_ids.add(snippet_id)

        _save_user_snippets()

    return {"snippet_id": snippet_id, "message": "Added successfully"}


@app.delete("/corpus/{snippet_id}")
def delete_snippet(snippet_id: str):
    with state.lock:
        if not any(s.snippet_id == snippet_id for s in state.corpus_snippets):
            raise HTTPException(status_code=404, detail="Snippet not found")
        if snippet_id in state.deleted_ids:
            raise HTTPException(status_code=409, detail="Already deleted")

        state.deleted_ids.add(snippet_id)
        _save_deleted_ids()

        # If user snippet, remove from user_snippets.json too
        if snippet_id in state.user_snippet_ids:
            state.user_snippet_ids.discard(snippet_id)
            _save_user_snippets()

    return {"message": "Deleted successfully"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Adaptive Code RAG — Search Server")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/step_3000.pt")
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--host",       default="127.0.0.1")
    args = parser.parse_args()

    t = threading.Thread(
        target=load_models, args=(args.config, args.checkpoint), daemon=True
    )
    t.start()

    print(f"\nStarting server → http://{args.host}:{args.port}")
    print("Models loading in background — poll /status for readiness.\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
