"""Code embeddings of the corpus programs, for the code-similarity study (similar.py). Runs locally on the CPU; the
code is not sent anywhere.

    ~/environments/emb/bin/python paper_rss/similar_embed.py [model ...]   ->  $OUT/emb_<name>.npz

Each program's code (similar_src.text) is cut into windows of WIN tokens of the model's tokenizer, every window is
embedded as a document (no query prefix), and the program's embedding is the mean of its windows' unit embeddings,
scaled to unit length. Models (open weights, code retrieval; architectures that transformers ships, so no code
from the model's repository runs):
  qodo      Qodo/Qodo-Embed-1-1.5B (Qwen2-based, 1.5B parameters)
  bgecode   BAAI/bge-code-v1 (Qwen2-based, 1.5B)
  gte       Alibaba-NLP/gte-modernbert-base (ModernBERT, 149M; general text and code retrieval)
  qwen3     Qwen/Qwen3-Embedding-8B (Qwen3-based, 8B; fp32 on this CPU)
A model named <name>:mem embeds the program's memory code alone (similar_src.mem_slice) -> emb_<name>_mem.npz.
"""
import os
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import similar_src as S  # noqa: E402

OUT = os.path.expanduser("~/memprint-data/2026-10-29-rsssim")
WIN = 512
MODELS = {"qodo": "Qodo/Qodo-Embed-1-1.5B", "bgecode": "BAAI/bge-code-v1",
          "gte": "Alibaba-NLP/gte-modernbert-base", "qwen3": "Qwen/Qwen3-Embedding-8B"}


def windows(tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    step = WIN - 16   # room for the special tokens
    return [tok.decode(ids[i:i + step]) for i in range(0, max(len(ids), 1), step)]


def main(names):
    torch.set_num_threads(os.cpu_count())
    os.makedirs(OUT, exist_ok=True)
    for arg in names:
        name, mem = arg.split(":")[0], arg.endswith(":mem")
        code = S.mem_slice if mem else S.text
        m = SentenceTransformer(MODELS[name], trust_remote_code=False, device="cpu")
        m.max_seq_length = WIN
        progs, embs, nwin, t0 = [], [], [], time.time()
        for p in S.PROGRAMS:
            w = windows(m.tokenizer, code(p))
            e = m.encode(w, batch_size=8, normalize_embeddings=True, convert_to_numpy=True)
            v = e.mean(0)
            progs.append(p); embs.append(v / np.linalg.norm(v)); nwin.append(len(w))
            print(f"{arg} {p}: {len(w)} windows, {time.time() - t0:.0f} s", flush=True)
        np.savez(f"{OUT}/emb_{name}{'_mem' if mem else ''}.npz", progs=np.array(progs), emb=np.array(embs), windows=np.array(nwin),
                 model=MODELS[name], seconds=time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
