"""Knowledge base: load markdown docs, chunk them, embed them, search them.

Chunks are the "## " sections of each markdown file, prefixed with the
document title so each chunk makes sense on its own. Vectors live in an
in-memory FAISS inner-product index; embeddings are L2-normalised, so the
score returned is cosine similarity (roughly 0.0 = unrelated, 1.0 = identical).
"""

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_WORDS = 180  # split very long sections so one chunk stays focused


def load_documents(kb_dir=KB_DIR):
    """Return [(filename, text)] for every .md file in the KB folder."""
    return [(p.name, p.read_text(encoding="utf-8")) for p in sorted(Path(kb_dir).glob("*.md"))]


def chunk_document(source, text):
    """Split one markdown doc into chunks of {"source", "heading", "text"}."""
    lines = text.strip().splitlines()
    title = lines[0].lstrip("# ").strip() if lines else source

    sections, heading, buf = [], "Overview", []
    for line in lines[1:]:
        if line.startswith("## "):
            if buf:
                sections.append((heading, buf))
            heading, buf = line[3:].strip(), []
        elif line.strip():
            buf.append(line)
    if buf:
        sections.append((heading, buf))

    chunks = []
    for heading, body in sections:
        words = " ".join(body).split()
        if len(words) <= MAX_WORDS:
            pieces = ["\n".join(body)]  # keep line breaks so numbered steps stay readable
        else:
            pieces = [" ".join(words[i:i + MAX_WORDS]) for i in range(0, len(words), MAX_WORDS)]
        for piece in pieces:
            chunks.append({"source": source, "heading": heading, "text": f"{title} - {heading}\n{piece}"})
    return chunks


class KnowledgeBase:
    def __init__(self, kb_dir=KB_DIR, model_name=EMBED_MODEL):
        self.model = SentenceTransformer(model_name)
        self.chunks = [c for src, text in load_documents(kb_dir) for c in chunk_document(src, text)]
        vectors = self._embed([c["text"] for c in self.chunks])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def _embed(self, texts):
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    def retrieve(self, query, k=4):
        """Return the top-k chunks as [{"text", "source", "heading", "score"}], best first."""
        scores, ids = self.index.search(self._embed([query]), k)
        return [
            {**self.chunks[i], "score": round(float(s), 4)}
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]


_default_kb = None


def retrieve(query, k=4):
    """Module-level helper that lazily builds one shared KnowledgeBase."""
    global _default_kb
    if _default_kb is None:
        _default_kb = KnowledgeBase()
    return _default_kb.retrieve(query, k)
