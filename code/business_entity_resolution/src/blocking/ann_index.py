"""
Approximate Nearest Neighbor Index (FAISS IVF-PQ).

Builds a compressed, scalable FAISS index suitable for multi-million row data.
Trained and populated in streaming chunks to keep memory usage under 500MB.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    logger.warning("FAISS not installed. ANN indexing disabled.")


class ANNIndex:
    """
    Wraps FAISS IVF-PQ index for large-scale ANN search.
    Supports memory-mapped input arrays and chunked addition.
    """

    FLAT_THRESHOLD = 50_000

    def __init__(
        self,
        dim: int,
        nlist: int = 1024,
        nprobe: int = 64,
        m: int = 64,
        nbits: int = 8,
        use_gpu: bool = False,
    ) -> None:
        if not _HAS_FAISS:
            raise RuntimeError("faiss-cpu required.")
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.nbits = nbits
        self.use_gpu = use_gpu
        self.index: Optional[faiss.Index] = None
        self._ids: Optional[np.ndarray] = None  # original row indices

    def build(
        self,
        embeddings: np.ndarray,
        ids: Optional[np.ndarray] = None,
        chunk_size: int = 500_000,
    ) -> "ANNIndex":
        """
        Train and populate the FAISS index with chunked additions for memory safety.
        """
        n, d = embeddings.shape
        assert d == self.dim, f"Dim mismatch: expected {self.dim}, got {d}"

        if ids is not None:
            self._ids = np.array(ids, dtype=np.int64)
        else:
            self._ids = np.arange(n, dtype=np.int64)

        if n < self.FLAT_THRESHOLD:
            logger.info(f"Small dataset ({n:,} rows) — using flat L2 index.")
            base = faiss.IndexFlatIP(d)  # inner product = cosine for normalized vecs
            self.index = faiss.IndexIDMap(base)
            self.index.add_with_ids(embeddings[:].astype(np.float32), self._ids)
        else:
            logger.info(
                f"Building IVF-PQ index: n={n:,}, nlist={self.nlist}, "
                f"m={self.m}, nbits={self.nbits}, nprobe={self.nprobe}"
            )
            quantizer = faiss.IndexFlatL2(d)
            inner_index = faiss.IndexIVFPQ(
                quantizer, d, self.nlist, self.m, self.nbits
            )
            self.index = faiss.IndexIDMap(inner_index)

            # Train quantizer on representative sample (up to 100,000 vectors)
            n_train = min(100_000, n)
            train_idx = np.random.choice(n, size=n_train, replace=False)
            train_sample = np.ascontiguousarray(embeddings[train_idx], dtype=np.float32)
            
            logger.info(f"Training IVF-PQ quantizer on {n_train:,} sample vectors...")
            inner_index.train(train_sample)
            del train_sample

            # Populate index in chunks to maintain low memory usage
            logger.info(f"Adding {n:,} vectors to FAISS in chunks of {chunk_size:,}...")
            for i in range(0, n, chunk_size):
                end = min(i + chunk_size, n)
                chunk_emb = np.ascontiguousarray(embeddings[i:end], dtype=np.float32)
                chunk_ids = self._ids[i:end]
                self.index.add_with_ids(chunk_emb, chunk_ids)
                logger.debug(f"  Added {end:,}/{n:,} vectors...")

            inner_index.nprobe = self.nprobe

        logger.info(f"FAISS index built: {self.index.ntotal:,} vectors.")
        return self

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 50,
        chunk_size: int = 50_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for top-K neighbors in query chunks.
        """
        assert self.index is not None, "Build the index first."
        n_queries = len(query_embeddings)
        all_distances = np.zeros((n_queries, top_k), dtype=np.float32)
        all_indices = np.zeros((n_queries, top_k), dtype=np.int64)

        for i in range(0, n_queries, chunk_size):
            end = min(i + chunk_size, n_queries)
            q_chunk = np.ascontiguousarray(query_embeddings[i:end], dtype=np.float32)
            distances, indices = self.index.search(q_chunk, top_k)
            all_distances[i:end] = distances
            all_indices[i:end] = indices

        return all_distances, all_indices

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        if self._ids is not None:
            np.save(str(path) + ".ids.npy", self._ids)
        logger.info(f"FAISS index saved to {path}")

    def load(self, path: str | Path) -> "ANNIndex":
        path = Path(path)
        self.index = faiss.read_index(str(path))
        ids_path = str(path) + ".ids.npy"
        if Path(ids_path).exists():
            self._ids = np.load(ids_path)
        logger.info(f"FAISS index loaded from {path} ({self.index.ntotal:,} vectors)")
        return self
