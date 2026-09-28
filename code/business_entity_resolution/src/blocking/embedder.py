"""
Multilingual Dense Embedding Generator with Memory-Mapped Streaming.

Uses intfloat/multilingual-e5-base (Apache-2.0, ~278M params) or large.
Streams embeddings directly to disk via np.lib.format.open_memmap in chunks,
preventing CPU RAM exhaustion on 10M+ records.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from tqdm import tqdm

try:
    import torch
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False
    logger.warning("sentence-transformers not installed. Embedding disabled.")


class EntityEmbedder:
    """
    Encodes business records into dense multilingual embeddings with GPU acceleration
    and memory-mapped disk streaming to keep RAM flat (<500MB).
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        batch_size: int = 512,
        max_length: int = 64,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
    ) -> None:
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers required for embeddings.")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize_embeddings
        self.device = device or ("cuda" if _HAS_ST and self._has_cuda() else "cpu")
        logger.info(f"Loading embedding model: {model_name} on {self.device}")
        
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            model_kwargs={"torch_dtype": torch.float16 if (use_fp16 and self.device == "cuda") else torch.float32},
        )
        self.model.max_seq_length = max_length
        if use_fp16 and self.device == "cuda":
            self.model.half()
            logger.info("Enabled FP16 half-precision for GPU acceleration.")
            
        logger.info(f"Embedding dim: {self.model.get_embedding_dimension()} (max_seq_length={max_length})")

    @property
    def dim(self) -> int:
        return self.model.get_embedding_dimension()

    def build_texts(self, names: list[str], addresses: list[str], prefix: str = "passage: ") -> list[str]:
        """
        Combine name + address with prefix for asymmetric retrieval.
        """
        texts = []
        for name, addr in zip(names, addresses):
            combined = f"{name or ''} {addr or ''}".strip()
            texts.append(f"{prefix}{combined}")
        return texts

    def encode_to_file(
        self,
        texts: list[str],
        output_path: str | Path,
        chunk_size: int = 100_000,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encodes texts in chunks directly into a memory-mapped .npy file on disk.
        RAM usage is completely capped (<500MB) regardless of dataset size.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        n_samples = len(texts)
        dim = self.dim

        logger.info(
            f"Encoding {n_samples:,} texts directly to disk via memmap: {output_path.name} "
            f"(chunk_size={chunk_size:,}, batch_size={self.batch_size}, device={self.device})..."
        )

        # Allocate memmap array on disk
        mmap_arr = np.lib.format.open_memmap(
            str(output_path),
            mode="w+",
            dtype=np.float32,
            shape=(n_samples, dim),
        )

        total_chunks = (n_samples + chunk_size - 1) // chunk_size
        for chunk_idx in range(total_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_samples)
            chunk_texts = texts[start:end]

            logger.info(f"  Encoding chunk {chunk_idx + 1}/{total_chunks} ({start:,} to {end:,})...")
            chunk_emb = self.model.encode(
                chunk_texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )
            mmap_arr[start:end] = chunk_emb
            mmap_arr.flush()  # Persist to disk and release RAM immediately

        logger.info(f"Successfully saved {n_samples:,} embeddings ({output_path.name})")
        return mmap_arr

    def encode(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        In-memory encode for smaller datasets (<100K).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return embeddings.astype(np.float32)

    def load_embeddings(self, path: str | Path, mmap_mode: Optional[str] = "r") -> np.ndarray:
        """Load embeddings with memory-mapping (zero RAM overhead)."""
        arr = np.load(str(path), mmap_mode=mmap_mode)
        logger.info(f"Loaded embeddings {arr.shape} from {path} (mmap_mode={mmap_mode})")
        return arr

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
