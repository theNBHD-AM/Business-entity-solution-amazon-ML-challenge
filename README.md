# Business Entity Resolution Pipeline

Production-grade entity resolution pipeline optimized for **macro-averaged F_0.5** at multi-million-record scale. Built for a competitive ML challenge with zero external API/registry lookups.

---

## Project Structure

```
business_entity_resolution/
├── code/business_entity_resolution/src/
│   ├── data/
│   │   ├── loader.py              # Streaming TSV loader (Polars/pandas chunked)
│   │   └── corruption_detector.py # Row-level encoding anomaly detection
│   ├── normalization/
│   │   ├── normalizer.py          # Master normalization orchestrator
│   │   ├── legal_suffix.py        # Data-derived legal suffix canonicalization
│   │   ├── address_parser.py      # Abbreviation expansion + field extraction
│   │   ├── transliterator.py      # Devanagari → Latin (indic-transliteration)
│   │   └── synthetic_france.py    # Synthetic French record augmentor
│   ├── blocking/
│   │   ├── embedder.py            # multilingual-e5-large dense embeddings
│   │   ├── ann_index.py           # FAISS IVF-PQ index (scalable ANN)
│   │   ├── minhash_lsh.py         # Character-ngram MinHash LSH (datasketch)
│   │   ├── phonetic_blocker.py    # Soundex + Metaphone (jellyfish)
│   │   ├── geo_blocker.py         # Country + city/postal bucket blocking
│   │   └── candidate_generator.py # Signal union + re-rank → candidate pairs
│   ├── features/
│   │   └── feature_builder.py     # 37-feature engineering per candidate pair
│   ├── model/
│   │   ├── gbm_classifier.py      # LightGBM/XGBoost k-fold trainer
│   │   ├── calibrator.py          # Isotonic / Platt probability calibration
│   │   └── hard_negative_miner.py # Near-miss negative mining + sample weights
│   ├── evaluation/
│   │   ├── metrics.py             # Exact F_0.5 scorer (per-entity macro-avg)
│   │   └── threshold_tuner.py     # Grid-search threshold → max F_0.5
│   ├── inference/
│   │   ├── predictor.py           # End-to-end inference → TSV output
│   │   └── dl_refiner.py          # Optional LLM verifier for ambiguous pairs
│   ├── pipeline.py                # Full orchestrator (train / test modes)
│   └── main.py                    # CLI entry point
├── utils/
│   ├── validate_submission.py     # Pre-submission validation (run before upload)
│   └── evaluate_local.py          # Local F_0.5 scorer vs. ground-truth labels
├── configs/
│   └── pipeline_config.yaml       # All hyperparameters in one place
├── data/
│   ├── raw/                       # Place TSV files here (not committed)
│   ├── processed/                 # Normalized Parquet files (auto-generated)
│   └── artifacts/                 # Embeddings, FAISS index, model files
├── outputs/                       # matching_results.tsv, candidate_pairs.tsv
├── METHODOLOGY.md                 # Full methodology document
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU (optional but recommended for embeddings):** Install `torch` with CUDA support
> before running. The pipeline auto-detects CUDA.

### 2. Place raw data

```
data/raw/
  source1_train.tsv
  source2_train.tsv
  source3_train.tsv
  train_labels.tsv
  source1_test.tsv
  source2_test.tsv
  source3_test.tsv
```

Expected TSV schema for all source files:
```
entity_id\tbusiness_name\tbusiness_address\tcountry
```

Expected schema for `train_labels.tsv`:
```
source1_id\tmatched_ids
```
where `matched_ids` is pipe-separated (e.g., `S2-001|S3-002`) or empty for singletons.

### 3. Run full pipeline (train + test)

```bash
cd business_entity_resolution
python run.py --config configs/pipeline_config.yaml --mode full
```

Or separately:

```bash
# Train only
python run.py --mode train

# Test only (requires trained artifacts)
python run.py --mode test
```

### 4. Validate outputs

```bash
python utils/validate_submission.py \
    --results outputs/matching_results.tsv \
    --candidates outputs/candidate_pairs.tsv \
    --source1 data/raw/source1_test.tsv \
    --source2 data/raw/source2_test.tsv \
    --source3 data/raw/source3_test.tsv
```

Must exit with code 0 before any submission.

### 5. Local F_0.5 evaluation (use at every iteration)

```bash
python utils/evaluate_local.py \
    --results outputs/matching_results.tsv \
    --labels data/raw/train_labels.tsv \
    --source1 data/raw/source1_train.tsv
```

---

## Configuration

All tunable hyperparameters live in [`configs/pipeline_config.yaml`](configs/pipeline_config.yaml).
Key parameters:

| Parameter | Default | Purpose |
|---|---|---|
| `blocking.embedding_model` | `intfloat/multilingual-e5-large` | Swap to `BAAI/bge-m3` or `LaBSE` |
| `blocking.final_top_k` | 30 | Candidates per S1 entity (↑ = higher recall, slower) |
| `blocking.faiss_nprobe` | 64 | FAISS probe cells (↑ = better recall, slower) |
| `model.gbm_backend` | `lightgbm` | Or `xgboost` |
| `model.hard_negative_ratio` | 5 | Negatives per positive |
| `model.calibration_method` | `isotonic` | Or `sigmoid` (Platt) |
| `threshold.search_range` | `[0.3, 0.95]` | Grid search bounds |
| `dl_refinement.enabled` | `false` | Enable LLM verifier for ambiguous pairs |

---

## Reproducing from Scratch

```bash
# 1. Clone / unzip the project
# 2. Install dependencies
pip install -r requirements.txt

# 3. Place raw data in data/raw/ (see schema above)

# 4. (Optional) Edit configs/pipeline_config.yaml

# 5. Full run
python -m code.business_entity_resolution.src.main \
    --config configs/pipeline_config.yaml \
    --mode full \
    --log-level INFO

# 6. Validate
python utils/validate_submission.py \
    --results outputs/matching_results.tsv \
    --candidates outputs/candidate_pairs.tsv \
    --source1 data/raw/source1_test.tsv \
    --source2 data/raw/source2_test.tsv \
    --source3 data/raw/source3_test.tsv

# 7. Evaluate locally
python utils/evaluate_local.py \
    --results outputs/matching_results.tsv \
    --labels data/raw/train_labels.tsv
```

---

## Design Decisions

### Metric-first design
Every decision is evaluated against **macro F_0.5**, not accuracy or F1. The
threshold tuner grid-searches explicitly for the F_0.5-maximizing threshold,
typically in the 0.60–0.85 range (higher than naive 0.5 due to precision weighting).

### Blocking recall is the ceiling
We invest in four independent blocking signals (ANN + LSH + phonetic + geo)
and take their union. Blocking recall is measured on validation before model training.
Target: ≥ 0.95.

### No external data
All normalization dictionaries (legal suffixes, address abbreviations, French
business patterns) are derived from either (a) frequency analysis of the training
data itself, or (b) generic publicly-known linguistic patterns. No external APIs,
geocoders, or business registries.

### Scale by default
- Streaming reads: never loads full TSV into memory
- Parquet intermediates: 3–5× smaller than TSV, 10× faster to read
- FAISS IVF-PQ: ~60× less memory than flat index at comparable recall
- LightGBM: scores millions of pairs in seconds

### France generalization
- Normalization is country-agnostic by design
- Multilingual embeddings handle French natively
- French legal suffixes (SARL/SAS/EURL) included in seed suffix map
- Synthetic French validation subset stress-tests zero-shot generalization

---

## License Compliance

| Component | License |
|---|---|
| `intfloat/multilingual-e5-large` | Apache-2.0 |
| `BAAI/bge-m3` (alternative) | MIT |
| `sentence-transformers/LaBSE` (alternative) | Apache-2.0 |
| `LightGBM` | MIT |
| `XGBoost` | Apache-2.0 |
| `FAISS` | MIT |
| `datasketch` (MinHash LSH) | MIT |
| `jellyfish` (phonetics) | MIT |
| `RapidFuzz` | MIT |
| `Polars` | MIT |
| `Qwen2.5-7B-Instruct` (optional DL) | Apache-2.0 |
| `Mistral-7B-Instruct-v0.3` (optional DL) | Apache-2.0 |
| `Phi-3.5-mini-instruct` (optional DL) | MIT |

All models are ≤ 8B parameters. ✅

---

## Contact / Questions

If additional training data samples are needed to refine any stage (e.g., more
examples of specific noise patterns, or the actual French test records for
calibration), provide them and the pipeline config/data paths can be updated accordingly.
