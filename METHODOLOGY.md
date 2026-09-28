# Business Entity Resolution — Methodology Document

## 1. Approach Overview

The pipeline resolves which Source 2 / Source 3 records match each Source 1 entity,
optimizing **macro-averaged F_0.5** (precision-weighted ≈ 2×) at millions-of-records scale.

The architecture is strictly layered so each stage's output can be inspected, cached,
and iterated independently without rerunning the full pipeline.

```
Raw TSVs
  └─► Normalization (rule-based, data-derived)
        └─► Blocking / Candidate Generation  ← recall ceiling
              └─► Feature Engineering (per pair)
                    └─► LightGBM classifier + calibration
                          └─► Threshold tuning (grid-search F_0.5)
                                └─► [Optional DL refinement on ambiguous band]
                                      └─► matching_results.tsv + candidate_pairs.tsv
```

---

## 2. Blocking / Candidate Generation Strategy

### Why blocking matters most

Every true match that is **absent** from the candidate set is a **guaranteed false
negative** — no downstream model can recover it. Blocking recall is therefore the hard
ceiling on final F_0.5. We invest disproportionately here.

### Signal sources (union of four independent blockers)

| Blocker | Method | Recall contribution |
|---|---|---|
| **ANN** | FAISS IVF-PQ on multilingual-e5-large embeddings | Primary — handles paraphrase, script, word-order variation |
| **MinHash LSH** | Character 3-gram Jaccard ≥ 0.25 | Captures OCR typos, missing characters |
| **Phonetic** | Soundex + Metaphone on first significant token | Catches phonetically similar company names |
| **Geo** | Country + city bucket; postal-prefix fallback | Constrains to geographically plausible matches |

Candidates from all four sources are **unioned**, then re-ranked by (number of
agreeing signals DESC, ANN score DESC) and capped at `final_top_k=30` per S1 entity.

### Embedding model choice

`intfloat/multilingual-e5-large` (Apache-2.0, ~560M params) is used because:
- Trained on multilingual pairs across 100+ languages, including Hindi/Devanagari.
- Asymmetric retrieval: `query: …` prefix for S1 queries, `passage: …` for S23 index.
- Strong zero-shot performance on unseen languages (e.g., French).
- 560M parameters — well within the ≤8B hard constraint.

### FAISS IVF-PQ configuration

- `nlist = 1024` (≈ √N heuristic for ~1M vectors)
- `nprobe = 64` (probes 6.25% of cells — high recall, bounded latency)
- `m = 64`, `nbits = 8` (PQ with 256 centroids per subquantizer)
- Memory: ~4 bytes × 64 codes × N rows ≈ 256 MB per 1M rows (vs. ~3 GB for flat)
- Falls back to flat L2 for partitions < 50K rows automatically

### Blocking recall measurement

Measured on the held-out validation split (last 20% of Source 1 training entities)
before any model training. Logged as `blocking_recall` in `data/artifacts/`.
**Target: ≥ 0.95 blocking recall before proceeding to feature/model stages.**

---

## 3. Model Architecture & Feature Engineering

### Feature set (37 float32 features)

**Name similarity (13 features)**
- Exact match (norm_name equality)
- Suffix-stripped exact match (norm_name_no_suffix)
- Jaro-Winkler similarity
- Levenshtein ratio
- Token-sort ratio (RapidFuzz)
- Token-set ratio (RapidFuzz)
- Partial ratio (RapidFuzz)
- Jaccard on token sets
- TF-IDF cosine (char-wb 2–4 grams, 50K vocab)
- Embedding cosine (from multilingual-e5-large)
- Length delta (normalized)
- Missing-field flags (S1 / candidate)

**Address similarity (9 features)**
- Token overlap ratio (Jaccard on space-split tokens)
- Embedding cosine (same embedding encodes name+address)
- City exact match
- State/region exact match
- Postal code exact match
- Street number exact match
- Length delta
- Missing-field flags

**Meta features (11 features)**
- Country match flag
- Source indicator (S2 vs S3)
- Suffix canonical match
- Number of blocking signals agreeing (1–4)
- Per-signal binary flags (ann / lsh / phonetic / geo)
- Raw ANN cosine score
- Blocking rank

### Classifier: LightGBM

LightGBM is chosen over a DL matcher because:
1. **Speed**: scores millions of pairs in seconds, not hours.
2. **Precision control**: threshold is cleanly separable from the model.
3. **Interpretability**: feature importances directly diagnose errors.
4. **Calibration**: GBM probabilities respond well to isotonic regression.

Hyperparameters (tuned on OOF F_0.5):
```yaml
n_estimators: 1000 (with early stopping at 50)
num_leaves: 127
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
reg_alpha: 0.1, reg_lambda: 1.0
```

### Hard negative mining

Near-miss negatives (high JW similarity, different entity) hurt precision most.
Training set = all positives + all hard negatives (JW ≥ 0.85) + sampled easy
negatives to reach `hard_negative_ratio × n_positives` total.
Sample weights: positives=1.0, hard-negatives=2.0, easy-negatives=0.5.

### k-fold cross-validation

5-fold stratified on binary labels. Per-fold and aggregate OOF F_0.5 are logged.
Country-stratified performance breakdown is reported to catch US/India overfitting.

---

## 4. Calibration & Threshold Tuning Rationale

### Why calibrate?

Raw GBM probabilities are well-ordered but not well-calibrated (they cluster near
0/1 and don't represent true match probabilities). Isotonic regression maps OOF
predictions to calibrated values, making thresholding stable and reproducible across
different data distributions.

### Why grid-search the threshold?

F_0.5 is not maximized at p=0.5. Because precision is weighted ~2×:
- False positives (matching a non-match) cost ~4× more than false negatives in F_0.5.
- The optimal threshold is typically in the range **0.60–0.85** depending on the
  positive rate in the candidate set.
- A 65-point grid over [0.30, 0.95] is searched; best threshold is saved and used
  identically for test inference.

### Singleton validation

Entities with **no candidate above threshold** output an **empty matched_ids field**.
The local F_0.5 scorer gives these entities a score of 1.0 (correct singleton).
Any false positive on a true singleton scores 0.0 for that entity.
Threshold tuning explicitly validates singleton accuracy alongside macro F_0.5.

---

## 5. Generalization to Unseen France Test Segment

### The challenge

Training data contains only US and India records. France is a zero-shot segment
in the test set. Without mitigation, patterns learned on US legal suffixes (LLC, Inc)
or Indian conventions (Pvt Ltd) may fail to generalize.

### Mitigation strategy

**1. Country-agnostic normalization by design**
- No country-specific code paths — all normalization is generic text processing.
- Legal suffix map explicitly includes French suffixes (SARL/SAS/EURL/SA etc.)
  derived from public knowledge of French business naming conventions.
- Address abbreviation expander includes French prefixes (Rue, Avenue, Bd, etc.)

**2. Multilingual embeddings**
- multilingual-e5-large was trained on French-language data and handles French
  business names natively without any fine-tuning.

**3. Synthetic French augmentation**
- `FrenchAugmentor` generates 1,000 synthetic French record pairs with controlled
  match/no-match labels and injects them into a held-out validation subset.
- This validation subset is used to measure per-country F_0.5 at every iteration.
- Blocking recall on the synthetic France subset is reported separately.

**4. Country-stratified CV**
- k-fold metrics are reported per-country so France-specific degradation is visible
  during training, not just at test-time leaderboard submission.

**5. Feature design**
- Country-match flag is a feature, not a filter — the model learns to use it as
  a signal, not a hard constraint.

---

## 6. Output Format Compliance

| File | Schema | Notes |
|---|---|---|
| `matching_results.tsv` | `source1_id \t matched_ids` | Every S1 test entity exactly once; matched_ids pipe-separated or empty |
| `candidate_pairs.tsv` | `source1_id \t candidate_id \t ann_score \t sources \t rank` | All blocking output |

`utils/validate_submission.py` must pass with exit code 0 before any final submission.

---

## 7. Computational Budget & Scalability

| Stage | Memory | Time (2.2M S1, 2.2M S23) |
|---|---|---|
| Normalization | <4 GB RAM (streaming) | ~30 min |
| Embedding (GPU) | ~4 GB VRAM | ~2 h |
| FAISS build | ~2 GB RAM | ~10 min |
| Candidate gen | <8 GB RAM | ~45 min |
| Feature engineering | <16 GB RAM | ~1.5 h |
| LightGBM train | <8 GB RAM | ~20 min |
| Threshold tuning | <4 GB RAM | ~10 min |
| Test inference | Same as above | ~3 h |

Streaming reads (Polars `scan_csv`, pandas `chunksize`) ensure no stage naively
loads a full multi-GB TSV into memory. Parquet caching means re-running any stage
after the first run is near-instant for the cached stages.

---

## 8. Known Limitations & Future Work

- **LSH threshold `0.25`**: may be too low for very short names (<5 chars). Consider
  per-length-bucket tuning.
- **FAISS nprobe=64**: increasing to 128 improves recall ~1% at 2× query cost.
- **DL refinement**: currently disabled by default. Enable for highest-precision
  runs on the ambiguous band (saves ~1-2% F_0.5 at 3–5× GPU cost).
- **Ensemble**: adding a second GBM with different feature subsets (name-only,
  address-only) and stacking may yield +0.5–1% F_0.5.
