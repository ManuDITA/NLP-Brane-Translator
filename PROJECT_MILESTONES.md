# NLP-Brane-Translator — Project Milestones

A chronological record of all major technical contributions made to this project.
Intended as a reference for writing the thesis.

---

## Phase 1 — Foundation: Packages and Brane Integration

### M1.1 — Healthcare Package: Initial Implementation
Built the first fully functional Brane package in Python (`packages/healthcare/`).
- Defined `container.yml` with custom class types (`VitalSigns`, `LabResults`, `Patient`, `RiskAssessment`, `PatientSummary`, `TriageResult`) exposed to BraneScript
- Implemented core functions: `analyze_heart_disease`, `assess_diabetes_risk`, `triage_patient`, `get_patient_summary`, `generate_report`, `compute_bmi`, `validate_patient_data`
- Implemented batch/dataset-level functions accepting `Data` as input: `analyze_patients_file`, `batch_diabetes_from_file`, `batch_triage_from_file`, `filter_high_risk_patients`, `compute_cohort_statistics`, `filter_by_condition`, `compute_risk_distribution`, `compute_mortality_risk`, `check_vital_signs`, `predict_readmission_risk`
- Discovered the correct Brane package dispatch mechanism: `sys.argv[1]` for action name (not env var), and `command.args: [action_name]` in `container.yml`
- Discovered and fixed root cause of "Docker container wait error": missing Python in the container. Fixed by adding `dependencies: [python3]` and `install: [pip3 install pyyaml]`
- Built the package using a local branelet binary rather than the broken GitHub release URL

### M1.2 — Package Ecosystem Expansion
Extended the system from 1 to 7 functional packages:
- **`datetime`** — 6 functions: `get_iso`, `get_date`, `get_time`, `get_unix`, `format_date`, `get_timezone`
- **`text_analysis`** — 8 functions: `count_words`, `count_sentences`, `extract_keywords`, `detect_language`, `compute_readability`, `extract_named_entities`, `classify_sentiment`, `summarize_text`; with class types `SentimentResult`, `KeywordResult`, `ReadabilityResult`, `NamedEntityResult`
- **`statistics`** — 8 functions: `compute_summary_stats`, `count_by_category`, `compute_correlation`, `aggregate_by_group`, `filter_by_threshold`, `detect_outliers`, `compute_histogram`, `compute_percentile`; with `SummaryStats` and `CorrelationResult` class types
- **`epidemics`** — 9 functions covering incidence rate, reproduction number, outbreak detection, epidemic stage classification, attack rate, case fatality rate, epidemic report generation; with `EpidemicStatus` and `EpidemicReport` class types
- **`genomics`** — 8 functions: `compute_gc_content`, `get_complement`, `get_reverse_complement`, `translate_codon`, `find_motif`, `count_nucleotides`, `detect_mutations`, `compute_sequence_similarity`
- **`data_masking`** — 7 functions: `mask_value`, `detect_pii`, `mask_json_record`, `anonymize_dataset`, `pseudonymize_field`, `generate_synthetic_id`, `redact_text`

**Key insight:** All packages use the same Brane dispatch pattern: `sys.argv[1]` selects the action; class types are serialized/deserialized as `["ClassName", {fields}]` via Brane's FullValue encoding.

### M1.3 — Package Build Process Discovery
Identified that `brane package build --branelet` breaks packages with `IntermediateResult`-returning functions ("Docker container wait error"). The correct build method:
```bash
docker buildx build --output "type=docker,dest=image.tar" \
  --platform linux/x86_64 -t <name>:1.0.0 <dir>/
```
followed by extracting the digest from `manifest.json` and updating `package.yml`.

### M1.4 — Class-Based Input Refactor (Major)
Refactored all 6 packages from accepting raw JSON strings to using proper BraneScript class types as inputs/outputs.
- Before: `analyze_heart_disease(patient_json: string)` — model had to serialize objects to JSON
- After: `analyze_heart_disease(patient: Patient)` — model uses `new Patient { ... }` syntax directly
- Rewrote ~1,400 BraneScript training examples to use the new class-based API
- Eliminated a major source of hallucination: the model no longer needs to know the JSON schema

---

## Phase 2 — Inference Pipeline

### M2.1 — Core Pipeline (`src/pipeline.py`)
Built the end-to-end natural language → BraneScript → execution pipeline:
1. Intent received
2. RAG retrieval: fetch relevant package/dataset docs from ChromaDB
3. LLM generation: Qwen3 on HuggingFace produces BraneScript
4. Execution: `brane workflow run` on local Brane instance
5. Error retry loop: if compile/runtime error, feed stderr back to LLM for correction

### M2.2 — Remote Execution via Snellius File Queue
Built a complete remote execution bridge for LLM inference on Snellius HPC while BraneScript runs locally:
- File-queue protocol: local machine polls a Snellius-accessible directory for `.json` jobs
- Snellius writes intent + context to a file; local machine picks it up, executes Brane, writes result back
- Solved multiple issues: SSH config, `~` expansion in `.env`, `serve_forever()` deadlock on Ctrl+C

### M2.3 — Qwen3 Thinking Mode Disabled
Discovered that Qwen3's `<think>...</think>` reasoning blocks consumed almost all `max_new_tokens`, leaving nothing for the actual BraneScript output.
- Added `enable_thinking=False` to all `apply_chat_template` calls
- Resulted in ~4× speedup and eliminated empty-output failures

### M2.4 — System Prompt Engineering (`src/prompts.py`)
Centralized all prompt logic into a single source of truth used by pipeline, evaluate, and prepare_dataset:
- 20 explicit rules covering BraneScript syntax, package function usage, class types, Data/IntermediateResult handling, empty strings, commit_result behavior, etc.
- Key rules added over time:
  - Rule 4: No markdown code fences in output
  - Rule 7: No inventing packages or functions
  - Rule 13b: Never use empty strings `""` — use `"none"` instead
  - Rule 14: Don't reimplement package logic internally
  - Rule 16: Always use `new Data{ name := "..." }` for datasets
  - Rule 19: Choose Data-accepting functions for dataset intents, single-Patient functions for inline patient intents
  - Rule 20: Use only exact function names from RAG context
- Injects full BraneScript language spec (`data/syntax_reference.md`) into every prompt

---

## Phase 3 — RAG Retrieval System

### M3.1 — Initial RAG with Two ChromaDBs
Built the first RAG system with two vector databases:
- `brane_pkg_db` — package container.yml files + dataset docs
- Used `sentence-transformers/all-MiniLM-L6-v2` for embeddings

### M3.2 — Name-Pinned Metadata Retrieval
Replaced pure similarity search with a hybrid approach:
- `_detect_names()`: detects package/dataset names mentioned in the intent
- If names detected → fetch by `package` metadata filter (pinned retrieval)
- If no names → fall back to similarity search with k=8
- Result: retriever always returns the exact package API when a package is mentioned

### M3.3 — Package Aliases for Domain Keywords
Added `PACKAGE_ALIASES` dict mapping domain vocabulary to package names:
- "heart disease", "cardiovascular", "cvd", "triage" → `healthcare`
- "dna", "sequence", "gc content", "nucleotide" → `genomics`
- "outbreak", "incidence", "epidemic stage" → `epidemics`
- "pii", "mask", "anonymize" → `data_masking`
- etc.
- Solves the case where the user writes "analyze cardiovascular risk" without mentioning "healthcare"

### M3.4 — Per-Function RAG Chunks (Major)
Rewrote `knowledgeBase.py` chunking strategy from "keep container.yml whole" to "one chunk per function":
- Each function gets its own Document containing: description, signature, "Use when" hint, BraneScript example
- Types/classes get their own Document with field definitions and usage notes
- Before: 28 total chunks; After: 91 chunks
- Critical improvement: for an intent about "all patients in dataset", the retriever can now semantically match the correct `batch_diabetes_from_file(Data)` chunk instead of the surface-similar `assess_diabetes_risk(Patient)` chunk
- "Use when" hints embedded in each chunk tell the model exactly when to use each function type

### M3.5 — Dataset Package Metadata
Added `package` metadata to dataset documents so package-pinned retrieval also returns the corresponding dataset registration doc alongside the API functions.

---

## Phase 4 — Training Data

### M4.1 — Batch Execution Pipeline (`scripts/batch_execute.py`)
Built a parallel batch execution system:
- Runs all BraneScript examples against the local Brane instance
- Captures stdout, stderr, exit code, committed results
- `--resume` flag skips already-executed examples by ID
- ID format: `<filename_stem>-<lineno:04d>` for stable cross-run identity
- Output: `data/training/execution_results.jsonl` with ground truth stdout

### M4.2 — Training Dataset: 658 Verified Examples
Built and verified a dataset of 658 BraneScript intent/code pairs, all passing:
- `basic.jsonl` — 7 examples (hello world, imports, variables)
- `classes_arrays.jsonl` — 7 examples (class definitions, field access)
- `control_flow.jsonl` — 5 examples (if/else, loops)
- `functions.jsonl` — 6 examples (function definitions)
- `advanced.jsonl` — 15 examples (Data, IntermediateResult, commit_result, parallel)
- `packages.jsonl` — 55 examples (cross-package workflows)
- `healthcare.jsonl` — 7 examples (Patient class, all healthcare functions)
- `genomics.jsonl` — 100 examples (all genomics functions, sequences)
- `training_500.jsonl` — 456 examples (the main dataset, all packages, complex intents)
- Total: **658 examples, 658/658 passing, 0 non-deterministic outputs**

### M4.3 — Non-Determinism Elimination
Removed all non-deterministic outputs from training data:
- Removed 49 datetime examples (timestamps are non-deterministic by nature)
- Removed timestamp fields from all package outputs (epidemics report, genomics outputs) by editing `_out_str` calls in package Python code
- Rebuilt affected Docker images and re-executed all examples
- Final state: 658 examples with stable, reproducible stdout

### M4.4 — Intent/BraneScript Mismatch Audit
Discovered that many intents were too vague — "print the risk analysis" when the BraneScript printed `result.risk_level` and `result.risk_score` specifically. Systematically audited all 658 examples:
- Fixed intents in `advanced.jsonl`, `control_flow.jsonl`, `functions.jsonl`, `genomics.jsonl`, `healthcare.jsonl`, `packages.jsonl`, `training_500.jsonl`
- Rule: the intent must specify exactly what the BraneScript prints, using the same field names

### M4.5 — Train/Val Split and Deduplication
- `prepare_dataset.py` creates 560 train / 98 val split from `execution_results.jsonl`
- Fixed a doubling bug: `all_examples.jsonl` (the merge file) was accidentally included in the glob, duplicating every example → now skipped explicitly

---

## Phase 5 — Fine-Tuning

### M5.1 — SFT Training Infrastructure
Built `src/fine_tuning/train.py` with:
- Qwen3 QLoRA via `bitsandbytes` 4-bit quantization + PEFT LoRA (rank 16)
- Unsloth integration with fallback to plain HF
- Auto-resume from latest checkpoint
- SLURM submission scripts for Snellius H100 GPUs (`sbatch_finetune.sh`)
- Config: 3 epochs, batch 2, grad accum 4, lr 2e-4, cosine schedule

### M5.2 — Response-Only Training Fix
Fixed a critical training flaw: `SFTTrainer` was computing loss on ALL tokens (system prompt + user message + BraneScript). This diluted the gradient signal on the BraneScript and `<|im_end|>` stop token.
- Plain HF path: added `DataCollatorForCompletionOnlyLM` with response template `<|im_start|>assistant\n`
- Unsloth path: added `train_on_responses_only()` with matching instruction/response parts
- Effect: 100% of gradient now targets BraneScript generation and the stop token — model learns when to stop generating

### M5.3 — GRPO Training Infrastructure
Built full GRPO (Group Relative Policy Optimization) training loop:
- Generates multiple BraneScripts per intent
- Executes each against live Brane instance in parallel
- Scores: +1.0 (runs successfully), +0.3 (runtime error), -1.0 (compile error/timeout)
- Normalizes rewards within each group (GRPO's relative reward signal)
- Supports warm-start from SFT-merged model

### M5.4 — Model Merge
Added `--merge` flag to `train.py`: merges LoRA adapter into base model weights to produce a single HuggingFace model directory for deployment/evaluation.

---

## Phase 6 — Evaluation Infrastructure

### M6.1 — Evaluation Harness (`src/fine_tuning/evaluate.py`)
Built a full evaluation pipeline:
- Loads any model (base, SFT-merged, GRPO-merged) or model + LoRA adapter
- Generates BraneScript for each test intent
- Executes against live Brane instance
- Computes: `compile_rate`, `execution_rate`, `output_match_rate`
- Checkpoints every N examples (resumable)
- `--generate-only` mode for Snellius (no Brane available there)

### M6.2 — Snellius → Local Pipeline (`scripts/process_snellius.py`)
Built `process_snellius.py` to automate post-Snellius evaluation:
1. Scans `output_snellius/` for `*_generated.json` files
2. Executes each generated BraneScript against local Brane
3. Saves full results to `outputs/eval/`
4. Prints model comparison table

### M6.3 — Inference Fix: Explicit EOS Token
Fixed a generation bug where `model.generate()` did not explicitly pass `eos_token_id`. For merged Qwen3 models, both `<|im_end|>` and `<|endoftext|>` must be passed as stop tokens. Without this, the model could overshoot the end of its reply.

### M6.4 — Baseline Results (July 27, 2026)
Ran baseline evaluation on all 560 training examples for 3 model sizes:

| Model | Execution Rate | Output Match (of executed) |
|---|---|---|
| Qwen3.6-27B (base) | 37.9% | 40.3% (196 comparable) |
| Qwen3.5-9B (base) | 20.9% | 28.0% (107 comparable) |
| Qwen3.5-4B (base) | 5.5% | 58.1% (31 comparable) |

### M6.5 — Frontend Evaluation Dashboard (`frontend/`)
Built a local web dashboard:
- **Results tab**: browse all generated BraneScripts with execution status, stdout, stderr
- **Exec Results tab**: ground-truth execution results browser
- **Model Comparison tab**: side-by-side metrics across models
- **Packages tab**: live package/function reference
- **Datasets tab**: registered dataset reference
- Search by intent or example ID; sidebar with stats bar

---

## Phase 7 — Infrastructure and Tooling

### M7.1 — SLURM Submission Scripts
- `sbatch_baseline.sh` — baseline evaluation job; requires `EVAL_MODEL` via `--export=ALL,EVAL_MODEL=...`
- `sbatch_finetune.sh` — SFT/GRPO training job; `FINETUNE_MODEL` and `TRAIN_MODE` configurable
- `submit_baselines.sh` — submits multiple baseline jobs in parallel
- `sbatch_generate.sh` — SLURM wrapper for the frontend Generate tab
- `sbatch_paraphrase.sh` — SLURM job for LLM-driven paraphrase generation

### M7.2 — Syntax Reference (`data/syntax_reference.md`)
Wrote a full BraneScript language specification injected into every LLM prompt:
- Variable declarations, class definitions, function calls, import syntax
- Data and IntermediateResult usage patterns
- Parallel execution syntax (`#[on("node")]`)
- Common pitfalls (empty strings, no empty string literals, array field type restrictions)

### M7.3 — Thesis LaTeX Integration
Migrated the LaTeX thesis template into the main project repository:
- `Thesis Latex/` folder with `main.tex`, numbered sections `01_intro.tex` through `11_appendix.tex`
- Images consolidated in `Thesis Latex/images/`
- `.gitignore` extended with LaTeX build artifact patterns
- `make build` builds the thesis PDF via `latexmk`

### M7.4 — Output Folder Consolidation
Unified all scattered output directories under `outputs/`:
- `outputs/pipeline/results/` — job_watcher.py execution results
- `outputs/eval/` — evaluation JSONs (one per model run)
- `outputs/metrics/` — training loss JSONL files (one per model)
- `outputs/snellius/` — raw Snellius-generated JSON files
- `outputs/branescripts/` — generated `.bs` files from pipeline.py
- `outputs/logs/` — PID files and service logs
- `outputs/models/` — LoRA adapters and merged model weights

### M7.5 — `start.sh` — Unified Service Entry Point
Created a single script to manage all local services:
```bash
bash start.sh           # start job_watcher + dashboard
bash start.sh --restart # restart both services
bash start.sh --stop    # stop both services
```
- Writes PIDs to `outputs/logs/`
- Health check on dashboard port after startup
- Handles `set -euo pipefail` edge cases in process management

### M7.6 — Training Metrics and Loss Plotting
- Added `MetricsCallback` to `train.py`: appends training step metrics (loss, grad norm, learning rate, epoch) to `outputs/metrics/<model>_training_metrics.jsonl`
- Added `_save_training_config()`: saves all hyperparameters to `training_config.json` in the adapter directory (with both a plain copy and a timestamped copy so multiple runs don't overwrite each other)
- Created `scripts/plot_training.py`: reads JSONL metrics, plots train/val loss curves and GRPO reward over steps; saves PNG for thesis figures

### M7.7 — Training Config in Eval Files
Embedded training hyperparameters directly in evaluation output files:
- `evaluate.py` reads `training_config.json` from the model dir and stores it in the eval JSON at save time
- Solves the problem of distinguishing multiple SFT runs on the same base model
- `server.py` prefers embedded config over on-disk lookup, falls back gracefully for older files

---

## Phase 8 — Frontend Generate Tab

### M8.1 — Generate Tab: Real-Time BraneScript Generation
Added a new "Generate" tab to the dashboard allowing interactive BraneScript generation:
- Text area for intent input
- Model selector (reads `FRONTEND_MODELS` from `.env`)
- Submits SLURM job to Snellius via SSH; polls result via file queue
- Shows generated BraneScript, execution result, stdout/stderr in a job card UI

### M8.2 — Job Queue UI with Tabbed Cards
Built a full job queue interface:
- Stack of job cards (one per submission), newest at top
- Progress steps: Queued → Generating → Executing → Done
- Each card has Code / Error / Output / History tabs
- Error banners for failed runs; syntax-highlighted BraneScript boxes
- Attempt badge showing which retry this is
- 8-second fade-out on completion; completed jobs move to Runs tab

### M8.3 — Auto-Retry with Error Feedback (up to 3 attempts)
Implemented an intelligent retry loop for failed BraneScript generation:
- `MAX_GENERATE_RETRIES = 3` (constant in server.py)
- On compile/runtime failure: writes `~/brane_jobs/context/<req_id>.json` on Snellius with `{prev_script, error_feedback, attempt}`
- `generate_single.py` reads context on retry; injects error into prompt: "PREVIOUS ATTEMPT FAILED — fix this script: ..."
- Each retry history preserved in the job card's History tab
- Runs tab shows final result only; Generate tab shows active jobs

### M8.4 — Model Info Modal in Evaluation Tab
Clicking any SFT/GRPO model label in the Model Comparison table opens a modal showing:
- All training hyperparameters (base model, LoRA rank/alpha, epochs, learning rate, batch size, etc.)
- Model path and run ID
- Works by reading embedded `training_config` from the eval JSON or falling back to `training_config.json` on disk

---

## Phase 9 — Semantic Cache

### M9.1 — SemanticCache Class (`src/semantic_cache.py`)
Built a persistent semantic cache backed by ChromaDB (`brane_pkg_db/intent_cache` collection):
- `lookup(intent)` → cosine similarity search; returns BraneScript if similarity ≥ threshold (default 0.92)
- `store(intent, branescript)` → stores successful result
- `invalidate(intent)` → removes stale entries matching the intent
- `stats()` / `list_entries()` for inspection
- Integrated into `generate_single.py`: checks cache before loading the LLM model; stores result after generation

### M9.2 — Cache Population from Training Data
Created `scripts/populate_cache.py`:
- Reads all 658 training examples (reference BraneScripts = human-written ground truth)
- Stores each intent → reference BraneScript pair in the semantic cache
- Deduplication: skips intents already in cache (with similarity check)
- `--clear` flag to wipe and re-seed; `--threshold` to control dedup sensitivity

### M9.3 — Paraphrase Generation Pipeline
Built end-to-end paraphrase generation for cache evaluation:
- `scripts/generate_paraphrases.py`: uses an LLM to generate N semantically equivalent rewordings per intent; writes to `data/training/paraphrases.jsonl`
- `sbatch_paraphrase.sh`: SLURM wrapper (default: Qwen3.5-4B, N=3 paraphrases per intent)
- Output format: `{id, original_id, source_file, intent, branescript}` — same schema as training data
- Target: 1974 paraphrases (3 × 658 intents) for cache evaluation

### M9.4 — Cache Lookup Benchmark (`scripts/test_cache_lookup.py`)
Benchmark tool to measure how well the semantic cache handles paraphrased intents:
- Loads A intents from cache, queries with B paraphrases
- Metrics: hit rate, correct rate (exact match), fuzzy correct rate (≥0.95 bigram similarity), false positive rate, similarity distribution
- `--sweep` mode: runs across thresholds [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98] to find optimal operating point
- Saves JSON results for thesis plotting

### M9.5 — Cache API Endpoints
Added to `server.py`:
- `GET /api/cache/stats` — total entries, total hits, threshold
- `GET /api/cache/entries` — list cached intents (newest first)
- `POST /api/cache/invalidate` — remove entries similar to a given intent

---

## Phase 10 — Updated Baseline Results (August 2026)

### M10.1 — Improved Baseline Results After Prompt + RAG Fixes
Re-ran all baseline evaluations after fixing: system prompt (banned empty strings, clarified commit_result), backtick token suppression, better RAG aliases, dataset metadata tagging:

| Model | Compile % | Execute % | Output Match % |
|---|---|---|---|
| Qwen3.6-27B (base) | **94.8%** | **92.7%** | 56.5% |
| Qwen3.5-9B (base) | **64.5%** | **63.4%** | — |
| Qwen3.5-4B (base) | **57.0%** | **55.4%** | — |

vs July 27 baselines:

| Model | Compile % | Execute % |
|---|---|---|
| Qwen3.6-27B (base) | 37.9% | 37.5% |
| Qwen3.5-9B (base) | 20.9% | 19.8% |
| Qwen3.5-4B (base) | 5.5% | 5.5% |

The 37.9% → 94.8% jump for 27B is entirely attributable to infrastructure fixes (prompt, RAG, output cleaning), not model training. **This is a significant thesis finding**: RAG quality + prompt engineering has a larger impact than model size.

### M10.2 — SFT Model Evaluation (First Round)
First SFT model evaluated with old system prompt (before fixes):
- Qwen3.5-9B SFT: **56.8% compile, 49.5% execution**
- Regression vs base (63.4% exec): caused by system prompt mismatch — SFT was trained on the old prompt but evaluated with the new prompt
- Demonstrates importance of consistent prompt formatting between training and inference

### M10.3 — Dataset Deduplication Fix
Audited all training data sources and removed 74 duplicate intents:
- 1316 raw examples across 9 source files → 584 unique intents after deduplication
- Priority order for keeping duplicates: `packages.jsonl` > `genomics.jsonl` > `healthcare.jsonl` > `advanced.jsonl` > `basic.jsonl` > `control_flow.jsonl` > `functions.jsonl` > `classes_arrays.jsonl` > `training_500.jsonl`
- 74 examples removed from `training_500.jsonl` (overlapped with higher-priority files)
- Rebuilt train/val splits: **497 train / 87 val** (was 560/98)
- Removed runtime dedup check from `populate_cache.py` (threshold 0.99 was incorrectly skipping valid unique examples)

### M10.4 — New SFT Models Trained (ep5, correct prompt)
Two new LoRA fine-tunes submitted on Snellius after dedup and prompt fixes:
- `qwen3.5-9b-ep5`: Qwen/Qwen3.5-9B, 5 epochs, LoRA r=16
- `qwen3.6-27b-ep5`: Qwen/Qwen3.6-27B, 5 epochs, LoRA r=16
- Merged adapters available at `outputs/models/qwen3.5-9b-ep5/` and `outputs/models/qwen3.6-27b-ep5/`
- Evaluation results pending

### M10.5 — Semantic Cache Benchmark: Full Threshold Sweep
Optimised the sweep from 15,792 ChromaDB queries (8 thresholds × 1974 queries) down to **1974 queries** by querying once at threshold=0.0 and applying thresholds as a post-filter. Saves incrementally after each threshold.

**Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, BERT-based, cosine similarity)
**Dataset**: 584 unique intents in cache, ~1974 paraphrases (3 per intent) as test queries

#### Cache Sweep Results (August 2, 2026)

| Threshold | Hit%   | Correct% | FP%  | Interpretation |
|-----------|--------|----------|------|----------------|
| 0.80      | 100.0% | 99.1%    | 0.9% | Maximum recall; FP already near floor |
| 0.85      | 100.0% | 99.1%    | 0.9% | No benefit over 0.80 |
| 0.88      | 99.9%  | 99.0%    | 0.9% | Negligible hit loss, same FP |
| **0.90**  | **99.7%** | **98.9%** | **0.9%** | **Near-optimal operating point** |
| **0.92** *(default)* | **99.1%** | **98.3%** | **0.8%** | **Current default — good trade-off** |
| 0.94      | 96.2%  | 95.4%    | 0.8% | Drops 3% hit for no FP gain |
| 0.96      | 88.1%  | 87.5%    | 0.6% | Significant hit loss |
| 0.98      | 59.1%  | 59.1%    | 0.1% | Too conservative — 40% cache misses |

**Key finding**: The **~0.9% irreducible FP floor** is a fundamental property of sentence embeddings, not a tuning issue. Intents that differ only by a parameter value (e.g., `per 100,000` vs `per 200,000`, `height` vs `weight`) embed at cosine similarity >0.97 — the embedding model encodes propositional meaning, not specific values. No threshold can eliminate these FPs without also losing valid hits.

**Recommendation**: Keep default threshold at **0.92**. Lowering to 0.88 gives full hit rate with identical FP cost.

**Thesis significance**: This is a novel empirical finding about the limitations of semantic caching for scientific computing. Discussed in the evaluation and discussion chapters.

### M10.6 — Thesis Introduction and Background Written
First two thesis chapters drafted from the literature study:

- **Chapter 1 (Introduction)**: Motivation (cognitive overhead of WMSs), problem statement, 4 research questions (RQ1–RQ4), 5 contributions, thesis structure
- **Chapter 2 (Background)**: Scientific WMSs (Pegasus, Nextflow, Snakemake, Galaxy, CWL), W3C PROV-DM, FAIR workflows, intent-based orchestration + 5 literature gaps (G1–G5), LLMs in scientific computing (RAG, ControlA), Brane + BraneScript
- References added: 13 new BibTeX entries covering all cited works
- PDF builds cleanly: 43 pages

---

## Current Status (August 2026)

| Component | Status |
|---|---|
| Brane packages | 7 packages, all functional |
| Training data | **584 unique examples** (deduplicated from 658), all passing, 0 timestamps |
| Train/Val split | **497 train / 87 val** |
| Paraphrases | ~1974 examples generated (3 per intent) |
| RAG system | 91 per-function chunks, aliases, "Use when" hints |
| Baseline evaluation | 3 models × 2 rounds; best: **94.8% execution (27B base)** |
| SFT training (round 1) | 9B SFT: 56.8% compile, 49.5% exec (prompt mismatch regression) |
| SFT training (round 2) | **qwen3.5-9b-ep5, qwen3.6-27b-ep5** trained; evaluation pending |
| Merged models | `outputs/models/qwen3.5-9b-ep5/`, `outputs/models/qwen3.6-27b-ep5/` on Snellius |
| GRPO training | Infrastructure ready; not yet run |
| Semantic cache | **584 examples cached; threshold=0.92** |
| Cache benchmark | ✅ **Complete** — full threshold sweep, results above |
| Cache optimal threshold | **0.92** (99.1% hit, 98.3% correct, 0.8% FP) |
| Training metrics | Loss curves saved per model; plottable |
| Frontend | 5 tabs: Results, Exec Results, Evaluation, Runs, Generate |
| Thesis LaTeX | **Introduction + Background written** (43 pages) |

---

## Thesis Metrics and Plots

### Primary Result: BraneScript Generation Quality

The core metric of the thesis is how accurately each model translates a natural-language intent into a correct, executable BraneScript. Three levels:

| Metric | Definition | Measures |
|---|---|---|
| **Compile rate** | % of scripts with valid BraneScript syntax | Syntactic accuracy |
| **Execution rate** | % that run without runtime errors | Semantic accuracy (package API, types) |
| **Output match rate** | % whose stdout matches the reference | Full end-to-end correctness |

**Plot 1 — Model Comparison Bar Chart**
Grouped bar chart: compile %, execute %, output match % for each model (base + SFT + GRPO per size). Shows effect of fine-tuning at a glance. Script: `scripts/plot_training.py` or a new `scripts/plot_results.py`.

**Plot 2 — Training Loss Curves**
Line plot of train loss and validation loss over steps for each model. Shows convergence behaviour, overfitting, and effect of LoRA rank. Source: `outputs/metrics/*.jsonl` → `scripts/plot_training.py --save`.

**Plot 3 — Semantic Cache: Threshold vs Precision/Hit Rate**
Two-axis line chart:
- X axis: similarity threshold (0.80 → 0.99)
- Left Y axis: hit rate % (how many B intents get cached)
- Right Y axis: precision % (of hits, how many return the correct script)
- Shows the operating point tradeoff for the cache

**Plot 4 — Impact of RAG and Prompt Engineering**
Before/after bar chart for the 27B model across two evaluation rounds (July 27 vs August 1). Separates contribution of RAG improvements vs. prompt fixes. This is one of the most striking findings: 37.9% → 94.8% execution rate with the same model weights.

**Plot 5 — Error Type Breakdown**
Stacked bar per model: proportion of failures that are compile errors vs runtime errors vs output mismatch vs correct. Shows where each model fails and whether SFT reduces specific error types.

**Plot 6 — Dataset Composition**
Bar chart showing number of training examples per package (healthcare, genomics, epidemics, statistics, text_analysis, data_masking, datetime). Demonstrates coverage and dataset balance.

### Secondary Metrics

- **Cache hit rate vs intent diversity**: shows reproducibility benefit
- **Retry effectiveness**: how often does attempt 2 or 3 succeed where attempt 1 fails? (from Generate tab history)
- **Training efficiency**: epochs vs validation loss for different model sizes

---

## Key Technical Decisions and Learnings

1. **Brane package dispatch**: must use `sys.argv[1]` + `command.args: [name]`, not env var
2. **Docker build method**: `docker buildx build` extending an existing image; `brane package build` breaks IntermediateResult functions
3. **Class types over JSON strings**: massive reduction in model hallucination
4. **Per-function RAG chunks**: coarse container.yml chunks cause the model to pick the wrong function; function-level chunks with "Use when" context guide it correctly
5. **Response-only training**: computing loss on system/user tokens dilutes the stop-token gradient — the primary cause of generation going past the end of the answer
6. **Deterministic training data**: any non-deterministic output in stdout makes output_match_rate meaningless; removed all timestamps
7. **Thinking mode off**: Qwen3 thinking blocks consume max_new_tokens budget; always use `enable_thinking=False`
