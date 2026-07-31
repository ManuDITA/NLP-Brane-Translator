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

---

## Current Status (July 2026)

| Component | Status |
|---|---|
| Brane packages | 7 packages, all functional |
| Training data | 658 examples, 658/658 passing, 0 timestamps |
| Train/Val split | 560 train / 98 val |
| RAG system | 91 per-function chunks, aliases, "Use when" hints |
| Baseline evaluation | 3 model sizes evaluated |
| SFT training | Completed (Qwen3.6-27B adapter) |
| Response-only training | Fixed (not yet retrained) |
| GRPO training | Infrastructure ready, not yet run |
| Thesis LaTeX | Skeleton in repo, ready for content |

---

## Key Technical Decisions and Learnings

1. **Brane package dispatch**: must use `sys.argv[1]` + `command.args: [name]`, not env var
2. **Docker build method**: `docker buildx build` extending an existing image; `brane package build` breaks IntermediateResult functions
3. **Class types over JSON strings**: massive reduction in model hallucination
4. **Per-function RAG chunks**: coarse container.yml chunks cause the model to pick the wrong function; function-level chunks with "Use when" context guide it correctly
5. **Response-only training**: computing loss on system/user tokens dilutes the stop-token gradient — the primary cause of generation going past the end of the answer
6. **Deterministic training data**: any non-deterministic output in stdout makes output_match_rate meaningless; removed all timestamps
7. **Thinking mode off**: Qwen3 thinking blocks consume max_new_tokens budget; always use `enable_thinking=False`
