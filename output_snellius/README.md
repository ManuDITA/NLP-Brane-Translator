# output_snellius/

Drop `*_generated.json` files produced by Snellius here, then run:

```bash
python scripts/process_snellius.py
```

That single command will:
1. Find all `*.json` files in this folder
2. Execute each generated BraneScript against the local Brane instance
3. Save full results to `outputs/eval/`
4. Print a comparison table across all models

## Typical Snellius workflow

```bash
# On Snellius — generate only (no Brane available there)
python src/fine_tuning/evaluate.py --model <path> --generate-only

# scp the output back
scp snellius:~/NLP-Brane-Translator/outputs/eval/*_generated.json output_snellius/

# Locally — execute + evaluate everything in one step
python scripts/process_snellius.py

# Options
python scripts/process_snellius.py --timeout 90   # longer timeout for slow scripts
python scripts/process_snellius.py --workers 4    # fewer parallel workers
python scripts/process_snellius.py --force        # re-run even if output exists
python scripts/process_snellius.py --summary      # just print results, no execution
```
