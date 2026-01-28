# TAQ-ICML: Task-Aware Quantization Methods

Three task-aware quantization methods for LLMs:

| Method | Scoring | Allocation |
|--------|---------|------------|
| **TAQ** | Entropy + Normalized Variance | Top 25% → 8-bit, rest → 4-bit |
| **TAQO** | Activation Norm | Top K → FP16, rest → 4-bit |
| **TAQoS** | Output-Sensitive KL Divergence | Top 25% → 8-bit, rest → 4-bit |

---

## Quick Start

```bash
# 1. Create environment
conda create -n taq python=3.10 -y
conda activate taq

# 2. Install dependencies
cd TAQ_ICML
pip install -r requirements.txt

# 3. Run demo
python run_taq_icml.py --run_type demo
```

---

## File Structure

```
TAQ_ICML/
├── run_taq_icml.py           # Main Python script (CLI)
├── taq_icml.ipynb            # Standalone notebook runner
├── requirements.txt          # Python dependencies
├── README.md                # This file
```

---

## Run Types

| Type | Models | Datasets | Methods | Calib | Eval | Purpose |
|------|--------|----------|---------|-------|------|---------|
| `demo` | 1 | 3 | 3 | 512 | 2048 | Quick verification |
| `test` | 5 | 3 | 3 | 32 | 64 | Fast validation |
| `full` | 5 | 3 | 3 | 512 | 2048 | Complete experiments |

---

## CLI Usage

```bash
# Demo run (1 model, 3 datasets, 3 methods)
python run_taq_icml.py --run_type demo

# Test run (all models, fast settings)
python run_taq_icml.py --run_type test

# Full run (all combinations)
python run_taq_icml.py --run_type full

# Specific model/method/dataset
python run_taq_icml.py --run_type full \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --method TAQ \
    --dataset trivia_qa

# Eval-only mode (requires cached model)
python run_taq_icml.py --run_type full \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --method TAQ \
    --dataset trivia_qa \
    --eval_only
```

---

## Models (5)

- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2-7B-Instruct`
- `google/gemma-2-9b-it`
- `meta-llama/Llama-3.1-8B-Instruct`

## Datasets (3)

- **TriviaQA**: Question answering (rc.nocontext, validation)
- **MMLU-Pro**: Multiple choice reasoning (test, stratified)
- **CodeMMLU**: Code understanding (test, stratified)

---

## Results & Caching

Results are saved to:
- `results/` - Experiment summaries (JSON) and detailed outputs (JSONL)
- `saved_models/` - Cached bit allocations (meta.json) for each (model, method, dataset, run_type)
- `datasets_local/` - Downloaded dataset cache

To view results:
```bash
# CLI
python print_results.py

# Or open results_report.ipynb in Jupyter
```

---

## Cluster Usage (SLURM)

For parallel execution on a cluster:

1. Open `commands_copy_paste.ipynb`
2. Run Cell 3 (Full Run) to get 45 tmux+srun commands
3. Copy-paste each command to start a background job

Each command:
- Creates a tmux session named `{model}_{method}_{dataset}_full`
- `cd` to work directory
- Activates conda environment
- Runs `srun` with GPU allocation

---

## Provenance

Each cached model includes `meta.json` with:
- Model name, method, dataset, run_type
- Config hash (for cache invalidation)
- Full RUN_CONFIGS and QUANT_CONFIG used
- Creation timestamp
- Number of layers
