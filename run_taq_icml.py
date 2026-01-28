#!/usr/bin/env python
"""
TAQ-ICML: Task-Aware Quantization Methods

Methods:
- TAQ: Matrix-entropy information + raw variance stability -> rank layers -> mixed precision allocation
- TAQO: Oracle per-layer sensitivity (task performance drop when only that layer is low-bit)
- TAQoS: Output-Sensitive KL Divergence -> Top 25% 8-bit, rest 4-bit

Run types:
- full: All 5 models × 3 datasets × 3 methods (calib=512, eval=2048)
- test: Fast validation (calib=32, eval=64)
- demo: 1 model × 1 dataset × 3 methods (calib=512, eval=2048)

Usage:
    python run_taq_icml.py --run_type demo
    python run_taq_icml.py --run_type test
    python run_taq_icml.py --run_type full
    python run_taq_icml.py --run_type full --model "meta-llama/Llama-3.1-8B-Instruct" --method TAQ --dataset trivia_qa

SLURM:
    srun -c 4 -A tdk -p tdk --gres=gpu:1 --pty python run_taq_icml.py --run_type full
"""

import os
import sys
import json
import random
import hashlib
import string
import re
import csv
import math
import time
import traceback
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Iterable
from collections import Counter

# Environment setup
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging

hf_logging.set_verbosity_warning()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
]

DATASETS = ["trivia_qa", "mmlu_pro", "code_mmlu"]
METHODS = ["TAQ", "TAQO", "TAQoS"]

DEMO_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEMO_DATASET = "trivia_qa"

# Run type configurations
RUN_CONFIGS = {
    "full": {"calib_size": 512, "eval_size": 2048, "score_samples": 256, "max_seq_len": 512},
    "test": {"calib_size": 32, "eval_size": 64, "score_samples": 32, "max_seq_len": 256},
    "demo": {"calib_size": 512, "eval_size": 2048, "score_samples": 256, "max_seq_len": 512},
}

# Quantization parameters
QUANT_CONFIG = {
    # Weight quantization
    "qgroup_size": 128,
    "reservoir_size": 256,      # Paper default: 256 token vectors / layer reservoir
    "score_batch_size": 8,
    "taqos_temperature": 1.0,

    # TAQ/TAQoS allocation: Top K% most important layers -> 8-bit, rest -> 4-bit
    "top_8bit_percent": 25,     # Top 25% layers -> 8-bit, rest -> 4-bit

    # TAQO (oracle) controls
    "taqo_eval_samples_test": 16,   # samples used for oracle sensitivity in test
    "taqo_eval_samples_full": 32,   # samples used for oracle sensitivity in demo/full
    "taqo_keep_topk": 8,            # keep top-k sensitive (non-edge) layers at FP16 (maps to threshold gamma)
    "keep_first": 2,            # First 2 layers for TAQO
    "keep_last": 2,             # Last 2 layers for TAQO

    # Dequant cache
    "cache_dequant": True,
}

print("[CONFIG] Configuration loaded", flush=True)


# =============================================================================
# METRICS (TriviaQA / SQuAD style)
# =============================================================================
def normalize_answer(s: str) -> str:
    """
    Official TriviaQA / SQuAD v1.1-style normalization:
      - lowercase
      - replace underscores with spaces
      - remove punctuation
      - remove articles
      - collapse whitespace
    """
    def replace_underscore(text: str) -> str:
        return text.replace("_", " ")

    def lower(text: str) -> str:
        return text.lower()

    def handle_punc(text: str) -> str:
        exclude = set(string.punctuation + "".join(["'", "'", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(
        remove_articles(
            handle_punc(
                lower(
                    replace_underscore(s)
                )
            )
        )
    ).strip()


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 with normalization, as in TriviaQA / SQuAD.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(
    metric_fn: Callable[[str, str], float],
    prediction: str,
    ground_truths: Iterable[str],
) -> float:
    scores = [metric_fn(prediction, gt) for gt in ground_truths]
    if not scores:
        return 0.0
    return max(scores)


# =============================================================================
# DATA LOADING (Exact splits - DO NOT CHANGE)
# =============================================================================
def stratified_disjoint_split(ds, label_col: str, calib_size: int, eval_size: int, seed: int = 42):
    """Stratified split maintaining label proportions."""
    n = len(ds)
    assert calib_size + eval_size <= n, f"calib_size + eval_size ({calib_size + eval_size}) > dataset size ({n})"

    labels = np.array(ds[label_col])
    rng = np.random.default_rng(seed)
    calib_idx, eval_idx = [], []
    unique_labels, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()

    calib_targets = np.floor(proportions * calib_size).astype(int)
    eval_targets = np.floor(proportions * eval_size).astype(int)

    def distribute_leftover(targets, size):
        leftover = size - targets.sum()
        if leftover <= 0:
            return targets
        remainders = (proportions * size) - np.floor(proportions * size)
        order = np.argsort(-remainders)
        for i in order[:leftover]:
            targets[i] += 1
        return targets

    calib_targets = distribute_leftover(calib_targets, calib_size)
    eval_targets = distribute_leftover(eval_targets, eval_size)

    for lab, c_t, e_t in zip(unique_labels, calib_targets, eval_targets):
        idxs = np.flatnonzero(labels == lab)
        rng.shuffle(idxs)
        need = c_t + e_t
        if need > len(idxs):
            raise ValueError(f"Not enough examples in category '{lab}' for calib+eval targets")
        calib_idx.extend(idxs[:c_t].tolist())
        eval_idx.extend(idxs[c_t:c_t + e_t].tolist())

    rng.shuffle(calib_idx)
    rng.shuffle(eval_idx)
    return ds.select(calib_idx), ds.select(eval_idx)


# Default cache folder for datasets
DATASET_CACHE_DIR = "cached_datasets"


def load_trivia_qa(calib_size: int, eval_size: int, data_dir: str):
    """TriviaQA: rc.nocontext, validation split, first calib_size for calib, next eval_size for eval."""
    subset, split = "rc.nocontext", "validation"
    
    # Always load fresh to ensure exact same samples as reference implementation
    print(f"[DATA] TriviaQA: loading from HuggingFace...", flush=True)
    hf_dataset = load_dataset("mandarjoshi/trivia_qa", subset)
    data = hf_dataset[split]
    # Exact split: first calib_size for calib, next eval_size for eval
    calib_data = data.select(range(calib_size))
    eval_data = data.select(range(calib_size, calib_size + eval_size))

    print(f"[DATA] TriviaQA: calib={len(calib_data)}, eval={len(eval_data)}", flush=True)
    return calib_data, eval_data


def load_mmlu_pro(calib_size: int, eval_size: int, data_dir: str):
    """MMLU-Pro: test split, stratified by category."""
    # Always load fresh to ensure exact same samples as reference implementation
    print(f"[DATA] MMLU-Pro: loading from HuggingFace...", flush=True)
    hf_dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    data = hf_dataset["test"]
    calib_data, eval_data = stratified_disjoint_split(
        data, label_col="category", calib_size=calib_size, eval_size=eval_size, seed=42
    )

    print(f"[DATA] MMLU-Pro: calib={len(calib_data)}, eval={len(eval_data)}", flush=True)
    return calib_data, eval_data


def load_code_mmlu(calib_size: int, eval_size: int, data_dir: str):
    """CodeMMLU: test split, stratified by subset."""
    # Always load fresh to ensure exact same samples as reference implementation
    print(f"[DATA] CodeMMLU: loading from HuggingFace...", flush=True)
    subsets = [
        'api_frameworks', 'code_completion', 'code_repair', 'dbms_sql',
        'execution_prediction', 'fill_in_the_middle', 'programming_syntax', 'software_principles'
    ]
    ds_list = []
    for subset in subsets:
        ds = load_dataset("Fsoft-AIC/CodeMMLU", name=subset)["test"]
        ds = ds.add_column("subset", [subset] * len(ds))
        ds_list.append(ds)
    data = concatenate_datasets(ds_list)
    calib_data, eval_data = stratified_disjoint_split(
        data, label_col="subset", calib_size=calib_size, eval_size=eval_size, seed=42
    )

    print(f"[DATA] CodeMMLU: calib={len(calib_data)}, eval={len(eval_data)}", flush=True)
    return calib_data, eval_data


DATASET_LOADERS = {
    "trivia_qa": load_trivia_qa,
    "mmlu_pro": load_mmlu_pro,
    "code_mmlu": load_code_mmlu
}


# =============================================================================
# MODEL UTILITIES
# =============================================================================
def _resolve_device(dev: str) -> str:
    return "cuda:0" if dev == "auto" and torch.cuda.is_available() else ("cpu" if dev == "auto" else dev)


def sanitize_generation_config(model):
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    if hasattr(gc, "do_sample"):
        gc.do_sample = False
    for k in ("temperature", "top_p", "top_k", "typical_p", "penalty_alpha"):
        if hasattr(gc, k):
            setattr(gc, k, None)


def load_model_and_tokenizer(model_name: str, device: str, token=None):
    device = _resolve_device(device)
    token = token or os.environ.get("HF_TOKEN", None)

    tok_kwargs = {"trust_remote_code": True}
    mdl_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    if token:
        tok_kwargs["token"] = token
        mdl_kwargs["token"] = token

    print(f"[MODEL] Loading: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, **mdl_kwargs).eval()

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    sanitize_generation_config(model)
    return model, tokenizer


def get_num_layers(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return len(model.model.decoder.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    return 0


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    return None


# =============================================================================
# QUANTIZATION MODULE (Vectorized for speed)
# =============================================================================
class QuantLinearWB(nn.Module):
    """Weight-only per-group affine quantization for wbit in {4, 8}. Vectorized implementation."""
    def __init__(self, linear: nn.Linear, wbit: int = 4, group_size: int = 128, cache_dequant: bool = True):
        super().__init__()
        assert wbit in (4, 8)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None

        W = linear.weight.detach().to(torch.float32).cpu()
        self.wbit, self.group_size, self.cache_dequant = int(wbit), int(group_size), bool(cache_dequant)
        self._W_cache = None

        G = max(1, self.group_size)
        out_features, in_features = W.shape
        in_groups = (in_features + G - 1) // G
        Q = 15 if self.wbit == 4 else 255

        # Pad input dimension to be divisible by group_size
        pad_size = in_groups * G - in_features
        if pad_size > 0:
            W = F.pad(W, (0, pad_size), value=0)

        # Reshape to [out_features, in_groups, G] for vectorized operations
        W_grouped = W.view(out_features, in_groups, G)

        # Compute min/max per group (vectorized)
        wmin = W_grouped.min(dim=2).values  # [out_features, in_groups]
        wmax = W_grouped.max(dim=2).values  # [out_features, in_groups]

        # Compute scales and zero points (vectorized)
        range_vals = wmax - wmin
        # Handle constant groups (range ~= 0)
        range_vals = torch.where(range_vals < 1e-8, torch.ones_like(range_vals), range_vals)
        scales = range_vals / Q
        zeros = -wmin / (scales + 1e-12)

        # Quantize (vectorized)
        qvals = torch.clamp(torch.round(W_grouped / (scales.unsqueeze(2) + 1e-12) + zeros.unsqueeze(2)), 0, Q).to(torch.uint8)

        # Pack codes
        if self.wbit == 4:
            # Pack two 4-bit values into one byte: [out, groups, G] -> [out, groups, G//2]
            packed = (qvals[:, :, 0::2] << 4) | qvals[:, :, 1::2]
            codes = packed.contiguous()
        else:
            codes = qvals.contiguous()

        self.register_buffer("qcodes", codes)
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("zeros", zeros.contiguous())

    def _dequant_weight_to(self, device):
        """Vectorized dequantization."""
        out, in_groups, code_dim = self.qcodes.shape
        G = self.group_size

        # Move buffers to device
        qcodes = self.qcodes.to(device)
        scales = self.scales.to(device, dtype=torch.float32)
        zeros = self.zeros.to(device, dtype=torch.float32)

        if self.wbit == 4:
            # Unpack: [out, groups, G//2] -> [out, groups, G]
            q = torch.empty((out, in_groups, G), dtype=torch.float32, device=device)
            q[:, :, 0::2] = ((qcodes >> 4) & 0xF).float()
            q[:, :, 1::2] = (qcodes & 0xF).float()
        else:
            q = qcodes.to(dtype=torch.float32)

        # Dequantize: (q - zero) * scale, vectorized
        Wfull = ((q - zeros.unsqueeze(2)) * scales.unsqueeze(2)).to(torch.float16)
        # Reshape back to [out, in_groups * G] and trim to original size
        Wfull = Wfull.view(out, -1)[:, :self.in_features]
        return Wfull

    def forward(self, x):
        if self.cache_dequant and self._W_cache is not None and self._W_cache.device == x.device:
            return F.linear(x, self._W_cache, self.bias)
        W = self._dequant_weight_to(x.device)
        if self.cache_dequant and not self.training:
            self._W_cache = W
        return F.linear(x, W, self.bias)


def quantize_with_bit_map(model, bit_map: List[int], group_size: int = 128, cache_dequant: bool = True):
    """Quantize model layers according to bit_map. bit_map values: 4, 8, 16."""
    layers = get_transformer_layers(model)
    if layers is None:
        print("[QUANT] Warning: Could not find layers", flush=True)
        return model
    nL = len(layers)
    replaced = 0
    for li in range(nL):
        bits = bit_map[li]
        if bits == 16:
            continue
        blk = layers[li]
        for name, mod in list(blk.named_modules()):
            if isinstance(mod, nn.Linear):
                parent = blk
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], QuantLinearWB(mod, wbit=bits, group_size=group_size, cache_dequant=cache_dequant))
                replaced += 1
        if (li + 1) % 5 == 0 or li == nL - 1:
            print(f"[QUANT] {100 * (li + 1) / nL:.0f}%", flush=True)
    print(f"[QUANT] Replaced {replaced} Linear modules", flush=True)
    return model


# =============================================================================
# CACHING SYSTEM
# =============================================================================
def compute_config_hash(model_name: str, method: str, dataset: str, run_type: str, quant_config: dict) -> str:
    """Compute stable hash for caching quantized models."""
    config = {
        "model": model_name,
        "method": method,
        "dataset": dataset,
        "run_type": run_type,
        "qgroup_size": quant_config.get("qgroup_size", 128),
        "reservoir_size": quant_config.get("reservoir_size", 256),
        "score_samples": RUN_CONFIGS[run_type]["score_samples"],
        # TAQ/TAQO/TAQoS top-K allocation
        "top_8bit_percent": quant_config.get("top_8bit_percent", 25),
        "taqo_keep_topk": quant_config.get("taqo_keep_topk", 8),
    }
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha1(config_str.encode()).hexdigest()[:16]


def get_cache_path(cache_dir: str, config_hash: str) -> str:
    return os.path.join(cache_dir, config_hash)


def save_cached_config(cache_path: str, bit_map: List[int], scores: dict, meta: dict):
    """Save bit map and metadata to cache."""
    os.makedirs(cache_path, exist_ok=True)
    with open(os.path.join(cache_path, "bit_map.json"), "w") as f:
        json.dump(bit_map, f)
    with open(os.path.join(cache_path, "scores.json"), "w") as f:
        json.dump(scores, f, indent=2)
    with open(os.path.join(cache_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[CACHE] Saved to {cache_path}", flush=True)


def load_cached_config(cache_path: str) -> Tuple[List[int], dict, dict]:
    """Load bit map and metadata from cache."""
    with open(os.path.join(cache_path, "bit_map.json"), "r") as f:
        bit_map = json.load(f)
    scores_path = os.path.join(cache_path, "scores.json")
    scores = json.load(open(scores_path)) if os.path.exists(scores_path) else {}
    with open(os.path.join(cache_path, "meta.json"), "r") as f:
        meta = json.load(f)
    print(f"[CACHE] Loaded from {cache_path}", flush=True)
    return bit_map, scores, meta


def check_cache_exists(cache_path: str) -> bool:
    return os.path.exists(os.path.join(cache_path, "bit_map.json"))


def get_quantized_model_path(cache_path: str) -> str:
    """Get path for the quantized model weights file."""
    return os.path.join(cache_path, "quantized_model.pt")


def check_quantized_model_exists(cache_path: str) -> bool:
    """Check if a cached quantized model exists."""
    return os.path.exists(get_quantized_model_path(cache_path))


def save_quantized_model(model, cache_path: str):
    """
    Save the quantized model state dict to cache.
    Only saves the quantized layers (QuantLinearWB) state to save space.
    """
    os.makedirs(cache_path, exist_ok=True)
    model_path = get_quantized_model_path(cache_path)
    
    # Save the full state dict - PyTorch will handle QuantLinearWB buffers
    state_dict = model.state_dict()
    torch.save(state_dict, model_path)
    
    # Calculate and print size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"[CACHE] Saved quantized model to {model_path} ({size_mb:.1f} MB)", flush=True)


def load_quantized_model(base_model, bit_map: List[int], cache_path: str, 
                         group_size: int = 128, cache_dequant: bool = True):
    """
    Load a cached quantized model.
    First applies quantization structure (creates QuantLinearWB layers),
    then loads the saved state dict.
    """
    model_path = get_quantized_model_path(cache_path)
    print(f"[CACHE] Loading quantized model from {model_path}...", flush=True)
    
    # First, apply the quantization structure (this creates QuantLinearWB layers)
    # but with freshly quantized weights
    q_model = quantize_with_bit_map(base_model, bit_map, group_size, cache_dequant)
    
    # Now load the saved state dict to restore exact quantized values
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    q_model.load_state_dict(state_dict)
    
    print(f"[CACHE] Quantized model loaded successfully", flush=True)
    return q_model


# =============================================================================
# PROMPT FORMATTING FOR SCORING (NO TEACHER FORCING)
# =============================================================================
def format_trivia_qa_score_prompts(dataset):
    texts = []
    for ex in dataset:
        q = ex.get("question", "")
        texts.append(f"Question: {q}\nAnswer:")
    return texts


def format_mmlu_pro_score_prompts(dataset):
    texts = []
    for ex in dataset:
        q = ex.get("question", "")
        options = ex.get("options", [])
        opts_str = "\n".join([f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)])
        texts.append(f"Question: {q}\n{opts_str}\nAnswer:")
    return texts


def format_code_mmlu_score_prompts(dataset):
    texts = []
    for ex in dataset:
        q = ex.get("question", "")
        options = ex.get("choices", ex.get("options", []))
        opts_str = "\n".join([f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)])
        texts.append(f"Question: {q}\n{opts_str}\nAnswer:")
    return texts


SCORE_PROMPT_FORMATTERS = {
    "trivia_qa": format_trivia_qa_score_prompts,
    "mmlu_pro": format_mmlu_pro_score_prompts,
    "code_mmlu": format_code_mmlu_score_prompts,
}


# =============================================================================
# TAQ SCORING: Matrix-entropy information + raw variance stability (paper)
# =============================================================================
def _zscore(xs: List[float]) -> List[float]:
    if not xs:
        return xs
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
    s = math.sqrt(v + 1e-12)
    return [(x - m) / s for x in xs]


class _Reservoir:
    """Fixed-size reservoir of token vectors for a layer (CPU float32)."""
    def __init__(self, capacity: int):
        self.capacity = int(max(1, capacity))
        self.buf: Optional[torch.Tensor] = None  # [m, d] float32 CPU
        self.seen = 0  # total candidates processed

    def add_candidates(self, X: torch.Tensor):
        """
        X: [n, d] float32 CPU
        """
        if X is None or X.numel() == 0:
            return
        if X.device.type != "cpu":
            X = X.detach().cpu()
        if X.dtype != torch.float32:
            X = X.float()

        n, d = X.shape
        if self.buf is None:
            take = min(self.capacity, n)
            self.buf = X[:take].contiguous()
            self.seen = take
            if take < n:
                rest = X[take:]
                self._reservoir_update(rest)
            return

        self._reservoir_update(X)

    def _reservoir_update(self, X: torch.Tensor):
        # Standard reservoir sampling update.
        # For each candidate at global index t (1-based), replace with prob k/t.
        k = self.capacity
        for i in range(X.shape[0]):
            self.seen += 1
            t = self.seen
            if self.buf.shape[0] < k:
                self.buf = torch.cat([self.buf, X[i:i+1]], dim=0)
                continue
            j = random.randint(1, t)  # 1..t
            if j <= k:
                self.buf[j - 1] = X[i]

    def get(self) -> Optional[torch.Tensor]:
        return self.buf


@torch.no_grad()
def score_layers_taq(model, tokenizer, texts, max_tokens, device, reservoir_size=256, batch_size=8):
    """
    Paper TAQ scoring (Eq. 3–5):

    Information: sample a reservoir {v_i}_{i=1..r_l} across calibration prompts.
      Build centered Z_l (r_l x d), define Gram K_l = (1/r_l) Z_l Z_l^T,
      and H_l = -sum_i \tilde{lambda}_i log \tilde{lambda}_i (matrix entropy).

    Stability: streaming moments over all token activations:
      Var_l = E[x^2] - (E[x])^2  (over all elements in X_l across valid tokens)
      S_l = -Var_l  (Eq. 4)

    Returns:
      info_scores [nL], stability_scores [nL]
    """
    device = _resolve_device(device)
    model.eval()
    nL = get_num_layers(model)

    reservoirs = [_Reservoir(reservoir_size) for _ in range(nL)]

    # streaming moments for raw variance over all activation elements
    sum_x = [0.0] * nL
    sum_x2 = [0.0] * nL
    n_elem = [0] * nL

    B = max(1, int(batch_size))
    total = len(texts)
    t0 = time.time()

    for start in range(0, total, B):
        batch = texts[start:start + B]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens).to(device)

        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple: embeddings + layers
        attn_mask = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))

        if (start + B) % (B * 4) == 0 or start + B >= total:
            print(f"[TAQ] Scoring: {100 * min(total, start + B) / total:.0f}% ({time.time() - t0:.1f}s)", flush=True)

        # layers: hs[1]..hs[nL]
        Bx, Tx = attn_mask.shape
        valid_mask_flat = attn_mask.reshape(Bx * Tx).bool()

        for li in range(1, len(hs)):
            X = hs[li]  # [B, T, D]
            Bx, Tx, Dx = X.shape

            X_flat = X.reshape(Bx * Tx, Dx)
            X_valid = X_flat[valid_mask_flat].float()  # [n_valid, D]

            if X_valid.numel() == 0:
                continue

            # Stability moments (raw variance over ALL elements)
            sum_x[li - 1] += float(X_valid.sum().item())
            sum_x2[li - 1] += float((X_valid * X_valid).sum().item())
            n_elem[li - 1] += int(X_valid.numel())

            # Information reservoir: sample token vectors across all positions (prompt tokens)
            # To keep runtime bounded, subsample candidates per batch before reservoir update.
            # This still matches the "small reservoir sampled across C" requirement.
            max_candidates = 128
            if X_valid.shape[0] > max_candidates:
                idx = torch.randperm(X_valid.shape[0], device=X_valid.device)[:max_candidates]
                cand = X_valid[idx]
            else:
                cand = X_valid

            reservoirs[li - 1].add_candidates(cand.detach().cpu().to(torch.float32))

    info_scores: List[float] = []
    stability_scores: List[float] = []

    for li in range(nL):
        # --- Information (matrix entropy on Gram spectrum) ---
        R = reservoirs[li].get()
        if R is None or R.numel() == 0 or R.shape[0] < 2:
            H = 0.0
        else:
            # Center: Z = R - mean
            Z = R - R.mean(dim=0, keepdim=True)

            # Use covariance trick: eigenvalues of (Z Z^T) and (Z^T Z) share non-zero spectrum up to scaling.
            # Entropy uses normalized spectrum, so scaling cancels.
            try:
                n = Z.shape[0]
                C = (Z.T @ Z) / max(1, n)  # [D, D], PSD
                evals = torch.linalg.eigvalsh(C).clamp(min=0)
                s = float(evals.sum().item())
                if s <= 0:
                    H = 0.0
                else:
                    p = evals / (s + 1e-12)
                    H = float((-(p * (p + 1e-12).log())).sum().item())
            except Exception:
                H = 0.0
        info_scores.append(H)

        # --- Stability (raw variance) ---
        if n_elem[li] > 0:
            mean = sum_x[li] / n_elem[li]
            var = max(0.0, (sum_x2[li] / n_elem[li]) - mean * mean)
            stability_scores.append(-float(var))
        else:
            stability_scores.append(0.0)

    return info_scores, stability_scores


def taq_allocate_bits(
    info_scores: List[float],
    stability_scores: List[float],
    keep_first: int = 2,
    keep_last: int = 2,
    high16_percent: float = 15.0,
    mid8_percent: float = 45.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    top_k_8bit_percent: float = 25.0,
):
    """
    TAQ allocation using top-K approach:
      - Rank all layers by R_l = alpha*z(H_l) + beta*z(S_l)
      - Top K% (default 25%) most important layers -> 8-bit
      - Rest -> 4-bit
    """
    nL = len(info_scores)

    Iz = _zscore(info_scores)
    Sz = _zscore(stability_scores)
    combined = [alpha * Iz[i] + beta * Sz[i] for i in range(nL)]

    # Calculate number of 8-bit layers (top 25%)
    n_8bit = max(1, int(math.ceil(nL * top_k_8bit_percent / 100.0)))
    
    # Sort all layers by combined score (descending - higher is more important)
    sorted_indices = sorted(range(nL), key=lambda i: combined[i], reverse=True)

    # Allocate bits: top K% get 8-bit, rest get 4-bit
    bit_map = [4] * nL
    for i in sorted_indices[:n_8bit]:
        bit_map[i] = 8

    return bit_map, {"info": info_scores, "stability": stability_scores, "combined": combined}


# =============================================================================
# TAQO (Oracle) SCORING: Per-layer sensitivity via task performance drop
# =============================================================================
_ANSWER_PREFIXES = ("answer:", "final answer:", "final:", "a:", "ans:", "prediction:", "the answer is")


def extract_pred_answer(generated_text: str) -> str:
    if not generated_text:
        return ""
    first = next((ln.strip() for ln in generated_text.strip().splitlines() if ln.strip()), "")
    low = first.lower()
    for pref in _ANSWER_PREFIXES:
        if low.startswith(pref):
            first = first[len(pref):].strip()
            break
    for stop in [".", ";", "—", "–", "|"]:
        if stop in first:
            first = first.split(stop, 1)[0].strip()
            break
    return first.strip(" '\"`")


def get_gold_answers(sample, dataset_type):
    golds = []
    if dataset_type == "trivia_qa":
        ans = sample.get("answer")
        if isinstance(ans, dict):
            if ans.get("value"):
                golds.append(ans["value"])
            for a in ans.get("aliases", []):
                if a:
                    golds.append(a)
        elif isinstance(ans, str) and ans:
            golds.append(ans)
    elif dataset_type in ["mmlu_pro", "code_mmlu"]:
        ans = sample.get("answer", "")
        if ans:
            golds.append(ans)
    return list(dict.fromkeys(golds)) or [""]


def _build_eval_prompt(sample, dataset_type: str) -> str:
    if dataset_type == "trivia_qa":
        q = sample.get("question", "")
        return f"Question: {q}\nAnswer:"
    else:
        q = sample.get("question", "")
        options = sample.get("choices", sample.get("options", []))
        opts_str = "\n".join([f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)])
        return f"Question: {q}\n{opts_str}\nAnswer:"


@torch.no_grad()
def _evaluate_on_subset(model, tokenizer, dataset, dataset_type: str, max_seq_len: int, device: str, max_new_tokens: int = 1024):
    """
    Returns (em, f1, n). For MMLU/CodeMMLU, "f1" is computed with the same string metric.
    """
    device = _resolve_device(device)
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = torch.device("cpu")

    total, em_sum, f1_sum = 0, 0.0, 0.0
    for sample in dataset:
        prompt = _build_eval_prompt(sample, dataset_type)
        golds = get_gold_answers(sample, dataset_type)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len).to(model_device)

        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        gen = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_pred_answer(gen)
        em = metric_max_over_ground_truths(exact_match_score, pred, golds)
        f1 = metric_max_over_ground_truths(f1_score, pred, golds)

        total += 1
        em_sum += em
        f1_sum += f1

    if total == 0:
        return 0.0, 0.0, 0
    return em_sum / total, f1_sum / total, total


def _swap_layer_linear_modules(layer: nn.Module, wbit: int, group_size: int, cache_dequant: bool):
    """
    Replace all nn.Linear under `layer` with QuantLinearWB at `wbit`.
    Returns list of (parent_module, attr_name, original_module) for restoration.
    """
    replaced = []
    for name, mod in list(layer.named_modules()):
        if isinstance(mod, nn.Linear):
            parent = layer
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]
            orig = getattr(parent, attr)
            setattr(parent, attr, QuantLinearWB(orig, wbit=wbit, group_size=group_size, cache_dequant=cache_dequant))
            replaced.append((parent, attr, orig))
    return replaced


def _restore_swapped_modules(replaced):
    for parent, attr, orig in replaced:
        setattr(parent, attr, orig)


@torch.no_grad()
def score_layers_taqo_oracle(
    model,
    tokenizer,
    calib_eval_subset,
    dataset_type: str,
    max_seq_len: int,
    device: str,
    low_bit: int = 4,
    group_size: int = 128,
    cache_dequant: bool = True,
):
    """
    Paper TAQO oracle sensitivity:
      Δ_l = performance_drop when ONLY layer l is quantized to low precision (others FP16).
    We compute Δ_l using the same task metric pipeline on a small calibration-eval subset.
    """
    device = _resolve_device(device)
    model.eval()
    layers = get_transformer_layers(model)
    if layers is None:
        raise RuntimeError("Could not locate transformer layers for TAQO.")

    nL = len(layers)

    print("[TAQO] Oracle Phase 0: baseline evaluation...", flush=True)
    base_em, base_f1, n = _evaluate_on_subset(model, tokenizer, calib_eval_subset, dataset_type, max_seq_len, device)
    base_score = 0.5 * (base_em + base_f1)
    print(f"[TAQO] Baseline: EM={base_em:.4f} F1={base_f1:.4f} (n={n})", flush=True)

    deltas = [0.0] * nL

    for li in range(nL):
        if (li + 1) % 5 == 0 or li == 0 or li == nL - 1:
            print(f"[TAQO] Sensitivity layer {li+1}/{nL}", flush=True)

        # swap only this layer's linears to low-bit
        replaced = _swap_layer_linear_modules(layers[li], wbit=low_bit, group_size=group_size, cache_dequant=cache_dequant)
        try:
            em, f1, _ = _evaluate_on_subset(model, tokenizer, calib_eval_subset, dataset_type, max_seq_len, device)
            score = 0.5 * (em + f1)
            deltas[li] = max(0.0, base_score - score)
        finally:
            _restore_swapped_modules(replaced)
            # clear any cached dequant weights that might persist
            torch.cuda.empty_cache()

    return deltas, {"baseline_em": base_em, "baseline_f1": base_f1, "baseline_score": base_score}


def taqo_allocate_bits_oracle(
    deltas: List[float],
    keep_first: int = 2,
    keep_last: int = 2,
    keep_topk: int = 8,
):
    """
    TAQO allocation (Eq. 7–8) implemented via a top-k threshold:
      - Always keep first/last layers at FP16
      - Among remaining, keep the top-k layers by Δ_l at FP16
      - All others -> 4-bit
    """
    nL = len(deltas)
    keep_first = int(max(0, min(keep_first, nL)))
    keep_last = int(max(0, min(keep_last, nL - keep_first)))
    edge_keep = set(range(keep_first)) | set(range(max(0, nL - keep_last), nL))

    mid = [i for i in range(nL) if i not in edge_keep]
    mid_sorted = sorted(mid, key=lambda i: deltas[i], reverse=True)

    keep_topk = int(max(0, keep_topk))
    chosen = set(edge_keep)
    for i in mid_sorted[:min(keep_topk, len(mid_sorted))]:
        chosen.add(i)

    bit_map = [4] * nL
    for i in chosen:
        bit_map[i] = 16

    return bit_map, {"delta": deltas}


# =============================================================================
# TAQoS SCORING: Output-Sensitive KL Divergence (unchanged)
# =============================================================================
@torch.no_grad()
def score_layers_taqos(model, tokenizer, texts, max_tokens, device, temperature=1.0):
    """
    TAQoS: Score layers by KL divergence between original and noise-injected outputs.
    """
    device = _resolve_device(device)
    model.eval()
    nL = get_num_layers(model)

    layer_ranges = [[] for _ in range(nL)]
    baseline_logits_all = []

    print("[TAQoS] Phase 1: Computing baselines...", flush=True)
    t0 = time.time()

    for i, text in enumerate(texts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)

        out = model(**enc, output_hidden_states=True, use_cache=False)
        baseline_logits = out.logits[:, -1, :].float()
        baseline_logits_all.append(baseline_logits.cpu())

        hs = out.hidden_states
        for li in range(1, len(hs)):
            h = hs[li][:, -1, :].float()
            h_range = (h.max() - h.min()).item()
            layer_ranges[li - 1].append(h_range)

        if (i + 1) % 20 == 0:
            print(f"  {100*(i+1)/len(texts):.0f}%", flush=True)

    delta = []
    for li in range(nL):
        avg_range = sum(layer_ranges[li]) / max(1, len(layer_ranges[li]))
        delta.append(avg_range / 15)

    print(f"[TAQoS] Phase 2: Computing KL divergence... ({time.time()-t0:.1f}s)", flush=True)

    kl_scores = [0.0] * nL

    def make_noise_hook(noise_scale):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            noise = torch.empty_like(h).uniform_(-noise_scale/2, noise_scale/2)
            h_noised = h + noise
            if isinstance(output, tuple):
                return (h_noised,) + output[1:]
            return h_noised
        return hook

    layers = get_transformer_layers(model)

    for li in range(nL):
        if (li + 1) % 5 == 0 or li == 0 or li == nL - 1:
            print(f"  Layer {li+1}/{nL}", flush=True)

        kl_sum = 0.0

        for i, text in enumerate(texts):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
            baseline_logits = baseline_logits_all[i].to(device)

            handle = layers[li].register_forward_hook(make_noise_hook(delta[li]))
            try:
                out = model(**enc, use_cache=False)
                noised_logits = out.logits[:, -1, :].float()

                p = F.softmax(baseline_logits / temperature, dim=-1)
                q = F.softmax(noised_logits / temperature, dim=-1)
                kl = F.kl_div(q.log(), p, reduction='batchmean').item()
                kl_sum += kl
            finally:
                handle.remove()

        kl_scores[li] = kl_sum / max(1, len(texts))

    print(f"[TAQoS] Done ({time.time()-t0:.1f}s)", flush=True)
    return kl_scores


def taqos_allocate_bits(kl_scores, top_8bit_percent=25):
    """TAQoS bit allocation: Higher KL -> more sensitive -> 8-bit."""
    nL = len(kl_scores)
    n_8bit = max(1, int(math.ceil(nL * top_8bit_percent / 100)))
    sorted_indices = sorted(range(nL), key=lambda i: kl_scores[i], reverse=True)

    bit_map = [4] * nL
    for i in sorted_indices[:n_8bit]:
        bit_map[i] = 8

    return bit_map, {"kl": kl_scores}


# =============================================================================
# CALIBRATION FORMATTING (kept for legacy; not used for scoring anymore)
# =============================================================================
def format_trivia_qa_calib(dataset):
    texts = []
    for ex in dataset:
        q = ex.get("question", "")
        ans = ex.get("answer")
        a = ans.get("value", "") if isinstance(ans, dict) else (ans if isinstance(ans, str) else "")
        texts.append(f"Question: {q}\nAnswer: {a}")
    return texts


def format_mmlu_pro_calib(dataset):
    texts = []
    for ex in dataset:
        q = ex.get("question", "")
        options = ex.get("options", [])
        opts_str = "\n".join([f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)])
        texts.append(f"Question: {q}\n{opts_str}\nAnswer: {ex.get('answer', '')}")
    return texts


def format_code_mmlu_calib(dataset):
    texts = []
    for ex in dataset:
        q = ex.get("question", "")
        options = ex.get("choices", ex.get("options", []))
        opts_str = "\n".join([f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)])
        texts.append(f"Question: {q}\n{opts_str}\nAnswer: {ex.get('answer', '')}")
    return texts


CALIB_FORMATTERS = {
    "trivia_qa": format_trivia_qa_calib,
    "mmlu_pro": format_mmlu_pro_calib,
    "code_mmlu": format_code_mmlu_calib
}


# =============================================================================
# EVALUATION (unchanged)
# =============================================================================
def evaluate_model(model, tokenizer, eval_data, dataset_type, gen_len, max_seq_len, device):
    device = _resolve_device(device)
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = torch.device("cpu")

    total, em_sum, f1_sum, results = 0, 0.0, 0.0, []
    t0 = time.time()

    for i, sample in enumerate(eval_data):
        if dataset_type == "trivia_qa":
            q = sample.get("question", "")
            prompt = f"Question: {q}\nAnswer:"
        else:
            q = sample.get("question", "")
            options = sample.get("choices", sample.get("options", []))
            opts_str = "\n".join([f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)])
            prompt = f"Question: {q}\n{opts_str}\nAnswer:"

        golds = get_gold_answers(sample, dataset_type)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len).to(model_device)

        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=gen_len,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        gen = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_pred_answer(gen)
        em = metric_max_over_ground_truths(exact_match_score, pred, golds)
        f1 = metric_max_over_ground_truths(f1_score, pred, golds)

        total += 1
        em_sum += em
        f1_sum += f1
        results.append({"question": q[:200], "golds": golds, "pred": pred, "em": em, "f1": f1})

        if (i + 1) % 100 == 0 or i + 1 == len(eval_data):
            print(f"[EVAL] {i+1}/{len(eval_data)}: EM={em_sum/total:.3f} F1={f1_sum/total:.3f} ({time.time()-t0:.1f}s)", flush=True)

    return (em_sum / total if total else 0.0), (f1_sum / total if total else 0.0), total, results


def save_results(results_dir, model_name, method, dataset_name, run_type, em, f1, n, detailed_results, bit_map, scores):
    os.makedirs(results_dir, exist_ok=True)
    model_safe = model_name.replace("/", "_")

    with open(os.path.join(results_dir, f"{model_safe}_{method}_{dataset_name}_{run_type}_results.jsonl"), "w") as f:
        for r in detailed_results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "model": model_name,
        "method": method,
        "dataset": dataset_name,
        "run_type": run_type,
        "em": em,
        "f1": f1,
        "n_examples": n,
        "em_percent": em * 100,
        "f1_percent": f1 * 100,
        "bit_map": bit_map,
        "scores": scores,
        "bits_summary": {
            "4-bit": bit_map.count(4),
            "8-bit": bit_map.count(8),
            "16-bit": bit_map.count(16),
        }
    }

    summary_path = os.path.join(results_dir, f"{model_safe}_{method}_{dataset_name}_{run_type}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SAVE] Results saved to {results_dir}", flush=True)
    return summary


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================
def run_single_experiment(model_name, method, dataset_name, run_type, device, data_dir, results_dir, cache_dir, hf_token=None, eval_only=False):
    """Run a single model-method-dataset combination."""
    run_config = RUN_CONFIGS[run_type]
    calib_size = run_config["calib_size"]
    eval_size = run_config["eval_size"]
    score_samples = run_config["score_samples"]
    max_seq_len = run_config["max_seq_len"]

    print(f"\n{'='*70}", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"Method: {method}", flush=True)
    print(f"Dataset: {dataset_name}", flush=True)
    print(f"Run type: {run_type}", flush=True)
    print(f"{'='*70}", flush=True)

    # Check for existing results
    model_safe = model_name.replace("/", "_")
    results_path = os.path.join(results_dir, f"{model_safe}_{method}_{dataset_name}_{run_type}_summary.json")
    if os.path.exists(results_path):
        print(f"[SKIP] Results exist: {results_path}", flush=True)
        with open(results_path) as f:
            return json.load(f)

    # Check for cached quantized model config
    config_hash = compute_config_hash(model_name, method, dataset_name, run_type, QUANT_CONFIG)
    cache_path = get_cache_path(cache_dir, config_hash)

    # In eval_only mode, require cache to exist
    if eval_only:
        if not check_cache_exists(cache_path):
            raise ValueError(f"[EVAL_ONLY] No cached config found at {cache_path}. Run without --eval_only first.")
        print(f"[EVAL_ONLY] Loading cached bit_map...", flush=True)
        bit_map, scores, meta = load_cached_config(cache_path)

        # Load model for evaluation only
        base_model, tokenizer = load_model_and_tokenizer(model_name, device, hf_token)
        n_layers = get_num_layers(base_model)
        print(f"[MODEL] {n_layers} layers", flush=True)

        # Load eval data only
        calib_data, eval_data = DATASET_LOADERS[dataset_name](calib_size, eval_size, data_dir)
    else:
        # Load model
        base_model, tokenizer = load_model_and_tokenizer(model_name, device, hf_token)
        n_layers = get_num_layers(base_model)
        print(f"[MODEL] {n_layers} layers", flush=True)

        # Load data
        calib_data, eval_data = DATASET_LOADERS[dataset_name](calib_size, eval_size, data_dir)

        # IMPORTANT: scoring prompts = NO teacher forcing
        calib_prompts = SCORE_PROMPT_FORMATTERS[dataset_name](calib_data)[:score_samples]

        # Check cache
        if check_cache_exists(cache_path):
            print(f"[CACHE] Loading cached bit_map...", flush=True)
            bit_map, scores, meta = load_cached_config(cache_path)
        else:
            # Score layers based on method
            if method == "TAQ":
                print("[TAQ] Scoring layers (paper)...", flush=True)
                info_scores, stability_scores = score_layers_taq(
                    base_model,
                    tokenizer,
                    calib_prompts,
                    max_seq_len,
                    device,
                    reservoir_size=QUANT_CONFIG["reservoir_size"],
                    batch_size=QUANT_CONFIG["score_batch_size"],
                )
                bit_map, scores = taq_allocate_bits(
                    info_scores,
                    stability_scores,
                    alpha=0.5,
                    beta=0.5,
                    top_k_8bit_percent=QUANT_CONFIG["top_8bit_percent"],
                )

            elif method == "TAQO":
                print("[TAQO] Oracle sensitivity scoring (paper)...", flush=True)

                # Build a small subset of calibration data for sensitivity eval
                if run_type == "test":
                    n_sens = min(QUANT_CONFIG["taqo_eval_samples_test"], len(calib_data))
                else:
                    n_sens = min(QUANT_CONFIG["taqo_eval_samples_full"], len(calib_data))
                calib_eval_subset = calib_data.select(range(n_sens))

                deltas, base = score_layers_taqo_oracle(
                    base_model,
                    tokenizer,
                    calib_eval_subset,
                    dataset_type=dataset_name,
                    max_seq_len=max_seq_len,
                    device=device,
                    low_bit=4,
                    group_size=QUANT_CONFIG["qgroup_size"],
                    cache_dequant=QUANT_CONFIG["cache_dequant"],
                )
                bit_map, scores = taqo_allocate_bits_oracle(
                    deltas,
                    keep_first=QUANT_CONFIG["keep_first"],
                    keep_last=QUANT_CONFIG["keep_last"],
                    keep_topk=QUANT_CONFIG["taqo_keep_topk"],
                )
                scores.update(base)

            elif method == "TAQoS":
                print("[TAQoS] Scoring layers...", flush=True)
                taqos_samples = min(32, len(calib_prompts)) if run_type == "test" else min(64, len(calib_prompts))
                kl_scores = score_layers_taqos(
                    base_model, tokenizer, calib_prompts[:taqos_samples],
                    max_seq_len, device, QUANT_CONFIG["taqos_temperature"]
                )
                bit_map, scores = taqos_allocate_bits(kl_scores, QUANT_CONFIG["top_8bit_percent"])
            else:
                raise ValueError(f"Unknown method: {method}")

            # Save to cache with full provenance
            import datetime
            meta = {
                "model": model_name,
                "method": method,
                "dataset": dataset_name,
                "run_type": run_type,
                "config_hash": config_hash,
                "created_at": datetime.datetime.now().isoformat(),
                "run_config": RUN_CONFIGS[run_type],
                "quant_config": QUANT_CONFIG,
                "n_layers": n_layers,
            }
            save_cached_config(cache_path, bit_map, scores, meta)

    n_4 = bit_map.count(4)
    n_8 = bit_map.count(8)
    n_16 = bit_map.count(16)
    print(f"[{method}] Allocation: 4-bit={n_4}, 8-bit={n_8}, 16-bit={n_16}", flush=True)

    # Reload and quantize (or load from cache)
    del base_model
    torch.cuda.empty_cache()
    base_model, _ = load_model_and_tokenizer(model_name, device, hf_token)
    
    # Check if quantized model is cached
    if check_quantized_model_exists(cache_path):
        print(f"[CACHE] Found cached quantized model, loading...", flush=True)
        q_model = load_quantized_model(
            base_model, bit_map, cache_path,
            QUANT_CONFIG["qgroup_size"], QUANT_CONFIG["cache_dequant"]
        )
    else:
        print(f"[QUANT] Quantizing model...", flush=True)
        q_model = quantize_with_bit_map(base_model, bit_map, QUANT_CONFIG["qgroup_size"], QUANT_CONFIG["cache_dequant"])
        # Save the quantized model for future runs
        save_quantized_model(q_model, cache_path)
    
    q_model.eval()

    # Evaluate
    print(f"[EVAL] Evaluating on {dataset_name}...", flush=True)
    em, f1, n, detailed_results = evaluate_model(
        q_model, tokenizer, eval_data, dataset_name, 1024, max_seq_len, device
    )
    print(f"[RESULTS] EM={em*100:.2f}% F1={f1*100:.2f}% ({n} examples)", flush=True)

    # Save results
    summary = save_results(
        results_dir, model_name, method, dataset_name, run_type,
        em, f1, n, detailed_results, bit_map, scores
    )

    # Cleanup
    del q_model
    del base_model
    torch.cuda.empty_cache()

    return summary


def run_experiments(args):
    """Run all experiments based on args."""
    run_type = args.run_type
    device = args.device
    data_dir = args.data_dir
    results_dir = args.results_dir
    cache_dir = args.cache_dir
    hf_token = args.hf_token
    eval_only = getattr(args, 'eval_only', False)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Determine what to run
    if args.model and args.model != "all":
        # Strip any stray quotes from model name (common CLI mistake)
        models = [args.model.strip('"').strip("'")]
    elif run_type == "demo":
        models = [DEMO_MODEL]
    else:
        models = MODELS

    if args.dataset and args.dataset != "all":
        datasets = [args.dataset]
    elif run_type == "demo":
        datasets = [DEMO_DATASET]
    else:
        datasets = DATASETS

    if args.method and args.method != "all":
        methods = [args.method]
    else:
        methods = METHODS

    print("="*70, flush=True)
    print(f"TAQ-ICML Experiments", flush=True)
    print("="*70, flush=True)
    print(f"Run type: {run_type}", flush=True)
    print(f"Models: {models}", flush=True)
    print(f"Methods: {methods}", flush=True)
    print(f"Datasets: {datasets}", flush=True)
    print(f"Config: {RUN_CONFIGS[run_type]}", flush=True)
    print("="*70, flush=True)

    all_results = []

    for model_name in models:
        for dataset_name in datasets:
            for method in methods:
                try:
                    summary = run_single_experiment(
                        model_name, method, dataset_name, run_type,
                        device, data_dir, results_dir, cache_dir, hf_token, eval_only
                    )
                    all_results.append(summary)
                except Exception as e:
                    print(f"[ERROR] {model_name}/{method}/{dataset_name}: {e}", flush=True)
                    traceback.print_exc()

    # Print final summary
    print(f"\n{'='*80}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print("="*80, flush=True)
    print(f"{'Model':<40} {'Method':<8} {'Dataset':<12} {'EM':>8} {'F1':>8}", flush=True)
    print("-"*80, flush=True)
    for r in all_results:
        print(f"{r['model']:<40} {r['method']:<8} {r['dataset']:<12} {r['em_percent']:>7.2f}% {r['f1_percent']:>7.2f}%", flush=True)

    # Save overall summary
    csv_path = os.path.join(results_dir, f"summary_{run_type}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "method", "dataset", "em", "f1", "n_examples"])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in ["model", "method", "dataset", "em", "f1", "n_examples"]})
    print(f"\nSummary saved to {csv_path}", flush=True)

    return all_results


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="TAQ-ICML: Task-Aware Quantization Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--run_type", type=str, default="demo", choices=["demo", "test", "full"],
                        help="Run type: demo (1 model), test (fast), full (all)")
    parser.add_argument("--method", type=str, default="all", choices=["TAQ", "TAQO", "TAQoS", "all"],
                        help="Method to run")
    parser.add_argument("--dataset", type=str, default="all", choices=["trivia_qa", "mmlu_pro", "code_mmlu", "all"],
                        help="Dataset to run")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model (HF name) or 'all'")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto/cuda/cpu")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Results directory")
    parser.add_argument("--cache_dir", type=str, default="saved_models",
                        help="Cache directory for quantized models")
    parser.add_argument("--data_dir", type=str, default="datasets_local",
                        help="Data directory")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip scoring, load cached bit_map and only run evaluation")

    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    run_experiments(args)


if __name__ == "__main__":
    main()
