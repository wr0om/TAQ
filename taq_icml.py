#!/usr/bin/env python
"""
TAQ-ICML: Task-Aware Quantization Methods

Three quantization methods:
- TAQ: Information (Entropy) + Normalized Variance → Top 25% 8-bit, rest 4-bit
- TAQO: Activation Norm scoring → Top K FP16, rest 4-bit  
- TAQoS: Output-Sensitive KL Divergence → Top 25% 8-bit, rest 4-bit

Running modes:
- full: All 5 models × 3 datasets × 3 methods (calib=512, eval=2048)
- test: Same as full but faster (calib=32, eval=64)
- demo: 1 model × 3 datasets × 3 methods (calib=512, eval=2048)

Usage:
    python taq_icml.py --mode full
    python taq_icml.py --mode test
    python taq_icml.py --mode demo --model "Qwen/Qwen2.5-7B-Instruct"
    
    # Specific combinations
    python taq_icml.py --model "meta-llama/Llama-3.1-8B-Instruct" --datasets trivia_qa --methods TAQ TAQO
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
from typing import List, Dict, Any, Optional, Tuple
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

hf_logging.set_verbosity_info()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

random.seed(42)
np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
]

DEFAULT_DATASETS = ["trivia_qa", "mmlu_pro", "code_mmlu"]
DEFAULT_METHODS = ["TAQ", "TAQO", "TAQoS"]

# Run mode configurations
RUN_CONFIGS = {
    "full": {"calib_size": 512, "eval_size": 2048, "score_samples": 256},
    "test": {"calib_size": 32, "eval_size": 64, "score_samples": 32},
    "demo": {"calib_size": 512, "eval_size": 2048, "score_samples": 256},
}

DEMO_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# =============================================================================
# METRICS (TriviaQA / SQuAD style)
# =============================================================================
def normalize_answer(s: str) -> str:
    """Official TriviaQA/SQuAD normalization."""
    s = s.replace("_", " ").lower()
    s = re.sub(r"\b(a|an|the|and|but|or|if|in|on|at|to|for|by|of)\b", " ", s)
    s = re.sub(r"[^a-z0-9]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths) -> float:
    scores = [metric_fn(prediction, gt) for gt in ground_truths]
    return max(scores) if scores else 0.0


# =============================================================================
# DATA LOADING (Preserved exact splits)
# =============================================================================
def stratified_disjoint_split(ds, label_col: str, calib_size: int, eval_size: int, seed: int = 42):
    """Stratified split maintaining label proportions."""
    n = len(ds)
    assert calib_size + eval_size <= n, "calib_size + eval_size must be <= dataset size"
    
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
            raise ValueError(f"Not enough examples in category '{lab}'")
        calib_idx.extend(idxs[:c_t].tolist())
        eval_idx.extend(idxs[c_t:c_t + e_t].tolist())
    
    rng.shuffle(calib_idx)
    rng.shuffle(eval_idx)
    return ds.select(calib_idx), ds.select(eval_idx)


def load_trivia_qa(calib_size: int, eval_size: int, data_dir: str):
    """Load TriviaQA with exact splits."""
    subset, split = "rc.nocontext", "validation"
    calib_path = os.path.join(data_dir, f"trivia_qa/{subset}/calib")
    eval_path = os.path.join(data_dir, f"trivia_qa/{subset}/eval")
    
    if os.path.exists(calib_path) and os.path.exists(eval_path):
        print(f"[TriviaQA] Loading from disk", flush=True)
        calib_data = load_from_disk(calib_path)
        eval_data = load_from_disk(eval_path)
        if len(calib_data) > calib_size:
            calib_data = calib_data.select(range(calib_size))
        if len(eval_data) > eval_size:
            eval_data = eval_data.select(range(eval_size))
    else:
        print(f"[TriviaQA] Downloading...", flush=True)
        hf_dataset = load_dataset("mandarjoshi/trivia_qa", subset)
        data = hf_dataset[split]
        # Exact split: first calib_size for calib, next eval_size for eval
        calib_data = data.select(range(calib_size))
        eval_data = data.select(range(calib_size, calib_size + eval_size))
        os.makedirs(os.path.dirname(calib_path), exist_ok=True)
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)
        calib_data.save_to_disk(calib_path)
        eval_data.save_to_disk(eval_path)
    
    print(f"[TriviaQA] Calib: {len(calib_data)}, Eval: {len(eval_data)}", flush=True)
    return calib_data, eval_data


def load_mmlu_pro(calib_size: int, eval_size: int, data_dir: str):
    """Load MMLU-Pro with stratified split by category."""
    calib_path = os.path.join(data_dir, "mmlu_pro/calib")
    eval_path = os.path.join(data_dir, "mmlu_pro/eval")
    
    if os.path.exists(calib_path) and os.path.exists(eval_path):
        print(f"[MMLU-Pro] Loading from disk", flush=True)
        calib_data = load_from_disk(calib_path)
        eval_data = load_from_disk(eval_path)
        if len(calib_data) > calib_size:
            calib_data = calib_data.select(range(calib_size))
        if len(eval_data) > eval_size:
            eval_data = eval_data.select(range(eval_size))
    else:
        print(f"[MMLU-Pro] Downloading...", flush=True)
        hf_dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        data = hf_dataset["test"]
        calib_data, eval_data = stratified_disjoint_split(
            data, label_col="category", calib_size=calib_size, eval_size=eval_size, seed=42
        )
        os.makedirs(os.path.dirname(calib_path), exist_ok=True)
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)
        calib_data.save_to_disk(calib_path)
        eval_data.save_to_disk(eval_path)
    
    print(f"[MMLU-Pro] Calib: {len(calib_data)}, Eval: {len(eval_data)}", flush=True)
    return calib_data, eval_data


def load_code_mmlu(calib_size: int, eval_size: int, data_dir: str):
    """Load CodeMMLU with stratified split by subset."""
    calib_path = os.path.join(data_dir, "code_mmlu/calib")
    eval_path = os.path.join(data_dir, "code_mmlu/eval")
    
    if os.path.exists(calib_path) and os.path.exists(eval_path):
        print(f"[CodeMMLU] Loading from disk", flush=True)
        calib_data = load_from_disk(calib_path)
        eval_data = load_from_disk(eval_path)
        if len(calib_data) > calib_size:
            calib_data = calib_data.select(range(calib_size))
        if len(eval_data) > eval_size:
            eval_data = eval_data.select(range(eval_size))
    else:
        print(f"[CodeMMLU] Downloading...", flush=True)
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
        os.makedirs(os.path.dirname(calib_path), exist_ok=True)
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)
        calib_data.save_to_disk(calib_path)
        eval_data.save_to_disk(eval_path)
    
    print(f"[CodeMMLU] Calib: {len(calib_data)}, Eval: {len(eval_data)}", flush=True)
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

    print(f"[Model] Loading: {model_name}", flush=True)
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
# QUANTIZATION MODULE
# =============================================================================
class QuantLinearWB(nn.Module):
    """Weight-only per-group affine quantization for wbit in {4, 8}."""
    def __init__(self, linear: nn.Linear, wbit: int = 4, group_size: int = 128, cache_dequant: bool = True):
        super().__init__()
        assert wbit in (4, 8)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None

        W = linear.weight.detach().to(torch.float32).cpu()
        self.wbit, self.group_size, self.cache_dequant = int(wbit), int(group_size), bool(cache_dequant)
        self._W_cache = None

        G = max(1, self.group_size)
        in_groups = (W.shape[1] + G - 1) // G
        scales = torch.empty((W.shape[0], in_groups), dtype=torch.float32)
        zeros = torch.empty((W.shape[0], in_groups), dtype=torch.float32)
        Q = 15 if self.wbit == 4 else 255
        codes = torch.empty((W.shape[0], in_groups, (G + 1) // 2 if self.wbit == 4 else G), dtype=torch.uint8)

        for o in range(W.shape[0]):
            row = W[o]
            for g in range(in_groups):
                s, e = g * G, min((g + 1) * G, W.shape[1])
                seg = row[s:e]
                if seg.numel() == 0:
                    scales[o, g], zeros[o, g] = 1.0, 0.0
                    continue
                wmin, wmax = float(seg.min()), float(seg.max())
                if wmax <= wmin + 1e-8:
                    scale, zp, qvals = 1.0, 0.0, torch.zeros_like(seg, dtype=torch.uint8)
                else:
                    scale = (wmax - wmin) / Q
                    zp = -wmin / (scale + 1e-12)
                    qvals = torch.clamp(torch.round(seg / (scale + 1e-12) + zp), 0, Q).to(torch.uint8)

                scales[o, g], zeros[o, g] = float(scale), float(zp)

                if self.wbit == 4:
                    if qvals.numel() % 2 == 1:
                        qvals = torch.cat([qvals, torch.zeros(1, dtype=torch.uint8)])
                    packed = (qvals[0::2] << 4) | qvals[1::2]
                    buf = torch.zeros((G + 1) // 2, dtype=torch.uint8)
                    buf[:packed.numel()] = packed
                    codes[o, g, :] = buf
                else:
                    buf = torch.zeros(G, dtype=torch.uint8)
                    buf[:qvals.numel()] = qvals
                    codes[o, g, :] = buf

        self.register_buffer("qcodes", codes.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("zeros", zeros.contiguous())

    def _dequant_weight_to(self, device):
        out, in_groups, _ = self.qcodes.shape
        G = self.group_size
        Wfull = torch.zeros((out, in_groups * G), dtype=torch.float16, device=device)
        for o in range(out):
            for g in range(in_groups):
                if self.wbit == 4:
                    packed = self.qcodes[o, g, :].to(device)
                    q = torch.empty(G, dtype=torch.float32, device=device)
                    q[0::2], q[1::2] = ((packed >> 4) & 0xF).float(), (packed & 0xF).float()
                else:
                    q = self.qcodes[o, g, :].to(device=device, dtype=torch.float32)
                scale = self.scales[o, g].to(device=device, dtype=torch.float32)
                zp = self.zeros[o, g].to(device=device, dtype=torch.float32)
                Wfull[o, g * G:(g + 1) * G] = ((q - zp) * scale).to(torch.float16)
        return Wfull[:, :self.in_features]

    def forward(self, x):
        if self.cache_dequant and self._W_cache is not None and self._W_cache.device == x.device:
            return F.linear(x, self._W_cache, self.bias)
        W = self._dequant_weight_to(x.device)
        if self.cache_dequant and not self.training:
            self._W_cache = W
        return F.linear(x, W, self.bias)


def quantize_with_bit_map(model, bit_map: List[int], group_size: int = 128, cache_dequant: bool = True):
    """Quantize model layers according to bit_map."""
    layers = get_transformer_layers(model)
    if layers is None:
        print("[QUANT] Warning: Could not find layers", flush=True)
        return model
    nL = len(layers)
    replaced = 0
    for li in range(nL):
        bits = bit_map[li]
        if bits == 16:
            print(f"[QUANT] Layer {li} -> FP16", flush=True)
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
        print(f"[QUANT] Layer {li} -> int{bits}", flush=True)
    print(f"[QUANT] Replaced {replaced} Linear modules", flush=True)
    return model


# =============================================================================
# MODEL CACHING
# =============================================================================
def get_config_hash(model_name: str, method: str, dataset: str, mode: str) -> str:
    """Generate unique hash for model config."""
    config_str = f"{model_name}_{method}_{dataset}_{mode}"
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def get_cached_model_path(models_dir: str, model_name: str, method: str, dataset: str, mode: str) -> str:
    """Get path for cached quantized model."""
    model_safe = model_name.replace("/", "_")
    config_hash = get_config_hash(model_name, method, dataset, mode)
    return os.path.join(models_dir, f"{model_safe}_{method}_{dataset}_{mode}_{config_hash}")


def save_quantized_model(model, tokenizer, bit_map: List[int], save_path: str, config: Dict):
    """Save quantized model and config."""
    os.makedirs(save_path, exist_ok=True)
    # Save bit map and config
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump({"bit_map": bit_map, **config}, f, indent=2)
    # Save model state dict (only quantized layers)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    tokenizer.save_pretrained(save_path)
    print(f"[CACHE] Saved model to {save_path}", flush=True)


def load_cached_model(model, save_path: str) -> Tuple[Any, List[int]]:
    """Load cached quantized model."""
    with open(os.path.join(save_path, "config.json"), "r") as f:
        config = json.load(f)
    bit_map = config.get("bit_map", [])
    model.load_state_dict(torch.load(os.path.join(save_path, "model.pt")))
    print(f"[CACHE] Loaded model from {save_path}", flush=True)
    return model, bit_map


def check_cached_model_exists(models_dir: str, model_name: str, method: str, dataset: str, mode: str) -> bool:
    """Check if cached model exists."""
    path = get_cached_model_path(models_dir, model_name, method, dataset, mode)
    return os.path.exists(os.path.join(path, "config.json"))


# =============================================================================
# TAQ SCORING: Entropy (all tokens) + Normalized Variance
# =============================================================================
@torch.no_grad()
def score_layers_taq(model, tokenizer, texts, max_tokens, device, reservoir_size=256, batch_size=8):
    """
    TAQ: Information (Entropy from ALL tokens) + Normalized Variance scoring.
    
    Entropy: Build G = X @ X.T from all valid tokens, compute eigenvalues, normalize to p, H = -sum(p log p)
    Stability: S = -Var(h) / (E|h| + ε)² (normalized variance, scale-invariant)
    """
    device = _resolve_device(device)
    nL = get_num_layers(model)
    
    # Storage for all-token entropy computation
    token_reservoirs = [[] for _ in range(nL)]  # List of [n_tokens, dim] tensors
    
    # Storage for normalized variance
    sum_h = [0.0] * nL
    sum_h2 = [0.0] * nL
    sum_abs_h = [0.0] * nL
    n_elems = [0] * nL

    B = max(1, int(batch_size))
    total = len(texts)
    t0 = time.time()
    
    for start in range(0, total, B):
        batch = texts[start:start + B]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        attn_mask = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))

        pct = 100 * min(total, start + B) / max(1, total)
        if (start + B) % (B * 4) == 0 or start + B >= total:
            print(f"[TAQ SCORING] {pct:.1f}%  elapsed={time.time() - t0:.1f}s", flush=True)

        for li in range(1, len(hs)):
            X = hs[li]  # [B, T, D]
            Bx, Tx, Dx = X.shape
            
            # Mask to valid tokens
            mask = attn_mask.unsqueeze(-1).expand_as(X)  # [B, T, D]
            X_masked = X * mask.float()
            
            # Flatten to [B*T, D] and filter valid
            X_flat = X_masked.reshape(Bx * Tx, Dx)
            valid_mask = attn_mask.reshape(Bx * Tx).bool()
            X_valid = X_flat[valid_mask].float()  # [n_valid, D]
            
            # Stats for normalized variance
            h_flat = X_valid
            sum_h[li-1] += float(h_flat.sum().item())
            sum_h2[li-1] += float((h_flat ** 2).sum().item())
            sum_abs_h[li-1] += float(h_flat.abs().sum().item())
            n_elems[li-1] += h_flat.numel()
            
            # Reservoir sampling for entropy (keep subset of token vectors)
            current_reservoir_size = sum(t.shape[0] for t in token_reservoirs[li-1]) if token_reservoirs[li-1] else 0
            if current_reservoir_size < reservoir_size:
                # Random sample from valid tokens
                n_take = min(reservoir_size - current_reservoir_size, X_valid.shape[0], 32)
                if n_take > 0:
                    idx = torch.randperm(X_valid.shape[0])[:n_take]
                    token_reservoirs[li-1].append(X_valid[idx].detach().cpu().to(torch.float32))
    
    info_scores = []
    stability_scores = []
    
    for li in range(nL):
        # Compute entropy from Gram matrix of token vectors
        if token_reservoirs[li]:
            R = torch.cat(token_reservoirs[li], dim=0)  # [n_tokens, D]
            if R.shape[0] > reservoir_size:
                idx = torch.randperm(R.shape[0])[:reservoir_size]
                R = R[idx]
            
            # Center
            R = R - R.mean(dim=0, keepdim=True)
            
            if R.shape[0] > 1:
                # Build Gram matrix G = R @ R.T
                G = (R @ R.T) / max(1, R.shape[0])
                # Eigenvalues
                evals = torch.linalg.eigvalsh(G).clamp(min=0)
                # Normalize to probability distribution
                if evals.sum() > 0:
                    p = evals / evals.sum()
                    # Entropy H = -sum(p * log(p))
                    H = float((-(p * (p + 1e-12).log())).sum().item())
                else:
                    H = 0.0
            else:
                H = 0.0
        else:
            H = 0.0
        info_scores.append(H)
        
        # Normalized variance: S = -Var(h) / (E|h| + ε)²
        if n_elems[li] > 0:
            mean = sum_h[li] / n_elems[li]
            var = max(0, (sum_h2[li] / n_elems[li]) - mean * mean)
            mean_abs = sum_abs_h[li] / n_elems[li]
            eps = 1e-8
            # Normalized (scale-invariant) stability
            normalized_var = var / ((mean_abs + eps) ** 2)
            stability = -normalized_var  # Higher is better (lower variance)
        else:
            stability = 0.0
        stability_scores.append(stability)
    
    return info_scores, stability_scores


def _zscore(xs):
    if not xs:
        return xs
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
    s = math.sqrt(v + 1e-12)
    return [(x - m) / s for x in xs]


def taq_allocate_bits(info_scores, stability_scores, top_8bit_percent=25, info_weight=0.5, var_weight=0.5):
    """
    TAQ bit allocation: Top 25% layers → 8-bit, rest → 4-bit.
    Combined score = info_weight * z(info) + var_weight * z(stability)
    """
    nL = len(info_scores)
    Iz = _zscore(info_scores)
    Sz = _zscore(stability_scores)
    combined = [info_weight * Iz[i] + var_weight * Sz[i] for i in range(nL)]
    
    n_8bit = max(1, int(round(nL * top_8bit_percent / 100)))
    sorted_indices = sorted(range(nL), key=lambda i: combined[i], reverse=True)
    
    bit_map = [4] * nL
    for i in sorted_indices[:n_8bit]:
        bit_map[i] = 8
    
    return bit_map, combined


# =============================================================================
# TAQO SCORING: Activation Norm
# =============================================================================
@torch.no_grad()
def score_layers_taqo(model, tokenizer, texts, max_tokens, device):
    """TAQO: Score layers by activation norm."""
    device = _resolve_device(device)
    model.eval()
    scores = []
    count = 0
    t0 = time.time()
    
    for i, t in enumerate(texts):
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
        outputs = model(**enc, output_hidden_states=True, use_cache=False)
        hs = outputs.hidden_states
        
        if not scores:
            scores = [0.0] * (len(hs) - 1)
        
        last_idx = enc["input_ids"].shape[1] - 1
        for li in range(1, len(hs)):
            scores[li - 1] += torch.linalg.vector_norm(hs[li][0, last_idx].float()).item()
        count += 1
        
        if (i + 1) % 50 == 0:
            print(f"[TAQO SCORING] {100*(i+1)/len(texts):.1f}%  elapsed={time.time() - t0:.1f}s", flush=True)
    
    if count > 0 and scores:
        scores = [s / count for s in scores]
    return scores


def taqo_allocate_bits(scores, topk=8, keep_first=2, keep_last=2):
    """TAQO bit allocation: Top K layers → FP16, rest → 4-bit."""
    n = len(scores)
    keep = set(range(min(keep_first, n))) | set(range(max(0, n - keep_last), n))
    remaining = sorted([i for i in range(n) if i not in keep], key=lambda i: scores[i], reverse=True)
    for i in remaining[:max(0, topk - len(keep))]:
        keep.add(i)
    return [16 if i in keep else 4 for i in range(n)]


# =============================================================================
# TAQoS SCORING: Output-Sensitive KL Divergence
# =============================================================================
@torch.no_grad()
def score_layers_taqos(model, tokenizer, texts, max_tokens, device, temperature=1.0):
    """
    TAQoS: Score layers by KL divergence between original and noise-injected outputs.
    
    For each layer ℓ:
    1. Run forward pass with noise injected at layer ℓ: h_ℓ' = h_ℓ + η, η ~ U(-Δ/2, Δ/2)
    2. Compute KL(softmax(z/T) || softmax(z'/T)) where z is original logits, z' is noised
    3. Higher KL → more sensitive layer → needs more bits
    """
    device = _resolve_device(device)
    model.eval()
    nL = get_num_layers(model)
    
    # First pass: get baseline logits and estimate noise scale per layer
    layer_ranges = [[] for _ in range(nL)]
    baseline_logits_all = []
    
    print("[TAQoS] Phase 1: Computing baseline logits and layer ranges...", flush=True)
    t0 = time.time()
    
    for i, text in enumerate(texts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
        
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
            baseline_logits = out.logits[:, -1, :].float()  # [1, vocab]
            baseline_logits_all.append(baseline_logits.cpu())
            
            hs = out.hidden_states
            for li in range(1, len(hs)):
                h = hs[li][:, -1, :].float()  # Last token hidden state
                h_range = (h.max() - h.min()).item()
                layer_ranges[li-1].append(h_range)
        
        if (i + 1) % 50 == 0:
            print(f"  Phase 1: {100*(i+1)/len(texts):.1f}%", flush=True)
    
    # Compute noise scale (step size) per layer: Δ = range / (2^b - 1), assuming 4-bit
    delta = []
    for li in range(nL):
        avg_range = sum(layer_ranges[li]) / max(1, len(layer_ranges[li]))
        delta.append(avg_range / 15)  # 4-bit: 2^4 - 1 = 15
    
    print(f"[TAQoS] Phase 2: Computing KL divergence per layer... elapsed={time.time()-t0:.1f}s", flush=True)
    
    # Second pass: inject noise at each layer and compute KL divergence
    kl_scores = [0.0] * nL
    
    # Hook for noise injection
    def make_noise_hook(layer_idx, noise_scale):
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
        if (li + 1) % 5 == 0 or li == 0:
            print(f"  Scoring layer {li+1}/{nL}...", flush=True)
        
        kl_sum = 0.0
        
        for i, text in enumerate(texts):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
            baseline_logits = baseline_logits_all[i].to(device)
            
            # Register hook to inject noise at layer li
            handle = layers[li].register_forward_hook(make_noise_hook(li, delta[li]))
            
            try:
                with torch.no_grad():
                    out = model(**enc, use_cache=False)
                    noised_logits = out.logits[:, -1, :].float()
                    
                    # KL divergence with temperature
                    p = F.softmax(baseline_logits / temperature, dim=-1)
                    q = F.softmax(noised_logits / temperature, dim=-1)
                    kl = F.kl_div(q.log(), p, reduction='batchmean').item()
                    kl_sum += kl
            finally:
                handle.remove()
        
        kl_scores[li] = kl_sum / max(1, len(texts))
    
    print(f"[TAQoS] Done. elapsed={time.time()-t0:.1f}s", flush=True)
    return kl_scores


def taqos_allocate_bits(kl_scores, top_8bit_percent=25):
    """TAQoS bit allocation: Higher KL → more sensitive → 8-bit."""
    nL = len(kl_scores)
    n_8bit = max(1, int(round(nL * top_8bit_percent / 100)))
    sorted_indices = sorted(range(nL), key=lambda i: kl_scores[i], reverse=True)
    
    bit_map = [4] * nL
    for i in sorted_indices[:n_8bit]:
        bit_map[i] = 8
    
    return bit_map


# =============================================================================
# CALIBRATION FORMATTING
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
# EVALUATION
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

        if (i + 1) % 100 == 0:
            print(f"[EVAL] {i+1}/{len(eval_data)}: EM {em_sum/total:.4f} | F1 {f1_sum/total:.4f} | elapsed={time.time()-t0:.1f}s", flush=True)

    return (em_sum / total if total else 0.0), (f1_sum / total if total else 0.0), total, results


def save_results(results_dir, model_name, method, dataset_name, mode, em, f1, n, detailed_results, bit_map, scores=None):
    os.makedirs(results_dir, exist_ok=True)
    model_safe = model_name.replace("/", "_")

    with open(os.path.join(results_dir, f"{model_safe}_{method}_{dataset_name}_{mode}_results.jsonl"), "w") as f:
        for r in detailed_results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "model": model_name,
        "method": method,
        "dataset": dataset_name,
        "mode": mode,
        "em": em,
        "f1": f1,
        "n_examples": n,
        "em_percent": em * 100,
        "f1_percent": f1 * 100,
        "bit_map": bit_map,
        "bits_summary": {
            "4-bit": bit_map.count(4),
            "8-bit": bit_map.count(8),
            "16-bit": bit_map.count(16),
        }
    }
    if scores:
        summary["scores"] = scores

    with open(os.path.join(results_dir, f"{model_safe}_{method}_{dataset_name}_{mode}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[RESULTS] Saved to {results_dir}", flush=True)
    return summary


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================
def run_single_experiment(model_name, method, dataset_name, mode, config, device, data_dir, results_dir, models_dir):
    """Run a single model-method-dataset combination."""
    run_config = RUN_CONFIGS[mode]
    calib_size = run_config["calib_size"]
    eval_size = run_config["eval_size"]
    score_samples = run_config["score_samples"]
    
    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"Method: {method}", flush=True)
    print(f"Dataset: {dataset_name}", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Check for existing results
    model_safe = model_name.replace("/", "_")
    results_path = os.path.join(results_dir, f"{model_safe}_{method}_{dataset_name}_{mode}_summary.json")
    if os.path.exists(results_path):
        print(f"[SKIP] Results exist: {results_path}", flush=True)
        with open(results_path) as f:
            return json.load(f)
    
    # Load model
    base_model, tokenizer = load_model_and_tokenizer(model_name, device, config.get("hf_token"))
    n_layers = get_num_layers(base_model)
    print(f"[Model] {n_layers} layers", flush=True)
    
    # Load data
    calib_data, eval_data = DATASET_LOADERS[dataset_name](calib_size, eval_size, data_dir)
    calib_texts = CALIB_FORMATTERS[dataset_name](calib_data)[:score_samples]
    
    # Check for cached quantized model
    cached_path = get_cached_model_path(models_dir, model_name, method, dataset_name, mode)
    
    if os.path.exists(os.path.join(cached_path, "config.json")):
        print(f"[CACHE] Loading cached model...", flush=True)
        with open(os.path.join(cached_path, "config.json"), "r") as f:
            cached_config = json.load(f)
        bit_map = cached_config["bit_map"]
        scores = cached_config.get("scores", {})
        
        # Reload and quantize
        del base_model
        torch.cuda.empty_cache()
        base_model, _ = load_model_and_tokenizer(model_name, device, config.get("hf_token"))
        q_model = quantize_with_bit_map(base_model, bit_map, config.get("qgroup_size", 128), True)
    else:
        # Score layers based on method
        if method == "TAQ":
            print("[TAQ] Scoring layers...", flush=True)
            info_scores, stability_scores = score_layers_taq(
                base_model, tokenizer, calib_texts,
                config.get("max_seq_len", 512), device,
                config.get("reservoir_size", 256), config.get("score_batch_size", 8)
            )
            bit_map, combined = taq_allocate_bits(info_scores, stability_scores, config.get("top_8bit_percent", 25))
            scores = {"info": info_scores, "stability": stability_scores, "combined": combined}
            
        elif method == "TAQO":
            print("[TAQO] Scoring layers...", flush=True)
            norm_scores = score_layers_taqo(
                base_model, tokenizer, calib_texts,
                config.get("max_seq_len", 512), device
            )
            bit_map = taqo_allocate_bits(norm_scores, config.get("keep_topk", 8), config.get("keep_first", 2), config.get("keep_last", 2))
            scores = {"norm": norm_scores}
            
        elif method == "TAQoS":
            print("[TAQoS] Scoring layers...", flush=True)
            kl_scores = score_layers_taqos(
                base_model, tokenizer, calib_texts[:min(64, len(calib_texts))],  # TAQoS is expensive, use fewer samples
                config.get("max_seq_len", 512), device, config.get("taqos_temperature", 1.0)
            )
            bit_map = taqos_allocate_bits(kl_scores, config.get("top_8bit_percent", 25))
            scores = {"kl": kl_scores}
        else:
            raise ValueError(f"Unknown method: {method}")
        
        n_4 = bit_map.count(4)
        n_8 = bit_map.count(8)
        n_16 = bit_map.count(16)
        print(f"[{method}] Bit allocation: 4-bit={n_4}, 8-bit={n_8}, 16-bit={n_16}", flush=True)
        
        # Reload and quantize
        del base_model
        torch.cuda.empty_cache()
        base_model, _ = load_model_and_tokenizer(model_name, device, config.get("hf_token"))
        q_model = quantize_with_bit_map(base_model, bit_map, config.get("qgroup_size", 128), True)
        
        # Save cached model
        save_config = {"bit_map": bit_map, "scores": scores, "method": method, "dataset": dataset_name, "mode": mode}
        os.makedirs(cached_path, exist_ok=True)
        with open(os.path.join(cached_path, "config.json"), "w") as f:
            json.dump(save_config, f, indent=2)
        print(f"[CACHE] Saved to {cached_path}", flush=True)
    
    q_model.eval()
    
    # Evaluate
    print(f"[EVAL] Evaluating on {dataset_name}...", flush=True)
    em, f1, n, detailed_results = evaluate_model(
        q_model, tokenizer, eval_data, dataset_name,
        config.get("gen_len", 64), config.get("max_seq_len", 512), device
    )
    print(f"[RESULTS] EM: {em*100:.2f}% | F1: {f1*100:.2f}% ({n} examples)", flush=True)
    
    # Save results
    summary = save_results(
        results_dir, model_name, method, dataset_name, mode,
        em, f1, n, detailed_results, bit_map, scores
    )
    
    # Cleanup
    del q_model
    del base_model
    torch.cuda.empty_cache()
    
    return summary


def run_experiments(config):
    """Run all experiments based on config."""
    mode = config["mode"]
    device = _resolve_device(config.get("device", "auto"))
    data_dir = config.get("data_dir", "datasets_local")
    results_dir = config.get("results_dir", "results")
    models_dir = config.get("models_dir", "quantized_models")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Determine what to run
    if mode == "demo":
        models = [config.get("model", DEMO_MODEL)]
    else:
        models = config.get("models", DEFAULT_MODELS)
        if config.get("model"):
            models = [config["model"]]
    
    datasets = config.get("datasets", DEFAULT_DATASETS)
    methods = config.get("methods", DEFAULT_METHODS)
    
    print("=" * 70, flush=True)
    print(f"TAQ-ICML Experiments", flush=True)
    print("=" * 70, flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Models: {models}", flush=True)
    print(f"Methods: {methods}", flush=True)
    print(f"Datasets: {datasets}", flush=True)
    print(f"Config: {RUN_CONFIGS[mode]}", flush=True)
    print("=" * 70, flush=True)
    
    all_results = []
    
    for model_name in models:
        for dataset_name in datasets:
            for method in methods:
                try:
                    summary = run_single_experiment(
                        model_name, method, dataset_name, mode,
                        config, device, data_dir, results_dir, models_dir
                    )
                    all_results.append(summary)
                except Exception as e:
                    print(f"[ERROR] {model_name}/{method}/{dataset_name}: {e}", flush=True)
                    traceback.print_exc()
    
    # Print final summary
    print(f"\n{'='*80}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Model':<45} {'Method':<8} {'Dataset':<12} {'EM':>8} {'F1':>8}", flush=True)
    print("-" * 85, flush=True)
    for r in all_results:
        print(f"{r['model']:<45} {r['method']:<8} {r['dataset']:<12} {r['em_percent']:>7.2f}% {r['f1_percent']:>7.2f}%", flush=True)
    
    # Save overall summary
    csv_path = os.path.join(results_dir, f"summary_{mode}.csv")
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
    
    parser.add_argument("--mode", type=str, default="full", choices=["full", "test", "demo"],
                        help="Run mode: full (all, slow), test (all, fast), demo (1 model)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Specific model to run (overrides mode defaults)")
    parser.add_argument("--datasets", "-d", type=str, nargs="+", default=None,
                        choices=["trivia_qa", "mmlu_pro", "code_mmlu"],
                        help="Datasets to run (default: all)")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        choices=["TAQ", "TAQO", "TAQoS"],
                        help="Methods to run (default: all)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (default: auto)")
    parser.add_argument("--data_dir", type=str, default="datasets_local",
                        help="Data directory")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Results directory")
    parser.add_argument("--models_dir", type=str, default="quantized_models",
                        help="Cached models directory")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token")
    
    # TAQ params
    parser.add_argument("--top_8bit_percent", type=float, default=25,
                        help="Percentage of top layers for 8-bit (TAQ/TAQoS)")
    parser.add_argument("--reservoir_size", type=int, default=256,
                        help="Reservoir size for TAQ entropy")
    parser.add_argument("--score_batch_size", type=int, default=8,
                        help="Batch size for scoring")
    
    # TAQO params
    parser.add_argument("--keep_topk", type=int, default=8,
                        help="Top K layers for FP16 (TAQO)")
    parser.add_argument("--keep_first", type=int, default=2,
                        help="First N layers to keep FP16")
    parser.add_argument("--keep_last", type=int, default=2,
                        help="Last N layers to keep FP16")
    
    # Other
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Max sequence length")
    parser.add_argument("--gen_len", type=int, default=64,
                        help="Generation length")
    parser.add_argument("--qgroup_size", type=int, default=128,
                        help="Quantization group size")
    
    args = parser.parse_args()
    
    config = {
        "mode": args.mode,
        "model": args.model,
        "models": DEFAULT_MODELS,
        "datasets": args.datasets or DEFAULT_DATASETS,
        "methods": args.methods or DEFAULT_METHODS,
        "device": args.device,
        "data_dir": args.data_dir,
        "results_dir": args.results_dir,
        "models_dir": args.models_dir,
        "hf_token": args.hf_token,
        "top_8bit_percent": args.top_8bit_percent,
        "reservoir_size": args.reservoir_size,
        "score_batch_size": args.score_batch_size,
        "keep_topk": args.keep_topk,
        "keep_first": args.keep_first,
        "keep_last": args.keep_last,
        "max_seq_len": args.max_seq_len,
        "gen_len": args.gen_len,
        "qgroup_size": args.qgroup_size,
    }
    
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    
    run_experiments(config)


if __name__ == "__main__":
    main()

