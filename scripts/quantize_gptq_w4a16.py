#!/usr/bin/env python3
"""
Quantize MiniCPM-SALA to W4A16 GPTQ on GPU and save a GPTQ-formatted checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/root/autodl-tmp/models")
    parser.add_argument("--calibration-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--sym", action="store_true", default=True)
    parser.add_argument("--desc-act", action="store_true", default=False)
    parser.add_argument("--max-shard-size", default="4GB")
    return parser.parse_args()


def torch_dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def maybe_disable_incompatible_defuser() -> bool:
    try:
        import transformers.core_model_loading  # noqa: F401
        return False
    except Exception:
        try:
            import defuser
        except Exception:
            print("[compat] transformers.core_model_loading missing, but defuser is not installed; skipping defuser compatibility patch")
            return False

        def _noop_replace_fused_blocks(*args, **kwargs):
            print("[compat] defuser.replace_fused_blocks disabled for this transformers build")
            return False

        def _noop_convert_model(model, *args, **kwargs):
            print("[compat] defuser.convert_model disabled for this transformers build")
            return model

        defuser.replace_fused_blocks = _noop_replace_fused_blocks
        defuser.convert_model = _noop_convert_model
        return True


def normalize_minicpm_sala_config(config):
    # Sparse MiniCPM-SALA attention explicitly requires flash_attention_2.
    config._attn_implementation = "flash_attention_2"
    config._attn_implementation_internal = "flash_attention_2"
    # Some newer transformers builds normalize "no scaling" into a synthetic
    # default rope type that MiniCPM-SALA's custom modeling code does not accept.
    if getattr(config, "rope_scaling", None):
        rope_scaling = dict(config.rope_scaling)
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if rope_type == "default":
            print("[compat] resetting rope_scaling from synthetic default to None")
            config.rope_scaling = None
            # Persist the reset through config cloning / to_dict() round-trips.
            config.__dict__["rope_scaling"] = None
    if getattr(config, "rope_scaling_type", None) == "default":
        print("[compat] resetting rope_scaling_type from default to None")
        config.rope_scaling_type = None
    return config


def maybe_enable_minicpm_sala_fa2_compat() -> bool:
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception as exc:
        print(f"[compat] unable to import PreTrainedModel for FA2 compatibility: {exc}")
        return False

    original = getattr(PreTrainedModel, "_flash_attn_can_dispatch", None)
    if original is None:
        print("[compat] _flash_attn_can_dispatch missing; skipping FA2 compatibility shim")
        return False

    def patched(self, *args, **kwargs):
        if (
            self.__class__.__name__.startswith("MiniCPMSALA")
            and getattr(self, "_supports_flash_attn_2", False)
            and not getattr(self, "_supports_flash_attn", False)
        ):
            self._supports_flash_attn = True
        return original(self, *args, **kwargs)

    PreTrainedModel._flash_attn_can_dispatch = patched
    print("[compat] installed MiniCPM-SALA-specific FA2 dispatch bridge")
    return True


def maybe_enable_tied_weight_keys_compat() -> bool:
    try:
        from transformers import modeling_utils as modeling_utils
    except Exception as exc:
        print(f"[compat] unable to import modeling_utils for tied-weight compatibility: {exc}")
        return False

    original = getattr(modeling_utils, "_get_tied_weight_keys", None)
    if original is None:
        print("[compat] _get_tied_weight_keys missing; skipping tied-weight compatibility shim")
        return False

    def patched(module):
        tied_weight_keys = []
        for name, submodule in module.named_modules():
            tied = getattr(submodule, "_tied_weights_keys", {}) or {}
            if isinstance(tied, dict):
                keys_iter = tied.keys()
            elif isinstance(tied, (list, tuple, set)):
                keys_iter = tied
            else:
                keys_method = getattr(tied, "keys", None)
                if keys_method is None:
                    continue
                keys_iter = keys_method()
            tied_weight_keys.extend([f"{name}.{k}" if name else k for k in keys_iter])
        return tied_weight_keys

    modeling_utils._get_tied_weight_keys = patched
    print("[compat] installed tied-weight key compatibility shim for list-based _tied_weights_keys")
    return True


def load_calibration_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    by_source = Counter(row.get("source", "<unknown>") for row in rows)
    by_task = Counter(row.get("task", "<unknown>") for row in rows)
    by_bucket = Counter(row.get("bucket", "<unknown>") for row in rows)
    raw_tokens = [row.get("raw_prompt_tokens", 0) for row in rows]
    calib_tokens = [row.get("calib_prompt_tokens", 0) for row in rows]
    raw_tokens.sort()
    calib_tokens.sort()
    return {
        "count": len(rows),
        "by_source": dict(sorted(by_source.items())),
        "by_task": dict(sorted(by_task.items())),
        "by_bucket": dict(sorted(by_bucket.items())),
        "raw_tokens": {
            "p50": raw_tokens[len(raw_tokens) // 2],
            "p90": raw_tokens[min(len(raw_tokens) - 1, int(len(raw_tokens) * 0.9))],
            "max": raw_tokens[-1],
        },
        "calib_tokens": {
            "p50": calib_tokens[len(calib_tokens) // 2],
            "p90": calib_tokens[min(len(calib_tokens) - 1, int(len(calib_tokens) * 0.9))],
            "max": calib_tokens[-1],
        },
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for GPTQ quantization")

    random.seed(args.seed)
    dtype = torch_dtype_from_name(args.dtype)
    model_path = Path(args.model_path)
    calibration_path = Path(args.calibration_path)
    output_path = Path(args.output_path)

    rows = load_calibration_rows(calibration_path)
    if not rows:
        raise SystemExit(f"Calibration file is empty: {calibration_path}")

    if len(rows) > args.num_samples:
        rng = random.Random(args.seed)
        rows = list(rows)
        rng.shuffle(rows)
        rows = rows[: args.num_samples]

    summary = summarize_rows(rows)
    texts = [row["calib_text"] for row in rows]

    print("=" * 72)
    print("MiniCPM-SALA GPTQ W4A16 quantization")
    print(f"model_path={model_path}")
    print(f"calibration_path={calibration_path}")
    print(f"output_path={output_path}")
    print(f"num_samples={len(texts)}")
    print(f"dtype={args.dtype}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print("=" * 72)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=False,
    )
    original_auto_config_from_pretrained = AutoConfig.from_pretrained

    def patched_auto_config_from_pretrained(*args, **kwargs):
        cfg = original_auto_config_from_pretrained(*args, **kwargs)
        return normalize_minicpm_sala_config(cfg)

    AutoConfig.from_pretrained = patched_auto_config_from_pretrained
    config = normalize_minicpm_sala_config(
        original_auto_config_from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )
    )
    compat_defuser_disabled = maybe_disable_incompatible_defuser()
    compat_fa2_enabled = maybe_enable_minicpm_sala_fa2_compat()
    compat_tied_weights_enabled = maybe_enable_tied_weight_keys_compat()

    from gptqmodel import GPTQModel, QuantizeConfig
    from gptqmodel.models.auto import MODEL_MAP, SUPPORTED_MODELS
    from gptqmodel.models.definitions.minicpm import MiniCPMGPTQ

    class MiniCPMSALAGPTQ(MiniCPMGPTQ):
        layer_modules_strict = False

    # MiniCPM-SALA mixes sparse MiniCPM attention and LightningAttention.
    # Quantize only the linear submodules that exist across both attention types.
    MiniCPMSALAGPTQ.module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:1", "v_proj:2", "o_proj:3"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj", "up_proj", "down_proj"),
        },
    ]
    MODEL_MAP["minicpm_sala"] = MiniCPMSALAGPTQ
    if "minicpm_sala" not in SUPPORTED_MODELS:
        SUPPORTED_MODELS.append("minicpm_sala")

    qc = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        sym=args.sym,
        desc_act=args.desc_act,
        device="cuda:0",
        offload_to_disk=False,
    )

    if compat_defuser_disabled:
        print("[compat] using GPTQ without defuser fused-block rewrites")
    if compat_fa2_enabled:
        print("[compat] using MiniCPM-SALA-specific FA2 dispatch bridge")
    if compat_tied_weights_enabled:
        print("[compat] using tied-weight key compatibility shim")

    started = time.time()
    print("[1/3] Loading base model...")
    model = GPTQModel.from_pretrained(
        str(model_path),
        quantize_config=qc,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model.layer_modules_strict = False

    try:
        param_device = next(model.model.parameters()).device
    except StopIteration:
        param_device = torch.device("cpu")
    print(f"[1/3] initial_device={param_device}")
    if param_device.type != "cuda":
        print("[1/3] moving model to cuda:0")
        model.model.to("cuda:0")

    print("[2/3] Running GPTQ quantization...")
    quant_report = model.quantize(
        calibration=texts,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_path": str(model_path),
        "calibration_path": str(calibration_path),
        "output_path": str(output_path),
        "num_samples": len(texts),
        "dtype": args.dtype,
        "bits": args.bits,
        "group_size": args.group_size,
        "sym": args.sym,
        "desc_act": args.desc_act,
        "compat_defuser_disabled": compat_defuser_disabled,
        "compat_fa2_bridge": compat_fa2_enabled,
        "compat_tied_weights": compat_tied_weights_enabled,
        "summary": summary,
        "quant_report": quant_report,
        "elapsed_s": round(time.time() - started, 2),
    }
    (output_path / "codex_gptq_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[3/3] Saving GPTQ checkpoint...")
    model.save(str(output_path), max_shard_size=args.max_shard_size)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("saved_quantize_config", (output_path / "quantize_config.json").exists())
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
