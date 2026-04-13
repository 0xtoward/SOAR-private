#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


TARGET_MODULE_LEAVES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank post-quant module error for MiniCPM-SALA GPTQ checkpoints."
    )
    parser.add_argument("--base-model-path", default="/root/autodl-tmp/models")
    parser.add_argument("--quant-model-path", required=True)
    parser.add_argument("--calibration-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
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
    config._attn_implementation = "flash_attention_2"
    config._attn_implementation_internal = "flash_attention_2"
    if getattr(config, "rope_scaling", None):
        rope_scaling = dict(config.rope_scaling)
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if rope_type == "default":
            print("[compat] resetting rope_scaling from synthetic default to None")
            config.rope_scaling = None
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
            if raw:
                rows.append(json.loads(raw))
    return rows


def collect_target_module_names(root_module, *, require_weight: bool) -> list[str]:
    names: list[str] = []
    for name, module in root_module.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in TARGET_MODULE_LEAVES:
            continue
        if require_weight and not hasattr(module, "weight"):
            continue
        names.append(name)
    return sorted(set(names))


def extract_probe_tensor(output) -> torch.Tensor | None:
    if isinstance(output, (tuple, list)):
        if not output:
            return None
        output = output[0]
    if not torch.is_tensor(output):
        return None
    if output.ndim >= 3:
        output = output[:, -1, :]
    elif output.ndim == 1:
        output = output.unsqueeze(0)
    elif output.ndim == 0:
        output = output.reshape(1, 1)
    return output.detach().to(torch.float32).cpu()


def family_for_module(module_name: str) -> str:
    parts = module_name.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return module_name


def layer_for_module(module_name: str) -> int | None:
    parts = module_name.split(".")
    try:
        layer_idx = parts.index("layers") + 1
    except ValueError:
        return None
    if layer_idx >= len(parts):
        return None
    try:
        return int(parts[layer_idx])
    except ValueError:
        return None


def register_hooks(model, module_names: list[str], sink: dict[str, torch.Tensor]):
    hooks = []
    module_map = dict(model.named_modules())
    for module_name in module_names:
        module = module_map.get(module_name)
        if module is None:
            continue

        def hook(_module, _inputs, output, *, _name=module_name):
            tensor = extract_probe_tensor(output)
            if tensor is not None:
                sink[_name] = tensor

        hooks.append(module.register_forward_hook(hook))
    return hooks


def summarize_rows(rows: list[dict]) -> dict:
    by_task = Counter(row.get("task", "<unknown>") for row in rows)
    prompt_tokens = sorted(int(row.get("calib_prompt_tokens", 0)) for row in rows)
    return {
        "count": len(rows),
        "by_task": dict(sorted(by_task.items())),
        "prompt_tokens": {
            "min": prompt_tokens[0] if prompt_tokens else 0,
            "max": prompt_tokens[-1] if prompt_tokens else 0,
        },
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for layer-error ranking")

    random.seed(args.seed)
    dtype = torch_dtype_from_name(args.dtype)
    base_model_path = Path(args.base_model_path)
    quant_model_path = Path(args.quant_model_path)
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

    print("=" * 72)
    print("MiniCPM-SALA GPTQ layer-error ranking")
    print(f"base_model_path={base_model_path}")
    print(f"quant_model_path={quant_model_path}")
    print(f"calibration_path={calibration_path}")
    print(f"output_path={output_path}")
    print(f"num_samples={len(rows)}")
    print(f"dtype={args.dtype}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print("=" * 72)
    print(json.dumps(summarize_rows(rows), ensure_ascii=False, indent=2))

    original_auto_config_from_pretrained = AutoConfig.from_pretrained

    def patched_auto_config_from_pretrained(*cfg_args, **cfg_kwargs):
        cfg = original_auto_config_from_pretrained(*cfg_args, **cfg_kwargs)
        return normalize_minicpm_sala_config(cfg)

    AutoConfig.from_pretrained = patched_auto_config_from_pretrained

    compat_defuser_disabled = maybe_disable_incompatible_defuser()
    compat_fa2_enabled = maybe_enable_minicpm_sala_fa2_compat()
    compat_tied_weights_enabled = maybe_enable_tied_weight_keys_compat()

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        use_fast=False,
    )

    from gptqmodel import GPTQModel

    started = time.time()
    print("[1/3] loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    base_model.to("cuda:0")
    base_model.eval()

    print("[2/3] loading quant model...")
    quant_wrapper = GPTQModel.load(
        str(quant_model_path),
        backend="auto",
        device="cuda:0",
        trust_remote_code=True,
    )
    quant_model = getattr(quant_wrapper, "model", quant_wrapper)
    quant_model.eval()

    base_modules = collect_target_module_names(base_model, require_weight=True)
    quant_modules = set(collect_target_module_names(quant_model, require_weight=False))
    module_names = [name for name in base_modules if name in quant_modules]
    if not module_names:
        raise SystemExit("No overlapping target modules found between base and quant models")

    base_outputs: dict[str, torch.Tensor] = {}
    quant_outputs: dict[str, torch.Tensor] = {}
    base_hooks = register_hooks(base_model, module_names, base_outputs)
    quant_hooks = register_hooks(quant_model, module_names, quant_outputs)

    stats = defaultdict(
        lambda: {
            "sq_diff": 0.0,
            "sq_base": 0.0,
            "sq_quant": 0.0,
            "dot": 0.0,
            "samples": 0,
            "missing": 0,
        }
    )

    try:
        print("[3/3] probing module outputs...")
        for idx, row in enumerate(rows, start=1):
            text = row.get("calib_text") or row.get("text")
            if not text:
                continue
            encoded = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            if encoded["input_ids"].numel() == 0:
                continue

            base_inputs = {key: value.to("cuda:0") for key, value in encoded.items()}
            quant_inputs = {key: value.to("cuda:0") for key, value in encoded.items()}
            base_outputs.clear()
            quant_outputs.clear()
            with torch.inference_mode():
                base_model(**base_inputs, use_cache=False)
                quant_model(**quant_inputs, use_cache=False)

            compared = 0
            for module_name in module_names:
                base_tensor = base_outputs.get(module_name)
                quant_tensor = quant_outputs.get(module_name)
                if base_tensor is None or quant_tensor is None:
                    stats[module_name]["missing"] += 1
                    continue
                diff = quant_tensor - base_tensor
                stats[module_name]["sq_diff"] += float(diff.pow(2).sum().item())
                stats[module_name]["sq_base"] += float(base_tensor.pow(2).sum().item())
                stats[module_name]["sq_quant"] += float(quant_tensor.pow(2).sum().item())
                stats[module_name]["dot"] += float((base_tensor * quant_tensor).sum().item())
                stats[module_name]["samples"] += 1
                compared += 1
            print(
                f"[probe] sample={idx}/{len(rows)} prompt_tokens={row.get('calib_prompt_tokens')} "
                f"compared_modules={compared}"
            )
    finally:
        for hook in base_hooks + quant_hooks:
            hook.remove()
        del base_model
        del quant_model
        del quant_wrapper
        torch.cuda.empty_cache()

    ranking = []
    for module_name in module_names:
        item = stats[module_name]
        base_norm = item["sq_base"] ** 0.5
        quant_norm = item["sq_quant"] ** 0.5
        relative_l2 = (item["sq_diff"] ** 0.5) / max(base_norm, 1e-12)
        cosine = item["dot"] / max(base_norm * quant_norm, 1e-12)
        ranking.append(
            {
                "module_name": module_name,
                "layer": layer_for_module(module_name),
                "family": family_for_module(module_name),
                "relative_l2": relative_l2,
                "cosine": cosine,
                "samples": item["samples"],
                "missing": item["missing"],
            }
        )
    ranking.sort(key=lambda row: row["relative_l2"], reverse=True)

    family_summary = defaultdict(
        lambda: {
            "count": 0,
            "relative_l2_sum": 0.0,
            "cosine_sum": 0.0,
            "max_relative_l2": 0.0,
        }
    )
    for row in ranking:
        family = row["family"]
        family_summary[family]["count"] += 1
        family_summary[family]["relative_l2_sum"] += row["relative_l2"]
        family_summary[family]["cosine_sum"] += row["cosine"]
        family_summary[family]["max_relative_l2"] = max(
            family_summary[family]["max_relative_l2"], row["relative_l2"]
        )

    family_rows = []
    for family, item in family_summary.items():
        family_rows.append(
            {
                "family": family,
                "count": item["count"],
                "avg_relative_l2": item["relative_l2_sum"] / item["count"],
                "avg_cosine": item["cosine_sum"] / item["count"],
                "max_relative_l2": item["max_relative_l2"],
            }
        )
    family_rows.sort(key=lambda row: row["avg_relative_l2"], reverse=True)

    payload = {
        "base_model_path": str(base_model_path),
        "quant_model_path": str(quant_model_path),
        "calibration_path": str(calibration_path),
        "num_samples": len(rows),
        "dtype": args.dtype,
        "probe": "last_token_output",
        "compat": {
            "defuser_disabled": compat_defuser_disabled,
            "fa2_bridge": compat_fa2_enabled,
            "tied_weights": compat_tied_weights_enabled,
        },
        "elapsed_s": round(time.time() - started, 2),
        "module_count": len(module_names),
        "family_summary": family_rows,
        "ranking": ranking,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"top5": ranking[:5], "output_path": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
