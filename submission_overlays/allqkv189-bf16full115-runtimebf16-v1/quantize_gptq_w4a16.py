#!/usr/bin/env python3
"""
Quantize MiniCPM-SALA to W4A16 GPTQ on GPU and save a GPTQ-formatted checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

TARGET_MODULE_LEAVES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "o_gate",
    "gate_proj",
    "up_proj",
    "down_proj",
)


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
    parser.add_argument(
        "--calibration-concat-size",
        type=int,
        default=None,
        help="Optional GPTQModel calibration chunk size. Long calibration rows will be split into fixed-size chunks.",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--sym", action="store_true", default=True)
    parser.add_argument("--no-sym", action="store_true", help="Disable symmetric quantization.")
    parser.add_argument("--desc-act", action="store_true", default=False)
    parser.add_argument(
        "--static-groups",
        action="store_true",
        default=False,
        help="Enable static group organization in GPTQ.",
    )
    parser.add_argument(
        "--act-group-aware",
        action="store_true",
        default=False,
        help="Enable group-aware activation ordering when supported by GPTQModel.",
    )
    parser.add_argument(
        "--no-true-sequential",
        action="store_true",
        help="Disable true sequential quantization.",
    )
    parser.add_argument(
        "--damp-percent",
        type=float,
        default=None,
        help="Optional explicit Hessian damping percentage override.",
    )
    parser.add_argument(
        "--damp-auto-increment",
        type=float,
        default=None,
        help="Optional damping auto-increment override.",
    )
    parser.add_argument(
        "--offload-to-disk",
        action="store_true",
        default=False,
        help="Enable GPTQModel disk offload for lower peak VRAM usage during quantization.",
    )
    parser.add_argument(
        "--offload-to-disk-path",
        default="",
        help="Optional explicit GPTQModel disk-offload root. Defaults to GPTQModel auto path when unset.",
    )
    parser.add_argument(
        "--vram-strategy",
        choices=["exclusive", "balanced"],
        default="exclusive",
        help="GPTQModel VRAM strategy for quantization.",
    )
    parser.add_argument(
        "--gc-mode",
        choices=["interval", "on_stage_end"],
        default="interval",
        help="GPTQModel garbage-collection strategy during quantization.",
    )
    parser.add_argument(
        "--cpu-cache-layer-outputs",
        action="store_true",
        default=False,
        help="Move cached layer outputs to CPU between GPTQ replay stages to trade speed for lower VRAM.",
    )
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument(
        "--dynamic-config-path",
        default="",
        help="Optional JSON file with per-module dynamic GPTQ overrides using +:regex / -:regex rules.",
    )
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


def move_nested_to_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().to("cpu")
    if isinstance(obj, list):
        return [move_nested_to_cpu(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(move_nested_to_cpu(item) for item in obj)
    if isinstance(obj, dict):
        return {key: move_nested_to_cpu(value) for key, value in obj.items()}
    return obj


def maybe_enable_cpu_layer_output_cache() -> bool:
    try:
        from gptqmodel.looper.loop_processor import LoopProcessor
    except Exception as exc:
        print(f"[compat] unable to import LoopProcessor for CPU output-cache shim: {exc}")
        return False

    original_receive_layer_inputs = getattr(LoopProcessor, "receive_layer_inputs", None)
    original_receive_input_cache = getattr(LoopProcessor, "receive_input_cache", None)
    if original_receive_layer_inputs is None or original_receive_input_cache is None:
        print("[compat] LoopProcessor cache hooks missing; skipping CPU output-cache shim")
        return False

    def patched_receive_layer_inputs(self, layer_inputs):
        self.inputs_cache.layer_inputs = move_nested_to_cpu(layer_inputs)

    def patched_receive_input_cache(self, input_cache):
        self.inputs_cache = input_cache
        self.inputs_cache.layer_inputs = move_nested_to_cpu(self.inputs_cache.layer_inputs)

    LoopProcessor.receive_layer_inputs = patched_receive_layer_inputs
    LoopProcessor.receive_input_cache = patched_receive_input_cache
    print("[compat] installed GPTQ CPU layer-output cache shim")
    return True


def maybe_enable_attnqkv_subset_diagnostics(target_layers: set[int]) -> bool:
    try:
        import gptqmodel.looper.module_looper as module_looper_mod
        import gptqmodel.looper.stage_layer as stage_layer_mod
        from gptqmodel.looper.gptq_processor import GPTQProcessor
        from gptqmodel.looper.module_looper import ModuleLooper
    except Exception as exc:
        print(f"[compat] unable to import GPTQModel internals for attnqkv diagnostics: {exc}")
        return False

    if getattr(ModuleLooper, "_soar_attnqkv_diag_installed", False):
        return False

    interesting_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj")

    def _is_interesting_name(name: str) -> bool:
        return "self_attn" in name or name.endswith(interesting_suffixes)

    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        return obj

    original_run_layer_stage = stage_layer_mod.run_layer_stage
    original_create_named_modules = ModuleLooper.create_named_modules
    original_preprocess = GPTQProcessor.preprocess

    def patched_run_layer_stage(
        looper,
        *,
        layers,
        layer_modules,
        layers_prefix,
        failsafe,
        shared_kv_cache_dict,
        pb,
        layer_count,
        region_timer,
        finalize_progress_cls,
        logger=None,
    ):
        if not getattr(looper, "_soar_attnqkv_diag_logged", False):
            payload = {
                "event": "stage_layer_modules",
                "layers_prefix": layers_prefix,
                "layer_modules": layer_modules,
                "target_layers": sorted(target_layers),
            }
            print("[attnqkv-debug] " + json.dumps(payload, ensure_ascii=False))
            looper._soar_attnqkv_diag_logged = True
        return original_run_layer_stage(
            looper,
            layers=layers,
            layer_modules=layer_modules,
            layers_prefix=layers_prefix,
            failsafe=failsafe,
            shared_kv_cache_dict=shared_kv_cache_dict,
            pb=pb,
            layer_count=layer_count,
            region_timer=region_timer,
            finalize_progress_cls=finalize_progress_cls,
            logger=logger,
        )

    def patched_create_named_modules(
        self,
        module,
        full,
        is_lm_head_module,
        layer_index,
        layers_prefix,
        names,
        processor,
        failsafe,
        layer_module=None,
    ):
        should_log = (
            not is_lm_head_module
            and layer_index in target_layers
            and any(_is_interesting_name(name) for name in names)
        )
        if should_log:
            rows = []
            for name in names:
                full_name = f"{layers_prefix}.{layer_index}.{name}" if layers_prefix else f"{layer_index}.{name}"
                dynamic_result = None
                try:
                    dynamic_result = processor.qcfg.dynamic_get(layer_name=full_name)
                except Exception as exc:  # pragma: no cover - diagnostics only
                    dynamic_result = {"error": str(exc)}
                rows.append(
                    {
                        "name": name,
                        "full_name": full_name,
                        "dynamic": _json_safe(dynamic_result),
                        "in_full": name in full,
                    }
                )
            payload = {
                "event": "create_named_modules_pre",
                "layer_index": layer_index,
                "layers_prefix": layers_prefix,
                "processor": processor.name() if hasattr(processor, "name") else type(processor).__name__,
                "names": names,
                "rows": rows,
            }
            print("[attnqkv-debug] " + json.dumps(payload, ensure_ascii=False))

        subset = original_create_named_modules(
            self,
            module=module,
            full=full,
            is_lm_head_module=is_lm_head_module,
            layer_index=layer_index,
            layers_prefix=layers_prefix,
            names=names,
            processor=processor,
            failsafe=failsafe,
            layer_module=layer_module,
        )

        if should_log:
            payload = {
                "event": "create_named_modules_post",
                "layer_index": layer_index,
                "names": names,
                "subset_keys": list(subset.keys()),
            }
            print("[attnqkv-debug] " + json.dumps(payload, ensure_ascii=False))

        return subset

    def patched_preprocess(self, module, failsafe=None, **kwargs):
        full_name = getattr(module, "full_name", "")
        layer_index = getattr(module, "layer_index", None)
        should_log = (
            layer_index in target_layers
            and ".self_attn." in full_name
            and full_name.rsplit(".", 1)[-1] in interesting_suffixes
        )
        if should_log:
            dynamic_result = None
            try:
                dynamic_result = self.qcfg.dynamic_get(layer_name=full_name)
            except Exception as exc:  # pragma: no cover - diagnostics only
                dynamic_result = {"error": str(exc)}
            payload = {
                "event": "gptq_preprocess_pre",
                "layer_index": layer_index,
                "module_name": getattr(module, "name", ""),
                "full_name": full_name,
                "dynamic": _json_safe(dynamic_result),
            }
            print("[attnqkv-debug] " + json.dumps(payload, ensure_ascii=False))

        result = original_preprocess(self, module, failsafe=failsafe, **kwargs)

        if should_log:
            payload = {
                "event": "gptq_preprocess_post",
                "layer_index": layer_index,
                "module_name": getattr(module, "name", ""),
                "full_name": full_name,
                "task_created": getattr(module, "name", "") in getattr(self, "tasks", {}),
            }
            print("[attnqkv-debug] " + json.dumps(payload, ensure_ascii=False))

        return result

    stage_layer_mod.run_layer_stage = patched_run_layer_stage
    module_looper_mod.run_layer_stage = patched_run_layer_stage
    ModuleLooper.create_named_modules = patched_create_named_modules
    GPTQProcessor.preprocess = patched_preprocess
    ModuleLooper._soar_attnqkv_diag_installed = True
    print(f"[compat] installed attnqkv subset diagnostics for layers={sorted(target_layers)}")
    return True


def sync_tokenizer_assets(model_path: Path, output_path: Path) -> list[str]:
    copied: list[str] = []
    for name in (
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "added_tokens.json",
        "chat_template.jinja",
    ):
        src = model_path / name
        dst = output_path / name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


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


def load_dynamic_config(path: Path | None) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]] | None]:
    if path is None:
        return None, None

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Dynamic config must be a JSON object: {path}")

    normalized: dict[str, dict[str, Any]] = {}
    for pattern, overrides in payload.items():
        if not isinstance(pattern, str):
            raise SystemExit(f"Dynamic config pattern must be a string: {pattern!r}")
        if not (pattern.startswith("+:") or pattern.startswith("-:")):
            raise SystemExit(f"Dynamic config pattern must start with +: or -:: {pattern}")
        try:
            re.compile(pattern[2:])
        except re.error as exc:
            raise SystemExit(f"Invalid dynamic config regex {pattern!r}: {exc}") from exc

        if pattern.startswith("-:"):
            if overrides is None or isinstance(overrides, bool):
                normalized[pattern] = {}
            elif isinstance(overrides, dict):
                normalized[pattern] = dict(overrides)
            else:
                raise SystemExit(
                    f"Negative dynamic rule {pattern!r} must map to an object, bool, or null"
                )
            continue

        if not isinstance(overrides, dict):
            raise SystemExit(f"Positive dynamic rule {pattern!r} must map to a JSON object")

        cleaned: dict[str, Any] = {}
        for key, value in overrides.items():
            if key in {"bits", "group_size"} and value is not None:
                cleaned[key] = int(value)
            elif key in {"sym", "desc_act"} and value is not None:
                cleaned[key] = bool(value)
            else:
                cleaned[key] = value
        normalized[pattern] = cleaned

    return payload, normalized


def build_minicpm_qkv_runtime_aliases(
    dynamic_config: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if not dynamic_config:
        return {}

    aliases: dict[str, dict[str, Any]] = {}
    suffix_patterns = (
        r"\\\.\(q_proj\|k_proj\|v_proj\)\$$",
        r"\\\.\(q_proj\|k_proj\|v_proj\|o_proj\)\$$",
        r"\\\.\(\?:q_proj\|k_proj\|v_proj\)\$$",
        r"\\\.\(\?:q_proj\|k_proj\|v_proj\|o_proj\)\$$",
        r"\\\.\[qkv\]_proj\$$",
    )

    for pattern, overrides in dynamic_config.items():
        alias_pattern = None
        for suffix in suffix_patterns:
            if re.search(suffix, pattern[2:]) is None:
                continue
            alias_pattern = pattern[:2] + re.sub(suffix, r"\\.qkv_proj$", pattern[2:])
            break
        if alias_pattern is None:
            continue
        if alias_pattern in dynamic_config or alias_pattern in aliases:
            continue
        aliases[alias_pattern] = dict(overrides)

    return aliases


def collect_quant_target_module_names(root_module) -> list[str]:
    names: list[str] = []
    for name, module in root_module.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in TARGET_MODULE_LEAVES:
            continue
        if not hasattr(module, "weight"):
            continue
        names.append(name)
    return sorted(set(names))


def resolve_dynamic_rule(
    dynamic_config: dict[str, dict[str, Any]] | None,
    module_name: str,
) -> tuple[str | None, str | None]:
    if not dynamic_config:
        return None, None
    for pattern in dynamic_config:
        expr = pattern[2:]
        if re.match(expr, module_name) is None:
            continue
        if pattern.startswith("-:"):
            return pattern, "exclude"
        return pattern, "override"
    return None, None


def build_dynamic_manifest(
    *,
    module_names: list[str],
    dynamic_config: dict[str, dict[str, Any]] | None,
    dynamic_get,
    bits: int,
    group_size: int,
    sym: bool,
    desc_act: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not dynamic_config:
        return [], []

    pattern_reports = [
        {
            "pattern": pattern,
            "mode": "exclude" if pattern.startswith("-:") else "override",
            "matches": [],
        }
        for pattern in dynamic_config
    ]
    pattern_index = {item["pattern"]: item for item in pattern_reports}

    module_reports: list[dict[str, Any]] = []
    for module_name in module_names:
        matched_pattern, mode = resolve_dynamic_rule(dynamic_config, module_name)
        if matched_pattern is not None:
            pattern_index[matched_pattern]["matches"].append(module_name)
        enabled = mode != "exclude"
        module_reports.append(
            {
                "module_name": module_name,
                "matched_pattern": matched_pattern,
                "mode": mode or "default",
                "enabled": enabled,
                "bits": dynamic_get(dynamic_config, module_name=module_name, key="bits", default=bits),
                "group_size": dynamic_get(
                    dynamic_config,
                    module_name=module_name,
                    key="group_size",
                    default=group_size,
                ),
                "sym": dynamic_get(dynamic_config, module_name=module_name, key="sym", default=sym),
                "desc_act": dynamic_get(
                    dynamic_config,
                    module_name=module_name,
                    key="desc_act",
                    default=desc_act,
                ),
            }
        )
    return pattern_reports, module_reports



def maybe_enable_gptq_hessian_activation_sanitize() -> bool:
    """Filter non-finite token rows before GPTQ Hessian construction.

    MiniCPM-SALA long calibration rows can produce NaNs in tensors captured by
    GPTQModel for Hessian construction. Those NaNs poison X^T X and force every
    affected module into RTN failsafe. Filtering entire token rows is cleaner
    than turning NaN into zero: the Hessian only sees rows that actually had
    finite activations, and nsamples reflects the kept rows.
    """
    try:
        import transformers as _transformers
        from torch import nn as _nn
        from gptqmodel.quantization import gptq as gptq_mod
        from gptqmodel.quantization.gptq import GPTQ
    except Exception as exc:
        print(f"[compat] unable to import GPTQ for Hessian activation sanitize: {exc}")
        return False

    if getattr(GPTQ, "_soar_hessian_activation_sanitize_installed", False):
        return False

    original_add_batch = GPTQ.add_batch
    original_process_batch = GPTQ.process_batch
    trace_path = os.environ.get("SOAR_GPTQ_NAN_TRACE_PATH", "")
    event_counts: dict[str, int] = {}

    def gptq_full_module_name(gptq_obj) -> str:
        named_module = getattr(gptq_obj, "_named_module", None)
        full_name = getattr(named_module, "full_name", None)
        return full_name or getattr(gptq_obj, "name", "<unknown>")

    def emit_trace(event: dict[str, Any]) -> None:
        if trace_path:
            try:
                with open(trace_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            except Exception as exc:
                print(f"[compat] unable to write GPTQ finite-row trace: {exc}")
        module_name = event["module"]
        count = event_counts.get(module_name, 0)
        if count < 5:
            print(
                f"[compat] filtered GPTQ Hessian activations for {module_name}: "
                f"batch={event.get('batch_index')}, bad_rows={event['bad_rows']}/{event['raw_rows']}, "
                f"nonfinite={event['nonfinite']}/{event['total_elements']}, "
                f"finite_absmax={event['finite_absmax']:.6g}"
            )
        event_counts[module_name] = count + 1

    def patched_add_batch(self, inp: torch.Tensor, out: torch.Tensor, batch_index: int | None = None):
        self._soar_current_batch_index = batch_index
        try:
            return original_add_batch(self, inp, out, batch_index=batch_index)
        finally:
            self._soar_current_batch_index = None

    def patched_process_batch(self, inp: torch.Tensor):
        # Current MiniCPM-SALA quantization targets are Linear/Conv1D-style
        # projections. Delegate uncommon modules to GPTQModel unchanged.
        if not isinstance(self.module, (_nn.Linear, _transformers.Conv1D)):
            return original_process_batch(self, inp)

        inp_device = gptq_mod.get_device(inp)
        reshaped_inp = inp.reshape(-1, inp.shape[-1]).contiguous()
        if self._tp_pad_cols:
            pad = reshaped_inp.new_zeros((reshaped_inp.shape[0], self._tp_pad_cols))
            reshaped_inp = torch.cat((reshaped_inp, pad), dim=1)
            del pad

        canonical_device = torch.device(inp_device)
        raw_rows = int(reshaped_inp.shape[0])
        if raw_rows == 0:
            del reshaped_inp
            return 0, None, canonical_device

        finite = torch.isfinite(reshaped_inp)
        if not bool(finite.all().item()):
            row_finite = finite.all(dim=1)
            kept_rows = int(row_finite.sum().item())
            bad_rows = raw_rows - kept_rows
            nonfinite = int((~finite).sum().item())
            total_elements = int(reshaped_inp.numel())
            finite_vals = reshaped_inp[finite]
            finite_absmax = float(finite_vals.float().abs().max().item()) if finite_vals.numel() else 0.0
            event = {
                "event": "gptq_hessian_finite_row_filter",
                "module": gptq_full_module_name(self),
                "leaf_module": getattr(self, "name", "<unknown>"),
                "layer_index": getattr(getattr(self, "_named_module", None), "layer_index", None),
                "batch_index": getattr(self, "_soar_current_batch_index", None),
                "raw_rows": raw_rows,
                "kept_rows": kept_rows,
                "bad_rows": bad_rows,
                "bad_row_ratio": bad_rows / raw_rows if raw_rows else 0.0,
                "nonfinite": nonfinite,
                "nan": int(torch.isnan(reshaped_inp).sum().item()),
                "posinf": int(torch.isposinf(reshaped_inp).sum().item()),
                "neginf": int(torch.isneginf(reshaped_inp).sum().item()),
                "total_elements": total_elements,
                "nonfinite_ratio": nonfinite / total_elements if total_elements else 0.0,
                "finite_absmax": finite_absmax,
                "shape": [int(dim) for dim in reshaped_inp.shape],
            }
            emit_trace(event)
            if kept_rows == 0:
                del reshaped_inp
                return 0, None, canonical_device
            reshaped_inp = reshaped_inp[row_finite].contiguous()

        batch_token_size = int(reshaped_inp.shape[0])
        try:
            xtx = self.compute_hessian_xtx(reshaped_inp).to(dtype=torch.float32)
        except RuntimeError as exc:
            if torch.device(inp_device).type == "cuda" and "out of memory" in str(exc).lower():
                gptq_mod.log.warn(
                    "GPTQ module '%s' fell back to CPU Hessian accumulation due to GPU OOM during batch processing.",
                    getattr(self, "name", "<unknown>"),
                )
                reshaped_inp_cpu = reshaped_inp.to(device=torch.device("cpu"))
                del reshaped_inp
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                canonical_device = torch.device("cpu")
                xtx = self.compute_hessian_xtx(reshaped_inp_cpu).to(dtype=torch.float32)
                xtx = xtx.detach()
                del reshaped_inp_cpu
            else:
                del reshaped_inp
                raise
        else:
            xtx = xtx.detach()
            del reshaped_inp

        self._snapshot_borrow_workspace_stats(context="process_batch")
        return batch_token_size, xtx, canonical_device

    GPTQ.add_batch = patched_add_batch
    GPTQ.process_batch = patched_process_batch
    GPTQ._soar_hessian_activation_sanitize_installed = True
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        Path(trace_path).write_text("", encoding="utf-8")
        print(f"[compat] GPTQ finite-row trace path: {trace_path}")
    print("[compat] installed GPTQ Hessian finite-row filter shim")
    return True

def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for GPTQ quantization")

    random.seed(args.seed)
    dtype = torch_dtype_from_name(args.dtype)
    model_path = Path(args.model_path)
    calibration_path = Path(args.calibration_path)
    output_path = Path(args.output_path)
    dynamic_config_path = Path(args.dynamic_config_path).expanduser() if args.dynamic_config_path else None
    sym = args.sym and not args.no_sym
    true_sequential = not args.no_true_sequential

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
    dynamic_config_raw, dynamic_config = load_dynamic_config(dynamic_config_path)
    runtime_dynamic_aliases = build_minicpm_qkv_runtime_aliases(dynamic_config)
    if dynamic_config is not None and runtime_dynamic_aliases:
        dynamic_config = dict(dynamic_config)
        dynamic_config.update(runtime_dynamic_aliases)

    print("=" * 72)
    print("MiniCPM-SALA GPTQ W4A16 quantization")
    print(f"model_path={model_path}")
    print(f"calibration_path={calibration_path}")
    print(f"output_path={output_path}")
    print(f"dynamic_config_path={dynamic_config_path or '<none>'}")
    print(f"num_samples={len(texts)}")
    print(f"dtype={args.dtype}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print("=" * 72)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if runtime_dynamic_aliases:
        print("[dynamic] added MiniCPM fused-qkv runtime aliases:")
        print(json.dumps(runtime_dynamic_aliases, ensure_ascii=False, indent=2))

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
    from gptqmodel.quantization.config import GcMode, VramStrategy, dynamic_get
    hessian_activation_sanitize_enabled = maybe_enable_gptq_hessian_activation_sanitize()
    cpu_output_cache_enabled = False
    if args.cpu_cache_layer_outputs:
        cpu_output_cache_enabled = maybe_enable_cpu_layer_output_cache()

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

    attnqkv_diag_enabled = False
    if dynamic_config_path is not None and "attnqkv-allowlist" in dynamic_config_path.name:
        attnqkv_diag_enabled = maybe_enable_attnqkv_subset_diagnostics(
            target_layers={0, 10, 12, 13, 14, 19, 20, 21, 23}
        )

    qc = QuantizeConfig(
        bits=args.bits,
        dynamic=dynamic_config,
        group_size=args.group_size,
        damp_percent=args.damp_percent,
        damp_auto_increment=args.damp_auto_increment,
        static_groups=args.static_groups,
        sym=sym,
        desc_act=args.desc_act,
        act_group_aware=args.act_group_aware,
        true_sequential=true_sequential,
        device="cuda:0",
        offload_to_disk=args.offload_to_disk,
        offload_to_disk_path=args.offload_to_disk_path or None,
        vram_strategy=VramStrategy(args.vram_strategy),
        gc_mode=GcMode(args.gc_mode),
    )

    # GPTQModel 5.8.0 rewrites `dynamic` to put all negative rules first.
    # That breaks our allowlist-style attnQKV experiments because `dynamic_get()`
    # is first-match and the global negative skip then shadows the earlier
    # positive allowlist. Restore the original JSON order for this experiment.
    if dynamic_config is not None and dynamic_config_path is not None and "attnqkv" in dynamic_config_path.name:
        qc.dynamic = dict(dynamic_config)
        print("[compat] restored attnqkv dynamic rule order after QuantizeConfig init")

    if compat_defuser_disabled:
        print("[compat] using GPTQ without defuser fused-block rewrites")
    if compat_fa2_enabled:
        print("[compat] using MiniCPM-SALA-specific FA2 dispatch bridge")
    if compat_tied_weights_enabled:
        print("[compat] using tied-weight key compatibility shim")
    if cpu_output_cache_enabled:
        print("[compat] caching GPTQ layer outputs on CPU between replay stages")
    if hessian_activation_sanitize_enabled:
        print("[compat] filtering non-finite GPTQ Hessian activation rows")
    if attnqkv_diag_enabled:
        print("[compat] attention allowlist diagnostics enabled")

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
    if param_device.type == "cpu":
        print("[1/3] moving model to cuda:0")
        model.model.to("cuda:0")
    elif param_device.type == "meta":
        print("[1/3] keeping model on meta device for GPTQModel-managed on-demand materialization")

    quant_target_modules = collect_quant_target_module_names(model.model)
    dynamic_pattern_reports, dynamic_module_reports = build_dynamic_manifest(
        module_names=quant_target_modules,
        dynamic_config=dynamic_config,
        dynamic_get=dynamic_get,
        bits=args.bits,
        group_size=args.group_size,
        sym=sym,
        desc_act=args.desc_act,
    )
    if dynamic_pattern_reports:
        print("[dynamic] matched modules:")
        print(
            json.dumps(
                {
                    "pattern_count": len(dynamic_pattern_reports),
                    "matched_module_count": sum(
                        1 for item in dynamic_module_reports if item["matched_pattern"] is not None
                    ),
                    "excluded_module_count": sum(
                        1 for item in dynamic_module_reports if not item["enabled"]
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    print("[2/3] Running GPTQ quantization...")
    quant_report = model.quantize(
        calibration=texts,
        calibration_concat_size=args.calibration_concat_size,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_path": str(model_path),
        "calibration_path": str(calibration_path),
        "output_path": str(output_path),
        "num_samples": len(texts),
        "calibration_concat_size": args.calibration_concat_size,
        "dtype": args.dtype,
        "bits": args.bits,
        "group_size": args.group_size,
        "sym": sym,
        "desc_act": args.desc_act,
        "static_groups": args.static_groups,
        "act_group_aware": args.act_group_aware,
        "true_sequential": true_sequential,
        "damp_percent": args.damp_percent,
        "damp_auto_increment": args.damp_auto_increment,
        "offload_to_disk": args.offload_to_disk,
        "offload_to_disk_path": args.offload_to_disk_path or None,
        "vram_strategy": args.vram_strategy,
        "gc_mode": args.gc_mode,
        "cpu_cache_layer_outputs": args.cpu_cache_layer_outputs,
        "dynamic_config_path": str(dynamic_config_path) if dynamic_config_path else None,
        "dynamic_config_raw": dynamic_config_raw,
        "dynamic_runtime_aliases": runtime_dynamic_aliases,
        "compat_defuser_disabled": compat_defuser_disabled,
        "compat_fa2_bridge": compat_fa2_enabled,
        "compat_tied_weights": compat_tied_weights_enabled,
        "summary": summary,
        "quant_target_module_count": len(quant_target_modules),
        "dynamic_pattern_matches": dynamic_pattern_reports,
        "dynamic_modules": dynamic_module_reports,
        "quant_report": quant_report,
        "elapsed_s": round(time.time() - started, 2),
    }

    print("[3/3] Saving GPTQ checkpoint...")
    model.save(str(output_path), max_shard_size=args.max_shard_size)
    copied_tokenizer_assets = sync_tokenizer_assets(model_path, output_path)
    manifest["copied_tokenizer_assets"] = copied_tokenizer_assets
    (output_path / "codex_gptq_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("saved_quantize_config", (output_path / "quantize_config.json").exists())
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
