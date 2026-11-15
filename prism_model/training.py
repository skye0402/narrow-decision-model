"""First AI Core execution: frozen-encoder training mechanics, never task quality."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from time import perf_counter

import torch
from torch import nn

from .evaluation import load_cases
from .neural import FrozenDecisionModel, NeuralJudge, OptionScorer, load_head, load_local_encoder, save_head
from .orchestrator import Orchestrator, resolve_reference
from .provenance import checked_file, file_sha256, verify_files, verify_source
from .renderer import group_row, requirement_row

SMOKE_CASE_IDS = ("01", "03", "06", "07", "13")
ENCODER_FILES = ("README.md", "config.json", "provenance.json", "pytorch_model.bin", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json")


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def state_sha256(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(memoryview(value.detach().cpu().contiguous().numpy()).cast("B"))
    return digest.hexdigest()


def select_device(name, expected_gpu):
    if name == "cpu":
        return torch.device("cpu"), {"device": "cpu", "cuda_execution_verified": False}
    if name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU validation must be selected explicitly")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    if props.name != expected_gpu:
        raise RuntimeError("Allocated GPU does not match the expected device")
    try:
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], timeout=10).decode().strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        driver = "unavailable"
    free_bytes, _ = torch.cuda.mem_get_info(0)
    return torch.device("cuda:0"), {"device": "cuda:0", "gpu_name": props.name, "gpu_free_bytes_at_start": free_bytes,
        "gpu_total_bytes": props.total_memory, "gpu_count": torch.cuda.device_count(),
        "compute_capability": list(torch.cuda.get_device_capability(0)), "driver_version": driver,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "cuda_execution_verified": True}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_smoke_data(directory):
    directory = Path(directory)
    manifest = json.loads(checked_file(directory, "dataset-manifest.json").read_text())
    if (manifest.get("version") != "training-data-1" or manifest.get("scope") != "development_mechanism_only"
            or manifest.get("case_ids") != list(SMOKE_CASE_IDS) or manifest.get("gold_labels_reviewed") is not False):
        raise ValueError("This execution accepts only the predeclared development smoke artifact")
    verify_files(directory, manifest["files_sha256"], required=("pilot_config.json", "smoke_cases.jsonl"),
                 exact=True, exclude=("dataset-manifest.json",))
    config = json.loads((directory/"pilot_config.json").read_text())
    cases = load_cases(directory/"smoke_cases.jsonl", config)
    if {case["case_id"] for case in cases} != set(SMOKE_CASE_IDS):
        raise ValueError("Unexpected smoke case selection")
    examples = []
    for case in cases:
        group = case["expected"]["group"]
        policy_id = case.get("policy_id") or config["default_policies"][group]
        context = {"request": case["request"], "reference": resolve_reference(case["request"], config), "policy_id": policy_id}
        for row, label in ((group_row(context, config), group),
                (requirement_row(context, config, config["policies"][policy_id]), case["expected"]["identification"])):
            examples.append({"case_id": case["case_id"], "row": row, "target": row["option_ids"].index(label)})
    return config, cases, examples, manifest


def cpu_roundtrip(encoder_dir, head_dir, examples, references):
    encoder, tokenizer, identity, _ = load_local_encoder(encoder_dir)
    head, metadata = load_head(head_dir, identity)
    model = FrozenDecisionModel(encoder, tokenizer, head, max_tokens=metadata["max_tokens"]).eval()
    results = []
    if len(examples) != len(references):
        raise ValueError("Verification row count mismatch")
    for example, reference in zip(examples, references):
        row = example["row"]
        if reference["text_sha256"] != row["text_sha256"] or reference["option_ids"] != row["option_ids"]:
            raise ValueError("Verification input changed")
        with torch.inference_mode():
            actual = model(row)
        expected = torch.tensor(reference["logits"], dtype=torch.float32)
        if not bool(torch.isfinite(actual).all()):
            raise ValueError("Non-finite CPU verification output")
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        results.append({"case_id": example["case_id"], "question_id": row["question_id"],
            "max_absolute_logit_difference": float((actual-expected).abs().max()),
            "max_absolute_probability_difference": float((actual.softmax(0)-expected.softmax(0)).abs().max()),
            "same_choice": int(actual.argmax()) == int(expected.argmax())})
    return model, results


def verify_bundle(directory, data_dir):
    directory = Path(directory)
    manifest = json.loads(checked_file(directory, "bundle-manifest.json").read_text())
    if (manifest.get("version") != "model-bundle-1" or manifest.get("complete") is not True
            or manifest.get("scope") != "development_mechanism_only"):
        raise ValueError("Incomplete or unsupported model bundle")
    verify_files(directory, manifest["files_sha256"], exact=True, exclude=("bundle-manifest.json",),
        required=("head/head.pt", "head/head.json", "encoder/pytorch_model.bin", "training-report.json", "source-manifest.json", "dataset-manifest.json"))
    report = json.loads((directory/"training-report.json").read_text())
    config, cases, examples, dataset = load_smoke_data(data_dir)
    if dataset != json.loads((directory/"dataset-manifest.json").read_text()):
        raise ValueError("Verification dataset differs from training")
    if report["calibrated"] is not False or report["qualifies_model_quality"] is not False or report["qualifies_cpu_budget"] is not False:
        raise ValueError("Prototype bundle contains unsupported qualification claims")
    model, comparisons = cpu_roundtrip(directory/"encoder", directory/"head", examples, report["reference_logits"])
    for case in cases:
        result = Orchestrator(config, NeuralJudge(model)).evaluate(case["request"])
        if result["status"] != "clarification_needed" or result["group_accepted"]:
            raise ValueError("Unset thresholds must prevent readiness")
    return {"kind": "cpu_model_bundle_verification", "rows": len(comparisons), "requests_gated": len(cases),
            "comparisons": comparisons, "calibrated": False, "qualifies_model_quality": False, "qualifies_cpu_budget": False}


def run_training(*, data_dir, encoder_dir, output_dir, source_root, source_manifest,
                 device="cuda", expected_gpu="NVIDIA L4", steps=60, seed=17, threads=2, max_tokens=512):
    if any(type(value) is not int or value < 1 for value in (steps, threads, max_tokens)):
        raise ValueError("Steps, threads, and token limit must be positive integers")
    output_dir, data_dir, encoder_dir = (Path(p).resolve() for p in (output_dir, data_dir, encoder_dir))
    if output_dir.exists() or any(output_dir.is_relative_to(p) or p.is_relative_to(output_dir) for p in (data_dir, encoder_dir, Path(source_root).resolve())):
        raise ValueError("Use a new output directory separate from inputs and software")
    verify_source(source_root, source_manifest)
    config, cases, examples, dataset = load_smoke_data(data_dir)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("highest")
    selected, runtime = select_device(device, expected_gpu)
    runtime.update(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
        transformers=importlib.metadata.version("transformers"), tokenizers=importlib.metadata.version("tokenizers"),
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(), precision="fp32")
    if selected.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected)
    emit("training_started", device=str(selected), examples=len(examples), steps=steps)
    encoder, tokenizer, identity, loading = load_local_encoder(encoder_dir)
    model = FrozenDecisionModel(encoder, tokenizer, OptionScorer(encoder.config.hidden_size), max_tokens=max_tokens).to(selected, dtype=torch.float32)
    before_encoder, before_head = state_sha256(encoder), state_sha256(model.head)
    synchronize(selected)
    started = perf_counter()
    cached = [(model.features(example["row"]), torch.tensor([example["target"]], device=selected)) for example in examples]
    synchronize(selected)
    extraction_seconds = perf_counter()-started
    emit("features_ready", rows=len(cached))
    model.train()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.003, weight_decay=0.01)

    def objective():
        return torch.stack([nn.functional.cross_entropy(model.head(features.context, features.options).unsqueeze(0), target)
                            for features, target in cached]).mean()

    losses, gradient_norms = [], []
    synchronize(selected)
    started = perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not bool(torch.isfinite(loss)):
            raise ValueError("Non-finite training loss")
        loss.backward()
        norm = torch.sqrt(sum(p.grad.square().sum() for p in model.head.parameters()))
        if not bool(torch.isfinite(norm)) or float(norm) <= 0:
            raise ValueError("Expected finite nonzero head gradients")
        losses.append(float(loss.detach())); gradient_norms.append(float(norm))
        optimizer.step()
        if step == 0 or (step+1) % 10 == 0 or step+1 == steps:
            emit("head_update", step=step+1, loss=losses[-1])
    synchronize(selected)
    training_seconds = perf_counter()-started
    with torch.no_grad():
        final_loss = float(objective())
    after_encoder = state_sha256(encoder)
    if (before_encoder != after_encoder or any(p.grad is not None for p in encoder.parameters())
            or state_sha256(model.head) == before_head or not final_loss < losses[0]):
        raise ValueError("Training mechanism did not validate")
    model.eval()
    references = []
    with torch.inference_mode():
        for example in examples:
            row = example["row"]
            references.append({"case_id": example["case_id"], "question_id": row["question_id"],
                "text_sha256": row["text_sha256"], "option_ids": row["option_ids"], "logits": model(row).cpu().tolist()})
    if selected.type == "cuda":
        runtime.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(selected),
                       gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(selected))
    training_info = {"scope": "development_mechanism_only", "source_manifest": source_manifest, "dataset_manifest": dataset,
        "case_ids": list(SMOKE_CASE_IDS), "gold_labels_reviewed": False, "steps": steps, "seed": seed,
        "learning_rate": 0.003, "weight_decay": 0.01, "optimizer": "AdamW", "frozen_encoder": True,
        "features_cached": True, "rows_per_update": len(examples)}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".training-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)/"model"
        staging.mkdir()
        (staging/"encoder").mkdir()
        for name in ENCODER_FILES:
            shutil.copyfile(checked_file(encoder_dir, name), staging/"encoder"/name)
        head_metadata = save_head(staging/"head", model.head, identity, max_tokens=max_tokens, training_info=training_info)
        # A fresh CPU encoder and restored head validate the saved files, including CUDA-to-CPU handoff.
        cpu_model, cpu_checks = cpu_roundtrip(staging/"encoder", staging/"head", examples, references)
        for case in cases:
            result = Orchestrator(config, NeuralJudge(cpu_model)).evaluate(case["request"])
            if result["status"] != "clarification_needed" or result["group_accepted"]:
                raise ValueError("Unset gates must prevent readiness")
        report = {"kind": "frozen_head_execution_smoke", "runtime": runtime, "source_manifest": source_manifest,
            "dataset_manifest": dataset, "encoder_identity": identity, "loading_info": loading,
            "training": {**training_info, "initial_loss": losses[0], "final_loss": final_loss,
                "loss_before_each_step": losses, "head_gradient_norms": gradient_norms,
                "encoder_state_before_sha256": before_encoder, "encoder_state_after_sha256": after_encoder,
                "encoder_unchanged": True, "head_changed": True, "feature_extraction_seconds": extraction_seconds,
                "head_training_seconds": training_seconds, "cached_feature_rows_processed": steps*len(examples)},
            "head": {"parameters": sum(p.numel() for p in model.head.parameters()), "weights_sha256": head_metadata["weights_sha256"]},
            "reference_logits": references, "cpu_reload": cpu_checks, "unset_gate_requests": len(cases),
            "calibrated": False, "qualifies_model_quality": False, "qualifies_cpu_budget": False,
            "notes": "Exposed draft development rows only. Cached frozen features; FP32 shared head. Timings are mechanism-run observations, not deployment latency or uncached training throughput. CPU/GPU logit checks use rtol=1e-3, atol=1e-3; calibration remains pending."}
        for name, value in (("training-report.json", report), ("source-manifest.json", source_manifest), ("dataset-manifest.json", dataset)):
            (staging/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
        bundle = {"version": "model-bundle-1", "scope": "development_mechanism_only", "complete": True,
            "files_sha256": {str(p.relative_to(staging)): file_sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()}}
        (staging/"bundle-manifest.json").write_text(json.dumps(bundle, indent=2)+"\n")
        verify_files(staging, bundle["files_sha256"], exact=True, exclude=("bundle-manifest.json",))
        staging.rename(output_dir)
    emit("artifact_complete", rows=len(examples), cuda_execution_verified=runtime["cuda_execution_verified"],
         qualifies_model_quality=False, qualifies_cpu_budget=False)
    return report
