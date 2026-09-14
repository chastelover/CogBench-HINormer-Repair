#!/usr/bin/env python3
"""Isolated official-runtime worker for one audited HINormer cap extension.

The worker never writes into the released result tree.  It imports the unchanged
official adapter, recreates the target trajectory from scratch, verifies the
complete stored prefix, uses validation-only evidence to choose 8k/12k, and
touches the test split exactly once after a cap has been selected.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class WorkerError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    def safe(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): safe(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(val) for val in item]
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_adapter(project: Path) -> Any:
    path = project / "scripts" / "run_official_hinormer.py"
    if not path.is_file() or path.is_symlink():
        raise WorkerError(f"unchanged official adapter is missing or linked: {path}")
    spec = importlib.util.spec_from_file_location("cogbench_official_adapter", path)
    if spec is None or spec.loader is None:
        raise WorkerError(f"cannot import official adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise WorkerError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkerError(f"expected a JSON object: {path}")
    return value


def history_frame(history: list[dict[str, float]], run_id: str) -> pd.DataFrame:
    frame = pd.DataFrame(history)
    frame.insert(0, "run_id", run_id)
    return frame


def history_records(path: Path) -> list[dict[str, Any]]:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise WorkerError(f"cannot read persisted history {path}: {exc}") from exc
    if "run_id" in frame:
        frame = frame.drop(columns=["run_id"])
    return frame.to_dict(orient="records")


def verify_csv_prefix(
    candidate_history: list[dict[str, float]],
    parent_path: Path,
    run_id: str,
    candidate_prefix_path: Path,
) -> dict[str, Any]:
    parent = pd.read_csv(parent_path)
    parent_rows = len(parent)
    if len(candidate_history) < parent_rows:
        raise WorkerError(
            f"extended trajectory ended at {len(candidate_history)} before reproducing "
            f"its {parent_rows}-epoch parent"
        )
    candidate = history_frame(candidate_history[:parent_rows], run_id)
    if list(candidate.columns) != list(parent.columns):
        raise WorkerError(
            "history-prefix columns differ: "
            f"candidate={list(candidate.columns)}, parent={list(parent.columns)}"
        )
    candidate_prefix_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(candidate_prefix_path, index=False)
    if candidate_prefix_path.read_bytes() != parent_path.read_bytes():
        numeric_columns = [name for name in parent.columns if name != "run_id"]
        mismatch: str | None = None
        for name in numeric_columns:
            left = pd.to_numeric(parent[name], errors="coerce").to_numpy(float)
            right = pd.to_numeric(candidate[name], errors="coerce").to_numpy(float)
            unequal = np.flatnonzero(left != right)
            if len(unequal):
                index = int(unequal[0])
                mismatch = (
                    f"row={index + 1}, column={name}, "
                    f"parent={left[index]!r}, candidate={right[index]!r}"
                )
                break
        raise WorkerError(
            "fixed-seed extended run failed exact persisted-prefix identity"
            + (f" ({mismatch})" if mismatch else " (CSV bytes differ)")
        )
    return {
        "verified": True,
        "epochs": parent_rows,
        "parent_file_sha256": sha256_file(parent_path),
        "candidate_prefix_file_sha256": sha256_file(candidate_prefix_path),
    }


def runtime_fingerprint(adapter: Any, torch: Any, dgl: Any, device: Any, aliases: Any, preflight: Any) -> dict[str, Any]:
    import scipy
    import sklearn

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", "unknown"),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "dgl": getattr(dgl, "__version__", "unknown"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_capability": (
            list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None
        ),
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_driver": adapter.optional_command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "dgl_legacy_api_aliases": aliases,
        **preflight,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--amendment-id", required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--caps", default="8000,12000")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = Path(args.project).resolve()
    output_root = Path(args.output_root).resolve()
    official_root = output_root / "official_baseline"
    stage_root = Path(args.stage_root).resolve()
    stage_root.mkdir(parents=True, exist_ok=True)
    progress_path = stage_root / "worker_progress.json"
    caps = [int(value) for value in args.caps.split(",") if value.strip()]
    if caps != sorted(set(caps)) or not caps:
        raise WorkerError("--caps must be a nonempty, strictly increasing list")

    adapter = load_adapter(project)
    protocol = read_json(official_root / "protocol.json")
    provenance_manifest = read_json(official_root / "provenance_manifest.json")
    raw = pd.read_csv(official_root / "raw_results.csv")
    mask = (
        pd.to_numeric(raw["data_seed"], errors="coerce").eq(args.data_seed)
        & np.isclose(pd.to_numeric(raw["alpha"], errors="coerce"), args.alpha)
        & pd.to_numeric(raw["train_seed"], errors="coerce").eq(args.train_seed)
    )
    if int(mask.sum()) != 1:
        raise WorkerError(f"target key selects {int(mask.sum())} rows, expected one")
    old_row = raw.loc[mask].iloc[0].to_dict()
    run_id = str(old_row["run_id"])
    run_dir = official_root / "runs" / run_id
    old_run_doc = read_json(run_dir / "run.json")
    old_prediction_manifest = read_json(run_dir / "prediction_manifest.json")
    base_signature = str(protocol.get("run_signature", ""))
    if not base_signature or str(old_row.get("run_signature")) != base_signature:
        raise WorkerError("target row is not bound to the immutable base protocol")

    repo_dir = Path(
        provenance_manifest.get("repository_path", project / "external" / "HINormer")
    ).resolve()
    provenance = adapter.ensure_official_repository(
        repo_dir,
        str(protocol.get("repository_commit", adapter.PINNED_COMMIT)),
        no_clone=True,
        allow_dirty=False,
    )
    torch, dgl = adapter.load_runtime()
    official_class, aliases = adapter.load_official_hinormer_class(
        Path(provenance["model_file"]), dgl, torch
    )
    device = adapter.resolve_device(torch, args.device)
    preflight = adapter.preflight_official_model(official_class, torch, dgl, device)
    runtime = runtime_fingerprint(adapter, torch, dgl, device, aliases, preflight)
    runtime_sha256 = adapter.canonical_hash(runtime)
    if runtime_sha256 != str(protocol.get("runtime_lock_sha256")):
        raise WorkerError(
            "official runtime differs from the immutable base protocol: "
            f"live={runtime_sha256}, expected={protocol.get('runtime_lock_sha256')}"
        )
    if runtime_sha256 != str(old_prediction_manifest.get("environment_sha256")):
        raise WorkerError("target prediction manifest uses another official runtime")

    import yaml

    config_path = Path(args.config).resolve()
    base = adapter.CogBenchConfig.from_yaml(config_path)
    expected = base.updated(seed=args.data_seed, alignment_alpha=args.alpha)
    instance, instance_source = adapter.obtain_run_instance(
        expected,
        instance_root=Path(args.canonical_root).resolve(),
        allow_runtime_generation=False,
    )
    input_sha256 = adapter.model_input_digest(instance)
    if input_sha256 != str(old_row.get("input_instance_sha256")):
        raise WorkerError("canonical target input digest differs from the released row")
    if input_sha256 != str(old_prediction_manifest.get("input_instance_sha256")):
        raise WorkerError("canonical target input digest differs from prediction manifest")

    hp = protocol.get("hyperparameters")
    if not isinstance(hp, dict):
        raise WorkerError("official protocol has no hyperparameters object")
    source_cap = int(float(old_row["source_max_epochs"]))
    parent_cap = int(float(old_row["result_max_epochs"]))
    if source_cap != int(hp["max_epochs"]) or parent_cap >= caps[0]:
        raise WorkerError(
            f"unexpected cap lineage source={source_cap}, parent={parent_cap}, caps={caps}"
        )
    effective_seed = adapter.derive_training_seed(args.data_seed, args.train_seed)
    sequence_seed = int(
        np.random.SeedSequence(
            [args.data_seed, int(round(args.alpha * 1_000_000)), 0x48494E]
        ).generate_state(1)[0]
    )
    parent_history_path = run_dir / "training_history.csv"
    parent_history_records = history_records(parent_history_path)
    decisions: list[dict[str, Any]] = []
    chosen: tuple[int, dict[str, Any], np.ndarray, list[dict[str, float]], dict[str, Any], dict[str, Any]] | None = None
    previous_cap = parent_cap
    previous_history_path = parent_history_path

    for iteration_offset, cap in enumerate(caps, start=1):
        atomic_json(
            {
                "status": "TRAINING",
                "target": [args.data_seed, args.alpha, args.train_seed],
                "cap": cap,
                "caps": caps,
                "test_split_evaluated": False,
                "updated_at": time.time(),
            },
            progress_path,
        )
        print(
            f"[official repair] train target={args.data_seed}/{args.alpha}/{args.train_seed} "
            f"from scratch to cap={cap}",
            flush=True,
        )
        metrics, val_prob, history, artifacts = adapter.train_one(
            instance,
            official_class,
            torch,
            dgl,
            device,
            effective_seed=effective_seed,
            sequence_seed=sequence_seed,
            len_seq=int(hp["len_seq"]),
            num_layers=int(hp["num_layers"]),
            num_gnns=int(hp["num_gnns"]),
            num_heads=int(hp["num_heads"]),
            dropout=float(hp["dropout"]),
            temperature=float(hp["temperature"]),
            beta=float(hp["beta"]),
            learning_rate=float(hp["learning_rate"]),
            weight_decay=float(hp["weight_decay"]),
            max_epochs=cap,
            min_epochs=int(hp["min_epochs"]),
            checkpoint_start_epoch=int(hp["checkpoint_start_epoch"]),
            patience=int(hp["patience"]),
            scheduler_patience=int(hp["scheduler_patience"]),
            scheduler_factor=float(hp["scheduler_factor"]),
            min_learning_rate=float(hp["min_learning_rate"]),
            gradient_clip_norm=float(hp["gradient_clip_norm"]),
            chunk_size=int(hp["chunk_size"]),
            verbose=True,
        )
        cap_dir = stage_root / f"cap_{cap}"
        cap_dir.mkdir(parents=True, exist_ok=True)
        prefix = verify_csv_prefix(
            history,
            previous_history_path,
            run_id,
            cap_dir / f"verified_prefix_{previous_cap}.csv",
        )
        history_path = cap_dir / "training_history.csv"
        history_frame(history, run_id).to_csv(history_path, index=False)
        plateau = adapter.assess_practical_plateau(
            history,
            stopped_early=metrics["stopped_early"],
            epochs_trained=metrics["epochs_trained"],
            max_epochs=cap,
            min_learning_rate=float(hp["min_learning_rate"]),
        )
        metrics.update(plateau)
        capped_or_non_early = bool(
            metrics["checkpoint_at_upper_boundary"] or not metrics["stopped_early"]
        )
        candidate_metadata = {
            **metrics,
            "repair_iteration": int(float(old_row.get("repair_iteration", 1))) + iteration_offset,
            "source_max_epochs": source_cap,
            "parent_max_epochs": previous_cap,
            "result_max_epochs": cap,
            "result_provenance": "repaired_v5",
        }
        plateau_admitted, admission_reason = adapter.validate_extended_repair_plateau(
            history,
            candidate_metadata,
            min_learning_rate=float(hp["min_learning_rate"]),
        )
        plateau_admitted = bool(capped_or_non_early and plateau_admitted)
        resolved = bool(not capped_or_non_early or plateau_admitted)
        decision = {
            "cap": cap,
            "parent_cap": previous_cap,
            "epochs_trained": int(metrics["epochs_trained"]),
            "best_epoch": int(metrics["best_epoch"]),
            "stopped_early": bool(metrics["stopped_early"]),
            "checkpoint_at_upper_boundary": bool(metrics["checkpoint_at_upper_boundary"]),
            "practical_plateau_certified": bool(metrics["practical_plateau_certified"]),
            "practical_plateau_reason": metrics["practical_plateau_reason"],
            "practical_plateau_val_auc_span": metrics["practical_plateau_val_auc_span"],
            "practical_plateau_val_ap_span": metrics["practical_plateau_val_ap_span"],
            "practical_plateau_val_loss_relative_change": metrics[
                "practical_plateau_val_loss_relative_change"
            ],
            "convergence_resolved": resolved,
            "resolution_reason": (
                "extended_repair_early_stop" if not capped_or_non_early else admission_reason
            ),
            "prefix": prefix,
            "history_file_sha256": sha256_file(history_path),
            "test_split_evaluated": False,
        }
        decisions.append(decision)
        atomic_json(
            {
                "schema_version": 1,
                "policy_version": adapter.PRACTICAL_PLATEAU_POLICY_VERSION,
                "metrics_scope": "validation_only",
                "test_metrics_used_for_cap_selection": False,
                "decisions": decisions,
            },
            stage_root / "validation_decisions.json",
        )
        print(
            f"[official repair] cap={cap} resolved={int(resolved)} "
            f"stopped_early={int(bool(metrics['stopped_early']))} "
            f"plateau={int(bool(plateau_admitted))} "
            f"reason={decision['resolution_reason']}",
            flush=True,
        )
        if resolved:
            chosen = (cap, metrics, val_prob, history, artifacts, candidate_metadata)
            break
        previous_cap = cap
        previous_history_path = history_path

    if chosen is None:
        atomic_json(
            {
                "status": "UNRESOLVED_AT_TERMINAL_CAP",
                "target": [args.data_seed, args.alpha, args.train_seed],
                "caps": caps,
                "test_split_evaluated": False,
                "decisions_path": str(stage_root / "validation_decisions.json"),
            },
            progress_path,
        )
        raise WorkerError(
            f"target remains unresolved at terminal cap {caps[-1]}; no test metric was read"
        )

    cap, metrics, val_prob, history, artifacts, candidate_metadata = chosen
    atomic_json(
        {
            "status": "FINAL_TEST_EVALUATION",
            "target": [args.data_seed, args.alpha, args.train_seed],
            "selected_cap": cap,
            "test_split_evaluated": False,
        },
        progress_path,
    )
    test_metrics, test_prob, test_health = adapter.finalize_test_evaluation(
        instance, artifacts, torch, chunk_size=int(hp["chunk_size"])
    )
    metrics.update(test_metrics)
    metrics.update(
        {
            "implementation_is_diagnostic_only": 0.0,
            "implementation_is_style_reimplementation": 0.0,
            "implementation_is_official_source": 1.0,
        }
    )
    capped_or_non_early = bool(
        metrics["checkpoint_at_upper_boundary"] or not metrics["stopped_early"]
    )
    plateau_admitted = bool(
        capped_or_non_early
        and float(metrics.get("practical_plateau_certified", 0.0)) == 1.0
    )
    metrics.update(
        {
            "practical_plateau_admitted_after_extended_repair": float(plateau_admitted),
            "convergence_resolved": 1.0,
            "convergence_resolution_reason": (
                "valid_extended_repair_plateau"
                if plateau_admitted
                else "extended_repair_early_stop"
            ),
        }
    )

    selected_dir = stage_root / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_history_path = selected_dir / "training_history.csv"
    history_frame(history, run_id).to_csv(selected_history_path, index=False)
    predictions_path = selected_dir / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        val_prob=val_prob.astype(np.float32),
        test_prob=test_prob.astype(np.float32),
        val_idx=instance.splits["val"],
        test_idx=instance.splits["test"],
    )

    final_prediction_path = run_dir / "predictions.npz"
    prediction_manifest = dict(old_prediction_manifest)
    prediction_manifest.update(
        {
            "run_id": run_id,
            "data_seed": args.data_seed,
            "alpha": args.alpha,
            "train_seed": args.train_seed,
            "effective_train_seed": effective_seed,
            "run_signature": base_signature,
            "environment_sha256": runtime_sha256,
            "input_instance_sha256": input_sha256,
            "val_probability_sha256": hashlib.sha256(
                np.ascontiguousarray(val_prob.astype(np.float32)).tobytes()
            ).hexdigest(),
            "test_probability_sha256": hashlib.sha256(
                np.ascontiguousarray(test_prob.astype(np.float32)).tobytes()
            ).hexdigest(),
            "predictions_path": str(final_prediction_path),
            "contains_labels": False,
            "protocol_amendment_id": args.amendment_id,
        }
    )
    atomic_json(prediction_manifest, selected_dir / "prediction_manifest.json")

    prior_iteration = int(float(old_row.get("repair_iteration", 1)))
    selected_parent_cap = int(decisions[-1]["parent_cap"])
    parent_records = history_records(
        parent_history_path
        if selected_parent_cap == parent_cap
        else stage_root / f"cap_{selected_parent_cap}" / "training_history.csv"
    )
    run_record = dict(old_row)
    run_record.update(metrics)
    run_record.update(
        {
            "run_id": run_id,
            "run_signature": base_signature,
            "seed": args.data_seed,
            "data_seed": args.data_seed,
            "train_seed": args.train_seed,
            "effective_train_seed": effective_seed,
            "sequence_seed": sequence_seed,
            "alpha": args.alpha,
            "repair_iteration": prior_iteration + len(decisions),
            "result_provenance": "repaired_v5",
            "repair_prefix_verified": 1.0,
            "initial_history_sha256": adapter.canonical_hash(parent_records),
            "repair_history_prefix_sha256": adapter.canonical_hash(parent_records),
            "source_max_epochs": source_cap,
            "parent_max_epochs": selected_parent_cap,
            "result_max_epochs": cap,
            "parent_run": f"extended_repair_cap_{selected_parent_cap}",
            "initial_best_epoch": float(old_row["best_epoch"]),
            "initial_checkpoint_at_boundary": float(old_row["checkpoint_at_boundary"]),
            "protocol_amendment_id": args.amendment_id,
            "base_run_signature": base_signature,
        }
    )
    health = dict(old_run_doc.get("health") or {})
    health.update(artifacts["health"])
    health.update(test_health)
    health.update(
        {
            key: metrics[key]
            for key in metrics
            if key.startswith("practical_plateau_")
        }
    )
    health.update(
        {
            "convergence_resolved": 1,
            "convergence_resolution_reason": run_record[
                "convergence_resolution_reason"
            ],
        }
    )
    atomic_json({"result": run_record, "health": health}, selected_dir / "run.json")
    if (run_dir / "best_checkpoint.pt").is_file():
        torch.save(
            {
                "model_state_dict": artifacts["state_dict"],
                "run_id": run_id,
                "run_signature": base_signature,
                "repository_commit": provenance["repository_commit"],
                "model_file_sha256": provenance["model_file_sha256"],
                "protocol_amendment_id": args.amendment_id,
            },
            selected_dir / "best_checkpoint.pt",
        )

    result = {
        "schema_version": 1,
        "status": "PASS",
        "amendment_id": args.amendment_id,
        "target": {
            "run_id": run_id,
            "data_seed": args.data_seed,
            "alpha": args.alpha,
            "train_seed": args.train_seed,
        },
        "base_run_signature": base_signature,
        "runtime_sha256": runtime_sha256,
        "canonical_input_sha256": input_sha256,
        "source_max_epochs": source_cap,
        "parent_max_epochs": selected_parent_cap,
        "selected_result_max_epochs": cap,
        "validation_only_cap_selection": True,
        "test_metrics_used_for_cap_selection": False,
        "test_evaluation_count": 1,
        "prefix_verifications": [item["prefix"] for item in decisions],
        "validation_decisions_path": str(stage_root / "validation_decisions.json"),
        "selected_artifacts": {
            name: {
                "path": str(selected_dir / name),
                "sha256": sha256_file(selected_dir / name),
            }
            for name in (
                "training_history.csv",
                "predictions.npz",
                "prediction_manifest.json",
                "run.json",
            )
        },
    }
    if (selected_dir / "best_checkpoint.pt").is_file():
        result["selected_artifacts"]["best_checkpoint.pt"] = {
            "path": str(selected_dir / "best_checkpoint.pt"),
            "sha256": sha256_file(selected_dir / "best_checkpoint.pt"),
        }
    atomic_json(result, stage_root / "worker_result.json")
    atomic_json(
        {
            "status": "PASS",
            "target": [args.data_seed, args.alpha, args.train_seed],
            "selected_cap": cap,
            "test_split_evaluated": True,
            "test_evaluation_count": 1,
            "result": str(stage_root / "worker_result.json"),
        },
        progress_path,
    )
    print(
        f"[official repair] PASS selected_cap={cap}; isolated candidate is ready",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except WorkerError as exc:
        raise SystemExit(f"OFFICIAL REPAIR WORKER ERROR: {exc}") from exc
