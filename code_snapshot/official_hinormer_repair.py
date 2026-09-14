#!/usr/bin/env python3
"""Transactional controller for the single unresolved official HINormer run."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


class RepairError(RuntimeError):
    pass


EXPECTED_SOURCE_HASHES = {
    "scripts/run_official_hinormer.py": "4cf78bd2c025b64f8456557e83ba55b3c9f439340e90e55a40483381a46be64d",
    "cogbench/convergence.py": "dd86c9965a832ab71ff5b7757b3724432cbf2a1a32a5fe70b2793c9bd68c0409",
    "scripts/check_final_release.py": "351169a1ce172b7d793d490f3d1559c2d3807f81ca42a0aa543fa68d44773e40",
}
EXPECTED_CONFIG_SHA256 = "0660789a0ad1726bfb48868a368e4d4decb057d3b5c76eb5a5f2bfa9151f8eff"


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RepairError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RepairError(f"expected JSON object: {path}")
    return value


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(safe(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def lock_path(output_root: Path) -> Path:
    identity = hashlib.sha256(str(output_root.resolve()).encode()).hexdigest()[:20]
    return Path(tempfile.gettempdir()) / f"cogbench-v5.2-autorepair-{identity}.lock"


@contextlib.contextmanager
def exclusive_lock(output_root: Path) -> Iterator[Path]:
    path = lock_path(output_root)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RepairError("O_NOFOLLOW is required for the shared controller lock")
    fd = os.open(path, flags | os.O_NOFOLLOW, 0o600)
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_nlink != 1:
        os.close(fd)
        raise RepairError(f"unsafe shared lock: {path}")
    handle = os.fdopen(fd, "r+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RepairError(f"another CogBench controller is active ({path})") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"official-hinormer-extension pid={os.getpid()}\n")
        handle.flush()
        yield path
    finally:
        handle.close()


def ancestor_pids() -> set[int]:
    result = {os.getpid()}
    current = os.getppid()
    while current > 1 and current not in result:
        result.add(current)
        try:
            status_text = (Path("/proc") / str(current) / "status").read_text()
            parent_line = next(line for line in status_text.splitlines() if line.startswith("PPid:"))
            current = int(parent_line.split()[1])
        except Exception:
            break
    return result


def live_writers(project: Path, output_root: Path) -> list[dict[str, Any]]:
    tokens = (
        "cogbench_autorepair.py",
        "run_final_suite.sh",
        "launch_final_suite.sh",
        "run_official_hinormer.py",
        "repair_auxiliary_runs.py",
        "repair_boundary_runs.py",
    )
    ignored = ancestor_pids()
    hits: list[dict[str, Any]] = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in ignored:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if not command or not any(token in command for token in tokens):
            continue
        if str(project) in command or str(output_root) in command:
            hits.append({"pid": int(entry.name), "command": command})
    return hits


def file_snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in sorted(paths) if path.is_file()}


def immutable_run_artifacts(official_root: Path, exclude_run_id: str) -> dict[str, str]:
    paths: list[Path] = []
    for run_dir in sorted((official_root / "runs").iterdir()):
        if not run_dir.is_dir() or run_dir.name == exclude_run_id:
            continue
        for name in ("training_history.csv", "training_history_initial_boundary.csv", "predictions.npz", "prediction_manifest.json", "best_checkpoint.pt"):
            path = run_dir / name
            if path.is_file():
                paths.append(path)
    return file_snapshot(paths)


def unresolved_mask(frame: pd.DataFrame) -> pd.Series:
    capped = (
        pd.to_numeric(frame["checkpoint_at_upper_boundary"], errors="coerce").eq(1)
        | pd.to_numeric(frame["stopped_early"], errors="coerce").eq(0)
    )
    admitted = (
        pd.to_numeric(
            frame["practical_plateau_admitted_after_extended_repair"], errors="coerce"
        ).eq(1)
        & frame["result_provenance"].astype(str).eq("repaired_v5")
        & pd.to_numeric(frame["repair_iteration"], errors="coerce").ge(1)
    )
    return capped & ~admitted


def certified_mask(frame: pd.DataFrame) -> pd.Series:
    capped = (
        pd.to_numeric(frame["checkpoint_at_upper_boundary"], errors="coerce").eq(1)
        | pd.to_numeric(frame["stopped_early"], errors="coerce").eq(0)
    )
    admitted = pd.to_numeric(
        frame["practical_plateau_admitted_after_extended_repair"], errors="coerce"
    ).eq(1)
    return capped & admitted & frame["result_provenance"].astype(str).eq("repaired_v5")


def recover_interrupted(official_root: Path) -> list[str]:
    transaction_root = official_root / ".official_extension_transactions"
    recovered: list[str] = []
    if not transaction_root.is_dir():
        return recovered
    for state_path in sorted(transaction_root.glob("*/state.json")):
        state = read_json(state_path)
        if state.get("status") in {"COMMITTED", "ROLLED_BACK"}:
            continue
        for item in reversed(state.get("mutations", [])):
            target = Path(item["target"])
            backup = Path(item["backup"])
            if item.get("existed"):
                if not backup.is_file():
                    raise RepairError(f"interrupted transaction has no backup for {target}")
                atomic_copy(backup, target)
            elif target.exists():
                target.unlink()
        state["status"] = "ROLLED_BACK"
        state["rollback_reason"] = "automatic recovery before a new invocation"
        state["rolled_back_at"] = time.time()
        atomic_json(state, state_path)
        recovered.append(str(state_path.parent))
    return recovered


def prepare_mutation(transaction: Path, target: Path, source: Path, ordinal: int) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise RepairError(f"staged mutation source is missing or linked: {source}")
    backup = transaction / "backups" / f"{ordinal:04d}.bin"
    existed = target.is_file()
    before = sha256_file(target) if existed else None
    if existed:
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
    return {
        "target": str(target),
        "source": str(source),
        "backup": str(backup),
        "existed": existed,
        "before_sha256": before,
        "after_sha256": sha256_file(source),
    }


def rollback(state: dict[str, Any], state_path: Path, reason: str) -> None:
    for item in reversed(state.get("mutations", [])):
        target = Path(item["target"])
        backup = Path(item["backup"])
        if item.get("existed"):
            atomic_copy(backup, target)
        elif target.exists():
            target.unlink()
    state["status"] = "ROLLED_BACK"
    state["rollback_reason"] = reason
    state["rolled_back_at"] = time.time()
    atomic_json(state, state_path)


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="/CogBench/CogBench_Final_Experiments_V5_2")
    parser.add_argument("--output-root", default="outputs/final_paper_v5_2")
    parser.add_argument("--canonical-root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--official-python",
        default="/usr/local/miniconda3/envs/cogbench-hinormer-cpu/bin/python",
    )
    parser.add_argument("--official-device", default="cpu")
    parser.add_argument("--caps", default="8000,12000")
    parser.add_argument("--expected-data-seed", type=int, default=269)
    parser.add_argument("--expected-alpha", type=float, default=0.9)
    parser.add_argument("--expected-train-seed", type=int, default=1003)
    parser.add_argument("--worker", default=str(package / "official_hinormer_worker.py"))
    parser.add_argument("--validator", default=str(package / "official_hinormer_validate.py"))
    parser.add_argument("--session-root", default=None)
    parser.add_argument("--test-only-skip-source-audit", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = Path(args.project).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (project / output_root).resolve()
    official_root = output_root / "official_baseline"
    config = Path(args.config).resolve() if args.config else project / "configs" / "final_paper.yaml"
    canonical_root = (
        Path(args.canonical_root).resolve()
        if args.canonical_root
        else output_root / "canonical_instances"
    )
    official_python = Path(args.official_python).resolve()
    worker = Path(args.worker).resolve()
    validator = Path(args.validator).resolve()
    for path, label in (
        (project, "project"),
        (output_root, "output root"),
        (official_root, "official result root"),
        (canonical_root, "canonical instance root"),
    ):
        if not path.is_dir():
            raise RepairError(f"{label} is missing: {path}")
    for path, label in ((config, "config"), (worker, "worker"), (validator, "validator")):
        if not path.is_file() or path.is_symlink():
            raise RepairError(f"{label} is missing or linked: {path}")
    if not official_python.is_file() or not os.access(official_python, os.X_OK):
        raise RepairError(f"official Python is not executable: {official_python}")
    if args.test_only_skip_source_audit:
        if os.environ.get("COGBENCH_TEST_MODE") != "1":
            raise RepairError("the source-audit bypass is available only under COGBENCH_TEST_MODE=1")
    else:
        observed_source_hashes = {
            relative: sha256_file(project / relative)
            for relative in EXPECTED_SOURCE_HASHES
            if (project / relative).is_file()
        }
        if observed_source_hashes != EXPECTED_SOURCE_HASHES:
            raise RepairError(
                "project source differs from the supported plateau-v2 release: "
                f"observed={observed_source_hashes}, expected={EXPECTED_SOURCE_HASHES}"
            )
        if sha256_file(config) != EXPECTED_CONFIG_SHA256:
            raise RepairError("final_paper.yaml differs from the supported frozen configuration")
    caps = [int(value) for value in args.caps.split(",") if value.strip()]
    if caps != [8000, 12000]:
        raise RepairError("this frozen recovery release permits only --caps 8000,12000")

    package_root = Path(__file__).resolve().parent
    session_root = (
        Path(args.session_root).resolve()
        if args.session_root
        else package_root / "runs" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    if session_root.exists():
        raise RepairError(f"session root already exists: {session_root}")
    session_root.mkdir(parents=True)
    state_path = session_root / "state.json"
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "PREFLIGHT",
        "project": str(project),
        "output_root": str(output_root),
        "official_python": str(official_python),
        "caps": caps,
        "started_at": time.time(),
    }
    atomic_json(state, state_path)

    with exclusive_lock(output_root) as shared_lock:
        state["shared_lock"] = str(shared_lock)
        recovered = recover_interrupted(official_root)
        state["recovered_transactions"] = recovered
        writers = live_writers(project, output_root)
        if writers:
            raise RepairError(f"live CogBench writers detected: {writers}")

        migration = read_json(output_root / "plateau_v2_migration_report.json")
        if (
            migration.get("status") != "PASS"
            or migration.get("policy_version_after") != "validation_practical_plateau_v2"
            or migration.get("transaction_committed") is not True
        ):
            raise RepairError("validated plateau-v2 migration receipt is not PASS")
        protocol = read_json(official_root / "protocol.json")
        base_signature = str(protocol.get("run_signature", ""))
        if not base_signature:
            raise RepairError("official base protocol has no run_signature")
        raw_path = official_root / "raw_results.csv"
        raw = pd.read_csv(raw_path)
        expected_rows = (
            len(protocol.get("alphas", []))
            * len(protocol.get("data_seeds", []))
            * len(protocol.get("train_seeds", []))
        )
        if expected_rows != 150 or len(raw) != expected_rows:
            raise RepairError(f"official grid is incomplete: rows={len(raw)}, expected={expected_rows}")
        if raw[["data_seed", "alpha", "train_seed"]].duplicated().any():
            raise RepairError("official raw results contain duplicate run keys")
        unresolved = raw.loc[unresolved_mask(raw)]
        existing_marker_path = output_root / ".completed" / "official_hinormer_extension.json"
        if existing_marker_path.is_file():
            existing_marker = read_json(existing_marker_path)
            marker_body = {
                key: value
                for key, value in existing_marker.items()
                if key != "marker_sha256"
            }
            receipt_path = Path(str(existing_marker.get("receipt_path", "")))
            if (
                existing_marker.get("status") != "PASS"
                or canonical_hash(marker_body) != existing_marker.get("marker_sha256")
                or not receipt_path.is_file()
                or sha256_file(receipt_path) != existing_marker.get("receipt_file_sha256")
            ):
                raise RepairError("existing official extension marker is invalid")
            if len(unresolved) != 0:
                raise RepairError(
                    "a committed official extension marker exists but unresolved rows remain"
                )
            state.update(
                {
                    "status": "PASS",
                    "already_committed": True,
                    "marker": str(existing_marker_path),
                    "receipt": str(receipt_path),
                    "finished_at": time.time(),
                }
            )
            atomic_json(state, state_path)
            print("OFFICIAL_HINORMER_EXTENSION=ALREADY_COMMITTED", flush=True)
            return
        if len(unresolved) != 1:
            raise RepairError(f"expected exactly one unresolved official run, found {len(unresolved)}")
        target = unresolved.iloc[0]
        target_tuple = (
            int(target["data_seed"]),
            float(target["alpha"]),
            int(target["train_seed"]),
        )
        expected_target = (
            args.expected_data_seed,
            args.expected_alpha,
            args.expected_train_seed,
        )
        if target_tuple != expected_target:
            raise RepairError(f"unresolved target changed: observed={target_tuple}, expected={expected_target}")
        if int(float(target.get("degenerate_predictions", 1))) != 0:
            raise RepairError("target prediction is degenerate; cap extension is not an admissible repair")
        if str(target.get("practical_plateau_policy_version")) != "validation_practical_plateau_v2":
            raise RepairError("target is not assessed under plateau-v2")

        run_id = str(target["run_id"])
        immutable_before = immutable_run_artifacts(official_root, run_id)
        amendment_id = f"official-hinormer-extension-{uuid.uuid4().hex}"
        stage_root = session_root / "worker"
        state.update(
            {
                "status": "WORKER_RUNNING",
                "target": {
                    "run_id": run_id,
                    "data_seed": target_tuple[0],
                    "alpha": target_tuple[1],
                    "train_seed": target_tuple[2],
                },
                "amendment_id": amendment_id,
                "base_run_signature": base_signature,
            }
        )
        atomic_json(state, state_path)
        command = [
            str(official_python),
            "-u",
            str(worker),
            "--project",
            str(project),
            "--output-root",
            str(output_root),
            "--canonical-root",
            str(canonical_root),
            "--config",
            str(config),
            "--stage-root",
            str(stage_root),
            "--amendment-id",
            amendment_id,
            "--data-seed",
            str(target_tuple[0]),
            "--alpha",
            str(target_tuple[1]),
            "--train-seed",
            str(target_tuple[2]),
            "--caps",
            args.caps,
            "--device",
            args.official_device,
        ]
        state["worker_command"] = command
        atomic_json(state, state_path)
        completed = subprocess.run(command, cwd=project)
        state["worker_exit_code"] = completed.returncode
        atomic_json(state, state_path)
        if completed.returncode != 0:
            raise RepairError(
                f"isolated official worker exited {completed.returncode}; live results were not changed"
            )
        worker_result = read_json(stage_root / "worker_result.json")
        if (
            worker_result.get("status") != "PASS"
            or worker_result.get("test_metrics_used_for_cap_selection") is not False
            or worker_result.get("test_evaluation_count") != 1
            or worker_result.get("base_run_signature") != base_signature
        ):
            raise RepairError("worker result does not satisfy the frozen repair contract")

        amendment_core = {
            "schema_version": 1,
            "amendment_type": "validation_only_fixed_seed_cap_extension",
            "amendment_id": amendment_id,
            "base_run_signature": base_signature,
            "target": worker_result["target"],
            "fixed_budget_ladder": caps,
            "selected_result_max_epochs": worker_result["selected_result_max_epochs"],
            "source_max_epochs": worker_result["source_max_epochs"],
            "parent_max_epochs": worker_result["parent_max_epochs"],
            "runtime_sha256": worker_result["runtime_sha256"],
            "canonical_input_sha256": worker_result["canonical_input_sha256"],
            "validation_only_cap_selection": True,
            "test_metrics_used_for_cap_selection": False,
            "test_evaluation_count": 1,
            "prefix_verifications": worker_result["prefix_verifications"],
            "official_adapter_sha256": sha256_file(project / "scripts" / "run_official_hinormer.py"),
            "plateau_policy_version": "validation_practical_plateau_v2",
        }
        amendment_signature = canonical_hash(amendment_core)
        effective_signature = canonical_hash(
            {"base_run_signature": base_signature, "amendment_signature": amendment_signature}
        )

        selected = stage_root / "selected"
        candidate_run_path = selected / "run.json"
        candidate_manifest_path = selected / "prediction_manifest.json"
        candidate_run = read_json(candidate_run_path)
        candidate_manifest = read_json(candidate_manifest_path)
        for document in (candidate_run["result"], candidate_manifest):
            document["protocol_amendment_id"] = amendment_id
            document["protocol_amendment_signature"] = amendment_signature
            document["effective_run_signature"] = effective_signature
            document["base_run_signature"] = base_signature
        atomic_json(candidate_run, candidate_run_path)
        atomic_json(candidate_manifest, candidate_manifest_path)
        worker_result["amendment_signature"] = amendment_signature
        worker_result["effective_run_signature"] = effective_signature
        for name in ("run.json", "prediction_manifest.json"):
            worker_result["selected_artifacts"][name]["sha256"] = sha256_file(
                selected / name
            )
        atomic_json(worker_result, stage_root / "worker_result.json")

        staged = session_root / "staged"
        staged.mkdir()
        staged_raw = raw.copy()
        target_mask = (
            pd.to_numeric(staged_raw["data_seed"], errors="coerce").eq(target_tuple[0])
            & np.isclose(pd.to_numeric(staged_raw["alpha"], errors="coerce"), target_tuple[1])
            & pd.to_numeric(staged_raw["train_seed"], errors="coerce").eq(target_tuple[2])
        )
        candidate_row = candidate_run["result"]
        for field, value in candidate_row.items():
            if field not in staged_raw.columns:
                staged_raw[field] = np.nan
            staged_raw.loc[target_mask, field] = value

        metadata_backfills: list[dict[str, Any]] = []
        staged_run_json: list[tuple[Path, Path]] = []
        for index in staged_raw.index[certified_mask(staged_raw) & ~target_mask]:
            current_parent = staged_raw.at[index, "parent_max_epochs"] if "parent_max_epochs" in staged_raw else np.nan
            if not pd.isna(current_parent):
                continue
            iteration = int(float(staged_raw.at[index, "repair_iteration"]))
            source_cap = int(float(staged_raw.at[index, "source_max_epochs"]))
            result_cap = int(float(staged_raw.at[index, "result_max_epochs"]))
            if iteration != 1 or result_cap <= source_cap:
                raise RepairError(
                    "cannot unambiguously backfill parent_max_epochs for an existing certified row"
                )
            other_run_id = str(staged_raw.at[index, "run_id"])
            other_live = official_root / "runs" / other_run_id / "run.json"
            other_doc = read_json(other_live)
            other_result = other_doc.get("result")
            if not isinstance(other_result, dict):
                raise RepairError(f"existing certified run has malformed run.json: {other_run_id}")
            if int(float(other_result.get("result_max_epochs", -1))) != result_cap:
                raise RepairError(f"existing certified row/run.json cap mismatch: {other_run_id}")
            staged_raw.at[index, "parent_max_epochs"] = source_cap
            other_result["parent_max_epochs"] = source_cap
            other_stage = staged / f"backfill_{other_run_id}.json"
            atomic_json(other_doc, other_stage)
            staged_run_json.append((other_stage, other_live))
            metadata_backfills.append(
                {
                    "run_id": other_run_id,
                    "parent_max_epochs": source_cap,
                    "result_max_epochs": result_cap,
                    "training_history_sha256": sha256_file(
                        official_root / "runs" / other_run_id / "training_history.csv"
                    ),
                    "metrics_changed": False,
                    "predictions_changed": False,
                }
            )

        staged_raw_path = staged / "raw_results.csv"
        staged_raw.to_csv(staged_raw_path, index=False)
        target_live_dir = official_root / "runs" / run_id
        source_targets: list[tuple[Path, Path]] = [
            (staged_raw_path, raw_path),
            (selected / "training_history.csv", target_live_dir / "training_history.csv"),
            (selected / "predictions.npz", target_live_dir / "predictions.npz"),
            (candidate_manifest_path, target_live_dir / "prediction_manifest.json"),
            (candidate_run_path, target_live_dir / "run.json"),
        ]
        if (selected / "best_checkpoint.pt").is_file():
            source_targets.append((selected / "best_checkpoint.pt", target_live_dir / "best_checkpoint.pt"))
        source_targets.extend(staged_run_json)

        transaction = official_root / ".official_extension_transactions" / amendment_id
        transaction.mkdir(parents=True)
        mutations = [
            prepare_mutation(transaction, target_path, source_path, ordinal)
            for ordinal, (source_path, target_path) in enumerate(source_targets)
        ]
        transaction_state = {
            "schema_version": 1,
            "transaction_id": amendment_id,
            "status": "PREPARED",
            "mutations": mutations,
            "created_at": time.time(),
        }
        transaction_state_path = transaction / "state.json"
        atomic_json(transaction_state, transaction_state_path)
        state["status"] = "COMMITTING"
        state["transaction"] = str(transaction)
        atomic_json(state, state_path)
        try:
            transaction_state["status"] = "APPLYING"
            atomic_json(transaction_state, transaction_state_path)
            for item in mutations:
                atomic_copy(Path(item["source"]), Path(item["target"]))
                if sha256_file(Path(item["target"])) != item["after_sha256"]:
                    raise RepairError(f"post-write hash mismatch: {item['target']}")

            immutable_after = immutable_run_artifacts(official_root, run_id)
            if immutable_after != immutable_before:
                raise RepairError("a non-target training/prediction artifact changed")
            validation_command = [
                str(official_python),
                "-u",
                str(validator),
                "--project",
                str(project),
                "--output-root",
                str(output_root),
            ]
            validation = subprocess.run(validation_command, cwd=project)
            if validation.returncode != 0:
                raise RepairError(f"post-commit official resume validation exited {validation.returncode}")
        except Exception as exc:
            rollback(transaction_state, transaction_state_path, f"{type(exc).__name__}: {exc}")
            raise

        transaction_state["status"] = "COMMITTED"
        transaction_state["committed_at"] = time.time()
        atomic_json(transaction_state, transaction_state_path)
        receipt = {
            **amendment_core,
            "amendment_signature": amendment_signature,
            "effective_run_signature": effective_signature,
            "transaction_committed": True,
            "transaction_state": str(transaction_state_path),
            "metadata_backfills": metadata_backfills,
            "metadata_backfill_count": len(metadata_backfills),
            "non_target_training_and_prediction_artifacts_unchanged": True,
            "non_target_immutable_artifact_count": len(immutable_before),
            "worker_result_path": str(stage_root / "worker_result.json"),
            "worker_result_sha256": sha256_file(stage_root / "worker_result.json"),
            "mutations": [
                {
                    "target": item["target"],
                    "before_sha256": item["before_sha256"],
                    "after_sha256": item["after_sha256"],
                }
                for item in mutations
            ],
            "created_at": time.time(),
        }
        receipt["report_sha256"] = canonical_hash(receipt)
        receipt_path = official_root / "protocol_amendments" / f"{amendment_id}.json"
        marker_path = output_root / ".completed" / "official_hinormer_extension.json"
        atomic_json(receipt, receipt_path)
        marker = {
            "schema_version": 1,
            "status": "PASS",
            "amendment_id": amendment_id,
            "amendment_signature": amendment_signature,
            "effective_run_signature": effective_signature,
            "receipt_path": str(receipt_path),
            "receipt_file_sha256": sha256_file(receipt_path),
            "report_sha256": receipt["report_sha256"],
        }
        marker["marker_sha256"] = canonical_hash(marker)
        atomic_json(marker, marker_path)
        state.update(
            {
                "status": "PASS",
                "selected_result_max_epochs": worker_result["selected_result_max_epochs"],
                "receipt": str(receipt_path),
                "marker": str(marker_path),
                "metadata_backfill_count": len(metadata_backfills),
                "finished_at": time.time(),
            }
        )
        atomic_json(state, state_path)
        print(
            "OFFICIAL_HINORMER_EXTENSION=PASS "
            f"selected_cap={worker_result['selected_result_max_epochs']} "
            f"metadata_backfills={len(metadata_backfills)}",
            flush=True,
        )
        print(f"STATE={state_path}", flush=True)
        print(f"RECEIPT={receipt_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        package_root = Path(__file__).resolve().parent
        states = sorted(
            (package_root / "runs").glob("*/state.json"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if states:
            try:
                failed_state = read_json(states[-1])
                if failed_state.get("status") != "PASS":
                    failed_state["status"] = "FAILED"
                    failed_state["error"] = f"{type(exc).__name__}: {exc}"
                    failed_state["finished_at"] = time.time()
                    atomic_json(failed_state, states[-1])
            except Exception:
                pass
        raise SystemExit(f"OFFICIAL HINORMER REPAIR ERROR: {exc}") from exc
