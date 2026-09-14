#!/usr/bin/env python3
"""Read-only validation of the amended official HINormer journal."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def load_adapter(project: Path) -> Any:
    path = project / "scripts" / "run_official_hinormer.py"
    spec = importlib.util.spec_from_file_location("cogbench_official_adapter_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    project = Path(args.project).resolve()
    root = Path(args.output_root).resolve() / "official_baseline"
    protocol = read_json(root / "protocol.json")
    provenance = read_json(root / "provenance_manifest.json")
    adapter = load_adapter(project)
    repo = adapter.ensure_official_repository(
        Path(provenance["repository_path"]),
        str(protocol["repository_commit"]),
        no_clone=True,
        allow_dirty=False,
    )
    torch, dgl = adapter.load_runtime()
    official_class, aliases = adapter.load_official_hinormer_class(
        Path(repo["model_file"]), dgl, torch
    )
    device = adapter.resolve_device(torch, str(provenance["environment"]["device"]))
    preflight = adapter.preflight_official_model(official_class, torch, dgl, device)
    runtime = runtime_fingerprint(adapter, torch, dgl, device, aliases, preflight)
    runtime_hash = adapter.canonical_hash(runtime)
    if runtime_hash != protocol.get("runtime_lock_sha256"):
        raise RuntimeError("live official runtime no longer matches protocol")

    frame = pd.read_csv(root / "raw_results.csv")
    rows = frame.to_dict(orient="records")
    retained = adapter.validate_resume_rows(
        rows,
        root,
        run_signature=str(protocol["run_signature"]),
        runtime_sha256=runtime_hash,
        min_learning_rate=float(protocol["hyperparameters"]["min_learning_rate"]),
        data_seeds=[int(value) for value in protocol["data_seeds"]],
        alphas=[float(value) for value in protocol["alphas"]],
        train_seeds=[int(value) for value in protocol["train_seeds"]],
        official_repository_adapter=True,
    )
    if len(retained) != len(rows) or len(rows) != 150:
        raise RuntimeError(
            f"resume validator retained {len(retained)}/{len(rows)} rows; expected 150/150"
        )
    capped = (
        pd.to_numeric(frame["checkpoint_at_upper_boundary"], errors="coerce").eq(1)
        | pd.to_numeric(frame["stopped_early"], errors="coerce").eq(0)
    )
    for index in frame.index[capped]:
        row = frame.loc[index]
        source = int(float(row["source_max_epochs"]))
        parent = int(float(row["parent_max_epochs"]))
        result = int(float(row["result_max_epochs"]))
        if not (result > source and result > parent):
            raise RuntimeError(
                f"row {index} has invalid extended-cap lineage: source={source}, "
                f"parent={parent}, result={result}"
            )
        if float(row["practical_plateau_admitted_after_extended_repair"]) != 1.0:
            raise RuntimeError(f"row {index} remains capped/non-early without an admitted plateau")
    print(
        f"OFFICIAL_RESUME_VALIDATION=PASS retained={len(retained)} "
        f"capped_certified={int(capped.sum())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
