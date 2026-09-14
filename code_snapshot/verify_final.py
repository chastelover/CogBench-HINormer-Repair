#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    marker_path = root / ".completed" / "official_hinormer_extension.json"
    report_path = root / "final_release_report.json"
    if not marker_path.is_file():
        raise SystemExit(f"FINAL_VERIFY=FAIL missing extension marker: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker_body = {key: value for key, value in marker.items() if key != "marker_sha256"}
    if canonical_hash(marker_body) != marker.get("marker_sha256"):
        raise SystemExit("FINAL_VERIFY=FAIL extension marker self-hash mismatch")
    receipt_path = Path(marker["receipt_path"])
    if not receipt_path.is_file():
        raise SystemExit(f"FINAL_VERIFY=FAIL missing amendment receipt: {receipt_path}")
    receipt_file_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    if receipt_file_hash != marker.get("receipt_file_sha256"):
        raise SystemExit("FINAL_VERIFY=FAIL amendment receipt file hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_body = {key: value for key, value in receipt.items() if key != "report_sha256"}
    if canonical_hash(receipt_body) != receipt.get("report_sha256"):
        raise SystemExit("FINAL_VERIFY=FAIL amendment receipt self-hash mismatch")
    if receipt.get("transaction_committed") is not True:
        raise SystemExit("FINAL_VERIFY=FAIL repair transaction is not committed")
    if not report_path.is_file():
        raise SystemExit(f"FINAL_VERIFY=WAIT final report is not created: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not (
        report.get("status") == "PASS"
        and report.get("blocker_count") == 0
        and report.get("blockers") == []
        and report.get("claim_ready") is True
        and report.get("claim_failure_count") == 0
        and report.get("claim_failures") == []
    ):
        raise SystemExit(
            "FINAL_VERIFY=FAIL strict final gate is not clean: "
            f"status={report.get('status')} blockers={report.get('blocker_count')} "
            f"claim_ready={report.get('claim_ready')} "
            f"claim_failures={report.get('claim_failure_count')}"
        )
    print(
        "FINAL_VERIFY=PASS "
        f"selected_cap={receipt.get('selected_result_max_epochs')} "
        f"metadata_backfills={receipt.get('metadata_backfill_count')} "
        "strict_gate=PASS blockers=0 claim_ready=True claim_failures=0"
    )


if __name__ == "__main__":
    main()
