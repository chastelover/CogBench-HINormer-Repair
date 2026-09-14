from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


controller = load("repair_controller", ROOT / "official_hinormer_repair.py")
worker = load("repair_worker", ROOT / "official_hinormer_worker.py")


class PrefixTests(unittest.TestCase):
    def records(self):
        return [
            {
                "epoch": float(i),
                "train_loss": 1.0 / i,
                "gradient_norm_before_clip": 0.1 * i,
                "val_loss": 0.5 / i,
                "val_auc": 0.7 + i / 1000,
                "val_ap": 0.6 + i / 1000,
                "learning_rate": 1e-5,
                "checkpoint_eligible": 1.0,
            }
            for i in range(1, 6)
        ]

    def test_exact_persisted_prefix_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_id = "seed_269__alpha_0p9__train_1003"
            parent = root / "parent.csv"
            worker.history_frame(self.records()[:3], run_id).to_csv(parent, index=False)
            result = worker.verify_csv_prefix(
                self.records(), parent, run_id, root / "candidate_prefix.csv"
            )
            self.assertTrue(result["verified"])
            self.assertEqual(result["epochs"], 3)
            self.assertEqual(
                result["parent_file_sha256"], result["candidate_prefix_file_sha256"]
            )

    def test_changed_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_id = "r"
            parent_records = self.records()[:3]
            parent_records[1]["val_ap"] += 0.01
            parent = root / "parent.csv"
            worker.history_frame(parent_records, run_id).to_csv(parent, index=False)
            with self.assertRaises(worker.WorkerError):
                worker.verify_csv_prefix(
                    self.records(), parent, run_id, root / "candidate_prefix.csv"
                )


class ControllerTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(
            [
                {
                    "checkpoint_at_upper_boundary": 0,
                    "stopped_early": 0,
                    "practical_plateau_admitted_after_extended_repair": 0,
                    "result_provenance": "repaired_v5",
                    "repair_iteration": 1,
                },
                {
                    "checkpoint_at_upper_boundary": 1,
                    "stopped_early": 0,
                    "practical_plateau_admitted_after_extended_repair": 1,
                    "result_provenance": "repaired_v5",
                    "repair_iteration": 1,
                },
                {
                    "checkpoint_at_upper_boundary": 0,
                    "stopped_early": 1,
                    "practical_plateau_admitted_after_extended_repair": 0,
                    "result_provenance": "new_v5",
                    "repair_iteration": 0,
                },
            ]
        )

    def test_masks_separate_unresolved_and_certified(self):
        frame = self.frame()
        self.assertEqual(controller.unresolved_mask(frame).tolist(), [True, False, False])
        self.assertEqual(controller.certified_mask(frame).tolist(), [False, True, False])

    def test_interrupted_transaction_is_rolled_back(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            official = root / "official_baseline"
            tx = official / ".official_extension_transactions" / "tx1"
            backup = tx / "backups" / "0000.bin"
            target = official / "raw_results.csv"
            backup.parent.mkdir(parents=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("new", encoding="utf-8")
            backup.write_text("old", encoding="utf-8")
            state = {
                "status": "APPLYING",
                "mutations": [
                    {
                        "target": str(target),
                        "backup": str(backup),
                        "existed": True,
                    }
                ],
            }
            controller.atomic_json(state, tx / "state.json")
            recovered = controller.recover_interrupted(official)
            self.assertEqual(len(recovered), 1)
            self.assertEqual(target.read_text(encoding="utf-8"), "old")
            self.assertEqual(
                json.loads((tx / "state.json").read_text())["status"], "ROLLED_BACK"
            )

    def test_budget_ladder_is_frozen(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "official_hinormer_repair.py"),
                "--caps",
                "7000,9000",
                "--project",
                str(ROOT / "missing"),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertNotEqual(completed.returncode, 0)


class SourceContractTests(unittest.TestCase):
    def test_test_evaluation_occurs_after_validation_ladder(self):
        source = (ROOT / "official_hinormer_worker.py").read_text(encoding="utf-8")
        selected_guard = source.index("if chosen is None:")
        final_test = source.index("adapter.finalize_test_evaluation")
        self.assertGreater(final_test, selected_guard)
        self.assertEqual(source.count("adapter.finalize_test_evaluation"), 1)

    def test_no_project_payload_is_bundled(self):
        self.assertFalse((ROOT / "payload").exists())


if __name__ == "__main__":
    unittest.main()
