from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class ControllerIntegrationTest(unittest.TestCase):
    def test_mock_worker_commits_target_and_metadata_backfill(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            project = temp / "project"
            output = project / "outputs" / "final"
            official = output / "official_baseline"
            canonical = output / "canonical_instances"
            scripts = project / "scripts"
            config = project / "configs" / "final_paper.yaml"
            for path in (official / "runs", canonical, scripts, config.parent):
                path.mkdir(parents=True, exist_ok=True)
            config.write_text("max_epochs: 1500\n", encoding="utf-8")
            (scripts / "run_official_hinormer.py").write_text("# mock adapter\n", encoding="utf-8")
            (output / "plateau_v2_migration_report.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "policy_version_after": "validation_practical_plateau_v2",
                        "transaction_committed": True,
                    }
                ),
                encoding="utf-8",
            )
            seeds = [13, 29, 47, 71, 101, 131, 167, 199, 233, 269]
            alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
            trains = [1001, 1002, 1003]
            protocol = {
                "run_signature": "b" * 64,
                "alphas": alphas,
                "data_seeds": seeds,
                "train_seeds": trains,
            }
            (official / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            rows = []
            for seed in seeds:
                for alpha in alphas:
                    for train in trains:
                        run_id = f"seed_{seed}_alpha_{alpha}_train_{train}"
                        row = {
                            "run_id": run_id,
                            "run_signature": "b" * 64,
                            "data_seed": seed,
                            "alpha": alpha,
                            "train_seed": train,
                            "checkpoint_at_upper_boundary": 0.0,
                            "checkpoint_at_boundary": 0.0,
                            "stopped_early": 1.0,
                            "practical_plateau_admitted_after_extended_repair": 0.0,
                            "practical_plateau_policy_version": "validation_practical_plateau_v2",
                            "result_provenance": "new_v5",
                            "repair_iteration": 0.0,
                            "source_max_epochs": 1500.0,
                            "result_max_epochs": 1500.0,
                            "best_epoch": 100.0,
                            "degenerate_predictions": 0.0,
                        }
                        if (seed, alpha, train) == (13, 0.1, 1001):
                            row.update(
                                {
                                    "stopped_early": 0.0,
                                    "practical_plateau_admitted_after_extended_repair": 1.0,
                                    "result_provenance": "repaired_v5",
                                    "repair_iteration": 1.0,
                                    "result_max_epochs": 5000.0,
                                    "best_epoch": 4900.0,
                                }
                            )
                        if (seed, alpha, train) == (269, 0.9, 1003):
                            row.update(
                                {
                                    "stopped_early": 0.0,
                                    "result_provenance": "repaired_v5",
                                    "repair_iteration": 1.0,
                                    "result_max_epochs": 5000.0,
                                    "best_epoch": 4975.0,
                                }
                            )
                        rows.append(row)
                        run_dir = official / "runs" / run_id
                        run_dir.mkdir()
                        pd.DataFrame([{"run_id": run_id, "epoch": 1.0}]).to_csv(
                            run_dir / "training_history.csv", index=False
                        )
                        (run_dir / "predictions.npz").write_bytes(f"pred-{run_id}".encode())
                        (run_dir / "prediction_manifest.json").write_text(
                            json.dumps({"run_id": run_id}), encoding="utf-8"
                        )
                        (run_dir / "run.json").write_text(
                            json.dumps({"result": row, "health": {}}), encoding="utf-8"
                        )
            pd.DataFrame(rows).to_csv(official / "raw_results.csv", index=False)
            target_run = official / "runs" / "seed_269_alpha_0.9_train_1003"
            non_target_history = official / "runs" / "seed_29_alpha_0.3_train_1002" / "training_history.csv"
            non_target_before = non_target_history.read_bytes()

            worker = temp / "fake_worker.py"
            worker.write_text(
                textwrap.dedent(
                    """
                    import argparse, hashlib, json
                    from pathlib import Path
                    import numpy as np
                    p=argparse.ArgumentParser()
                    for name in ('project','output-root','canonical-root','config','stage-root','amendment-id','data-seed','alpha','train-seed','caps','device'):
                        p.add_argument('--'+name, required=True)
                    a=p.parse_args(); stage=Path(a.stage_root); selected=stage/'selected'; selected.mkdir(parents=True)
                    run_id='seed_269_alpha_0.9_train_1003'; base='b'*64
                    (selected/'training_history.csv').write_text('run_id,epoch\\n%s,1.0\\n'%run_id)
                    np.savez_compressed(selected/'predictions.npz', val_prob=np.array([.2]), test_prob=np.array([.8]))
                    manifest={'run_id':run_id,'run_signature':base}
                    result={'run_id':run_id,'run_signature':base,'data_seed':269,'alpha':.9,'train_seed':1003,
                      'checkpoint_at_upper_boundary':0.0,'checkpoint_at_boundary':0.0,'stopped_early':1.0,
                      'practical_plateau_admitted_after_extended_repair':0.0,
                      'practical_plateau_policy_version':'validation_practical_plateau_v2','result_provenance':'repaired_v5',
                      'repair_iteration':2.0,'source_max_epochs':1500.0,'parent_max_epochs':5000.0,
                      'result_max_epochs':8000.0,'best_epoch':6000.0,'degenerate_predictions':0.0}
                    (selected/'prediction_manifest.json').write_text(json.dumps(manifest))
                    (selected/'run.json').write_text(json.dumps({'result':result,'health':{}}))
                    arts={}
                    for n in ('training_history.csv','predictions.npz','prediction_manifest.json','run.json'):
                      x=selected/n; arts[n]={'path':str(x),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()}
                    wr={'status':'PASS','target':{'run_id':run_id,'data_seed':269,'alpha':.9,'train_seed':1003},
                      'base_run_signature':base,'runtime_sha256':'r'*64,'canonical_input_sha256':'i'*64,
                      'source_max_epochs':1500,'parent_max_epochs':5000,'selected_result_max_epochs':8000,
                      'test_metrics_used_for_cap_selection':False,'test_evaluation_count':1,
                      'prefix_verifications':[{'verified':True}],'selected_artifacts':arts}
                    (stage/'worker_result.json').write_text(json.dumps(wr))
                    """
                ),
                encoding="utf-8",
            )
            validator = temp / "fake_validator.py"
            validator.write_text("print('OFFICIAL_RESUME_VALIDATION=PASS retained=150')\n", encoding="utf-8")
            session = temp / "session"
            command = [
                sys.executable,
                str(ROOT / "official_hinormer_repair.py"),
                "--project",
                str(project),
                "--output-root",
                str(output),
                "--canonical-root",
                str(canonical),
                "--config",
                str(config),
                "--official-python",
                sys.executable,
                "--worker",
                str(worker),
                "--validator",
                str(validator),
                "--session-root",
                str(session),
                "--test-only-skip-source-audit",
            ]
            done = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**__import__("os").environ, "COGBENCH_TEST_MODE": "1"},
            )
            self.assertEqual(done.returncode, 0, done.stdout)
            final = pd.read_csv(official / "raw_results.csv")
            target = final[(final.data_seed == 269) & (final.alpha == .9) & (final.train_seed == 1003)].iloc[0]
            backfill = final[(final.data_seed == 13) & (final.alpha == .1) & (final.train_seed == 1001)].iloc[0]
            self.assertEqual(int(target.result_max_epochs), 8000)
            self.assertEqual(int(target.parent_max_epochs), 5000)
            self.assertEqual(int(backfill.parent_max_epochs), 1500)
            self.assertEqual(non_target_history.read_bytes(), non_target_before)
            self.assertNotEqual((target_run / "predictions.npz").read_bytes(), b"pred-seed_269_alpha_0.9_train_1003")
            marker = json.loads(
                (output / ".completed" / "official_hinormer_extension.json").read_text()
            )
            self.assertEqual(marker["status"], "PASS")
            state = json.loads((session / "state.json").read_text())
            self.assertEqual(state["status"], "PASS")
            second_session = temp / "session_second"
            second_command = list(command)
            second_command[second_command.index("--session-root") + 1] = str(second_session)
            second = subprocess.run(
                second_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**__import__("os").environ, "COGBENCH_TEST_MODE": "1"},
            )
            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertIn("ALREADY_COMMITTED", second.stdout)


if __name__ == "__main__":
    unittest.main()
