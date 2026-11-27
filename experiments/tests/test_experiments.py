# Copyright (c) 2026 RoboJEPA experiment contributors.
# Licensed under the LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path

import numpy as np
from experiments.runner import commands, elites, ladder, parser, task_spec, write_json
from experiments.metrics import module_metrics, summarize, trial_metrics


def configs(*args):
    return ladder(parser().parse_args(["run", "--output", "/tmp/test-experiment", *args]))


def config(*args):
    return configs(*args)[0]


def call(decision=1):
    return dict(module="predictor", decision=decision, batch_size=2, output_tokens=600, latency_s=0.2,
                flops=800, allocated_before_bytes=100, allocated_after_bytes=200,
                peak_allocated_bytes=300, peak_increment_bytes=200)


class ExperimentsTest(unittest.TestCase):
    def test_obstacle_adjustments_and_budget(self):
        c = config()
        self.assertEqual((c["planning_steps"], c["horizon"], c["embodiment_id"]), (18, 5, 3))
        server, planner = commands(c, Path("/tmp/trial"))
        for flag, value in [("--reach_min_distance", "0.25"), ("--reach_max_distance", "0.5"), ("--obstacle_clearance_m", "0.03")]:
            self.assertEqual(server[server.index(flag) + 1], value)
        self.assertEqual(planner[planner.index("--num_batches") + 1], "64")

    def test_all_tasks(self):
        for task in ("reach", "object_reach", "obstacle_reach", "push"):
            c = config("--task", task)
            server, planner = commands(c, Path("/tmp/trial"))
            self.assertIn("franka_omron_state_delta", server)
            self.assertEqual(planner[planner.index("--embodiment_id") + 1], "3")
            spec = task_spec(c, 17)
            self.assertEqual(spec["goal"]["mode"], "goal")
            self.assertIn(17, spec["init"].values())

    def test_invalid_settings(self):
        for args in [("--samples", "1"), ("--samples", "17"), ("--port", "29531"), ("--seed", "-1"), ("--reach-min-distance", "0.8")]:
            with self.assertRaises(ValueError):
                config(*args)

    def test_module_aggregation(self):
        m = module_metrics([call(), call()])
        self.assertEqual(m["tokens_per_second"], 3000)
        self.assertEqual(m["latency_amortized_sample_s"], 0.1)
        self.assertEqual(m["flops"], 1600)
        self.assertEqual(m["macs"], 800)
        self.assertEqual(m["average_boundary_allocated_bytes"], 150)
        self.assertEqual(m["peak_allocated_bytes"], 300)
        c = call()
        c["flops"] = None
        self.assertIsNone(module_metrics([c])["flops"])
        self.assertIsNone(module_metrics([]))

    def test_success_steps_and_failed_trial_denominator(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            # push: success is the simulator's flag at any step
            c = config("--task", "push", "--trials", "3")
            write_json(out / "config.json", c)
            for seed, success in [(0, [False, True, False]), (1, [False] * 3)]:
                trial = out / f"trial_{seed:05d}"
                trial.mkdir()
                np.savez(trial / "episode_task_0.npz", env_rewards=[0.2, 0.8, 0.5], env_info_success=success)
                write_json(trial / "efficiency.json", {"calls": [call(i) for i in range(1, 4)],
                    "decisions": [dict(index=i, latency_s=0.3) for i in range(1, 4)]})
                m = trial_metrics(trial, c)
                self.assertEqual(m["planning_steps_to_success"], 2 if seed == 0 else None)
                self.assertEqual(m["task_progress"], 0.5)
                write_json(trial / "metrics.json", m)
            failed = out / "trial_00002"
            failed.mkdir()
            write_json(failed / "failure.json", {"error": "OOM"})
            s = summarize(out)
            self.assertEqual(s["success_rate"], 0.5)
            self.assertEqual(s["mean_planning_steps_to_success"], 2)
            self.assertEqual(s["failed_seeds"], [2])
            self.assertTrue((out / "trials.csv").exists())

    def test_reach_success_is_final_distance(self):
        with tempfile.TemporaryDirectory() as td:
            trial = Path(td) / "trial_00000"
            trial.mkdir()
            c = config("--success-distance", "0.05", "--success-reference", "target")
            # the host task's flag is always False; a dip below 5 cm that ends at 7 cm is not a success
            np.savez(trial / "episode_task_0.npz", env_rewards=[0.2, 0.9, 0.8], env_info_success=[False] * 3,
                     env_info_reach_distance_to_goal=[0.3, 0.04, 0.07], env_info_reach_d0=[0.4] * 3)
            write_json(trial / "efficiency.json", {"calls": [], "decisions": [dict(index=i, latency_s=0.3) for i in (1, 2, 3)]})
            m = trial_metrics(trial, c)
            self.assertFalse(m["success"])
            self.assertTrue(m["success_any_step"])
            self.assertEqual((m["final_distance_m"], m["start_distance_m"], m["planning_steps_to_success"]), (0.07, 0.4, 2))
            # goal_pose reference: distance to the pose where the goal image was taken (poses[0])
            poses = np.zeros((5, 7))
            poses[0, :3] = [1, 0, 0]
            poses[1:, 0] = [0.5, 0.9, 0.95, 0.97]
            np.savez(trial / "episode_task_0.npz", env_rewards=[0.2, 0.9, 0.8], env_info_success=[False] * 3,
                     env_info_reach_distance_to_goal=[0.3, 0.04, 0.07], env_info_reach_d0=[0.4] * 3, poses=poses)
            m = trial_metrics(trial, config())
            self.assertEqual(m["distance_reference"], "goal_pose")
            self.assertAlmostEqual(m["final_distance_m"], 0.03)
            self.assertAlmostEqual(m["start_distance_m"], 0.5)
            self.assertTrue(m["success"])
            self.assertEqual(m["final_distance_target_m"], 0.07)

    def test_ladder_and_elites(self):
        cs = configs("--preset", "smoke")
        self.assertEqual([c["samples"] for c in cs], [16, 64, 256])
        self.assertEqual({(c["trials"], c["planning_steps"], c["cem_iterations"]) for c in cs}, {(1, 6, 3)})
        self.assertTrue(cs[0]["output"].endswith("n0016"))
        # explicit options beat the preset
        self.assertEqual(config("--preset", "smoke", "--samples", "32", "--trials", "2")["trials"], 2)
        self.assertEqual([elites(n, 0.02, 4) for n in (4, 16, 64, 256, 512, 1024)], [2, 4, 4, 5, 10, 20])
        server, planner = commands(config("--samples", "1024"), Path("/tmp/trial"))
        self.assertEqual(planner[planner.index("--cutoff") + 1], "20")
        self.assertEqual(config("--samples", "8")["batch_size"], 8)

    def test_flops_counted_once_per_shape(self):
        import torch
        from experiments.instrument import Recorder
        layer = torch.nn.Linear(10, 20, bias=False)
        r = Recorder("once")
        f = r.wrap(layer, "encoder")
        for _ in range(3):
            f(torch.ones(2, 10))
        self.assertEqual([c["flops_counted"] for c in r.calls], [True, False, False])
        self.assertEqual([c["flops"] for c in r.calls], [800] * 3)
        m = module_metrics(r.calls)
        self.assertEqual((m["timed_calls"], m["flops"]), (2, 2400))

    def test_missing_success_is_not_failure(self):
        with tempfile.TemporaryDirectory() as td:
            trial = Path(td) / "trial_00000"
            trial.mkdir()
            np.savez(trial / "episode_task_0.npz", env_rewards=[1.0])
            write_json(trial / "efficiency.json", {"calls": [], "decisions": []})
            with self.assertRaisesRegex(ValueError, "success flags"):
                trial_metrics(trial, config())

    def test_actual_flop_counter_and_output_preserved(self):
        import torch
        from experiments.instrument import Recorder
        layer = torch.nn.Linear(10, 20, bias=False)
        x = torch.ones(2, 10)
        r = Recorder()
        torch.testing.assert_close(r.wrap(layer, "encoder")(x), layer(x))
        self.assertEqual(r.calls[0]["flops"], 800)
        self.assertEqual(r.calls[0]["output_tokens"], 2)
        self.assertIsNone(r.calls[0]["peak_allocated_bytes"])


if __name__ == "__main__":
    unittest.main()
