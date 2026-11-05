"""Independent invariants and a fixture; no benchmark results are generated."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision.geometry import Geometry, L


class GeometryTests(unittest.TestCase):
    def test_joint_scaling_preserves_canonical_state(self):
        torch.manual_seed(42)
        old, new = 0.7, 2.4
        origin = L.origin(3, old, dtype=torch.float64).expand(6, 4)
        tangent = torch.cat([torch.zeros(6, 1), .1*torch.randn(6, 3)], dim=-1).double()
        x = L.exp_map(origin, tangent, old)
        m = L.project_to_tangent(x, torch.randn_like(x)*.02, old)
        a = (old/new)**.5
        y, n = a*x, a*m
        self.assertLess(float((L.lorentz_inner(y, y)+1/new).abs().max()), 1e-10)
        self.assertLess(float(L.lorentz_inner(y, n).abs().max()), 1e-10)
        self.assertTrue(torch.allclose(old**.5*x, new**.5*y, atol=1e-12, rtol=0))
        self.assertTrue(torch.allclose(old**.5*m, new**.5*n, atol=1e-12, rtol=0))

    def test_a5_isolates_cross_curvature_momentum(self):
        torch.manual_seed(42)
        full = Geometry([(8, 3)], 'B5', 'cpu', rho=.1)
        point_only = Geometry([(8, 3)], 'A5', 'cpu', rho=.1)
        point_only.load_state_dict(full.state_dict())
        for model in [full, point_only]:
            loss = model.chart_points()[0][:, 1:].square().sum()+model.K().sum()
            loss.backward(); model.step()
        self.assertTrue(torch.allclose(full.points[0], point_only.points[0]))
        self.assertFalse(torch.allclose(full.m[0], point_only.m[0], atol=1e-10, rtol=1e-5))


class ProtocolTests(unittest.TestCase):
    def test_scope_is_an_exact_partition(self):
        cfgs = json.loads((ROOT/'experiments/revision/manifest.json').read_text())
        scope = json.loads((ROOT/'experiments/revision/execution_scope.json').read_text())
        ids = {c['id'] for c in cfgs}
        deferred = set(scope['deferred_ids'])
        self.assertEqual(len(ids), len(cfgs))
        self.assertEqual(len(ids), 893)
        self.assertEqual(len(deferred), 188)
        self.assertEqual(len(ids-deferred), 705)
        self.assertTrue(all(c['seed'] == 42 for c in cfgs))
        blocks = {}
        for c in cfgs:
            if c.get('suite') == 'new_tasks_v1':
                blocks.setdefault(c['dataset'], []).append(c)
        self.assertEqual(len(blocks), 6)
        self.assertTrue(all(len(block) == 30 for block in blocks.values()))

    def test_cpu_entry_point_produces_only_a_marked_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(ROOT/'scripts/run_experiment.py'),
                            '--smoke', '--device', 'cpu', '--work-dir', tmp],
                           check=True, stdout=subprocess.DEVNULL)
            out = Path(tmp)/'runs/fixture_mechanism_B5'
            result = json.loads((out/'result.json').read_text())
            self.assertTrue(result['config']['fixture_only'])
            self.assertEqual(result['epochs_completed'], 3)
            rows = [json.loads(s) for s in (out/'trajectory.jsonl').read_text().splitlines()]
            self.assertEqual([r['epoch'] for r in rows], [0, 1, 2])
            self.assertTrue(all(torch.isfinite(torch.tensor(r['train_loss'])) for r in rows))
            self.assertTrue((Path(tmp)/'experiments/revision/manifest.json').is_file())


if __name__ == '__main__':
    unittest.main()
