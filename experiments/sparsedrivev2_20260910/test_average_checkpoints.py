"""Analytic CPU tests for mean precision and immutable bank/BN buffers."""
from collections import OrderedDict
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
import average_checkpoints as average


def states():
    return [OrderedDict(weight=torch.tensor([large, small], dtype=torch.float32),
                        half_weight=torch.tensor([half], dtype=torch.float16),
                        bank=torch.tensor([[1., 2.]]), bn_running_mean=torch.tensor([.125]),
                        bn_num_batches_tracked=torch.tensor(7, dtype=torch.int64))
            for large, small, half in ((1e8, 1., .25), (3., 2., .75), (-1e8, 6., 1.25))]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.]))
        self.register_buffer("bank", torch.tensor([1., 2.]))
        self.register_buffer("bn_num_batches_tracked", torch.tensor(7, dtype=torch.int64))


class AveragingContracts(unittest.TestCase):
    def test_float64_mean_cast_once_and_exact_buffer_copy(self):
        source = states()
        result = average.average_state_dicts(source, ("weight", "half_weight"),
                                             ("bank", "bn_running_mean", "bn_num_batches_tracked"))
        self.assertTrue(torch.equal(result["weight"], torch.tensor([1., 3.])))
        self.assertTrue(torch.equal(result["half_weight"], torch.tensor([.75], dtype=torch.float16)))
        for name in ("bank", "bn_running_mean", "bn_num_batches_tracked"):
            self.assertTrue(torch.equal(result[name], source[0][name]))
            self.assertNotEqual(result[name].data_ptr(), source[0][name].data_ptr())

    def test_bank_and_bn_changes_are_rejected(self):
        for key in ("bank", "bn_running_mean", "bn_num_batches_tracked"):
            with self.subTest(key=key):
                source = states()
                source[1][key] += 1
                with self.assertRaisesRegex(ValueError, "Persistent buffer changed"):
                    average.average_state_dicts(source, ("weight", "half_weight"),
                                                ("bank", "bn_running_mean", "bn_num_batches_tracked"))

    def test_unclassified_tensor_and_nonfinite_parameter_rejected(self):
        source = states()
        with self.assertRaisesRegex(ValueError, "State keys differ"):
            average.average_state_dicts(source, ("weight",), ("bank", "bn_running_mean", "bn_num_batches_tracked"))
        source[1]["weight"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            average.average_state_dicts(source, ("weight", "half_weight"),
                                        ("bank", "bn_running_mean", "bn_num_batches_tracked"))

    def test_full_source_receipts_and_recomputed_average(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            model = TinyModel()
            manifest = {"arguments": {"run_dir": str(root / "run")}}
            records = []
            for step, value in zip(average.STEPS, (1., 4., 7.)):
                state = copy.deepcopy(model.state_dict())
                state["weight"].fill_(value)
                payload = {"model": state, "manifest": manifest, "step": step, "epoch": 2, "result": {"step": step}}
                path = root / f"step{step}.pth"
                torch.save(payload, path)
                digest = average.sha(path)
                path.with_suffix(".json").write_text(json.dumps({"sha256": digest, "embedded_step": step}))
                records.append({"path": str(path), "sha256": digest, "step": step})
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps({"steps": average.STEPS, "runs": ["run"]}))
            epochs = []
            state = average.average_state_dicts(average.source_states(records, manifest, epochs),
                                                dict(model.named_parameters()), dict(model.named_buffers()))
            metadata = {"version": 1, "method": average.METHOD, "steps": average.STEPS, "sources": records,
                        "protocol_path": str(protocol_path), "protocol_sha256": average.sha(protocol_path),
                        "postprocessor_path": average.__file__, "postprocessor_sha256": average.sha(average.__file__)}
            output = {"model": state, "manifest": manifest, "step": 2000, "epoch": epochs[-1],
                      "result": None, "averaging": metadata}
            verified = average.verify_averaged_state(output, model)
            self.assertTrue(verified["recomputed_average_bitwise_equal"])
            self.assertEqual(state["weight"].item(), 4.)
            output["model"]["weight"] += .01
            with self.assertRaisesRegex(ValueError, "Averaged tensor differs"):
                average.verify_averaged_state(output, model)

    def test_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "average.pth"
            average.write_immutable(path, {"test": torch.tensor(1)})
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                average.write_immutable(path, {"test": torch.tensor(2)})
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
