"""CPU-only synthetic checks for root or public-release manuscript FDG code.

Set MOPE_MODULE_PATH for pytest, or run directly with --module-path PATH.
No images, study results, Cellpose models, or full training runs are used.
"""

import argparse
from contextlib import contextmanager, redirect_stdout
import copy
import importlib.util
import io
import os
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

_MODULE = None
_DEFAULT_MODULE = Path(__file__).resolve().parents[1] / "few_shot_cellpose_mope_ranking.py"


def _mope():
    global _MODULE
    if _MODULE is None:
        source = Path(os.environ.get("MOPE_MODULE_PATH", _DEFAULT_MODULE)).resolve()
        if source.is_dir():
            source = source / "few_shot_cellpose_mope_ranking.py"
        spec = importlib.util.spec_from_file_location("mope_contract_target", source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load MoPE module from {source}")
        _MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MODULE)
    return _MODULE


@contextmanager
def _rejects(*error_types):
    try:
        yield
    except error_types:
        pass
    else:
        raise AssertionError("Expected rejection with " + str(error_types))


def _synthetic_checkpoint():
    mope = _mope()
    mope.set_training_seed(42)
    model = mope.MoPERanker().cpu().eval()
    rng = np.random.default_rng(42)
    healthy = rng.normal(-0.5, 0.2, size=(4, 25))
    diseased = rng.normal(0.8, 0.2, size=(3, 25))
    h_norm, d_norm, mean, denominator = mope.normalize_features(healthy, diseased)
    calibrator = mope.SeverityCalibrator(model)
    with redirect_stdout(io.StringIO()):
        thresholds = calibrator.calibrate(h_norm, d_norm)
    _, feature_names = mope.get_feature_vector({})
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feat_mean": mean,
        "feat_std": denominator,
        "feature_names": feature_names,
        "thresholds": thresholds,
        "calibrator_h_stats": calibrator.h_stats,
        "calibrator_d_stats": calibrator.d_stats,
        "history": None,
        "loo_binary_acc": None,
        "loo_ranking_acc": None,
        "config": mope.build_model_config(model, epochs=16, batch_size=128, seed=42),
    }
    return model, checkpoint, np.vstack([h_norm, d_norm])


def test_pooled_normalization_and_explicit_healthy_sensitivity():
    mope = _mope()
    # Unequal class sizes distinguish pooled kernels from class-mean averaging.
    healthy = np.array([[0.0, 2.0, 7.0], [2.0, 4.0, 7.0]])
    diseased = np.array([[100.0, 8.0, 7.0]])
    pooled = np.vstack([healthy, diseased])
    h_norm, d_norm, mean, denominator = mope.normalize_features(healthy, diseased)
    np.testing.assert_allclose(mean, pooled.mean(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(denominator, pooled.std(axis=0) + 1e-10, rtol=1e-12, atol=0)
    np.testing.assert_allclose(h_norm, (healthy - mean) / denominator)
    np.testing.assert_allclose(d_norm, (diseased - mean) / denominator)
    np.testing.assert_allclose(np.vstack([h_norm, d_norm]).mean(axis=0), 0, atol=1e-12)
    np.testing.assert_array_equal(np.vstack([h_norm, d_norm])[:, 2], 0)
    assert np.isfinite(h_norm).all() and np.isfinite(d_norm).all()
    explicit = mope.normalize_features(healthy, diseased, reference="all_training")
    for default_value, explicit_value in zip((h_norm, d_norm, mean, denominator), explicit):
        np.testing.assert_allclose(default_value, explicit_value)
    h_sens, d_sens, h_mean, h_denominator = mope.normalize_features(
        healthy, diseased, reference="healthy"
    )
    np.testing.assert_allclose(h_mean, healthy.mean(axis=0))
    np.testing.assert_allclose(h_denominator, healthy.std(axis=0) + 1e-10)
    np.testing.assert_allclose(h_sens.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(d_sens, (diseased - h_mean) / h_denominator)
    assert not np.allclose(h_mean, mean)
    # This distinguishes std + epsilon from max(std, epsilon).
    _, _, _, custom_denominator = mope.normalize_features(healthy, diseased, epsilon=0.25)
    np.testing.assert_allclose(custom_denominator, pooled.std(axis=0) + 0.25)


def test_normalization_rejects_invalid_training_features():
    mope = _mope()
    valid = np.ones((2, 25), dtype=np.float64)
    for invalid in (np.zeros((0, 25)), np.zeros(25), np.full((2, 25), np.nan)):
        with _rejects(ValueError):
            mope.normalize_features(invalid, valid)
    with _rejects(ValueError):
        mope.normalize_features(valid, np.ones((2, 24)))
    with _rejects(ValueError):
        mope.normalize_features(valid, valid, epsilon=0)
    with _rejects(ValueError):
        mope.normalize_features(valid, valid, reference="evaluation")


def test_gate_matches_group_mean_absolute_deviation():
    mope = _mope()
    gate = mope.DeviationGuidedGating(initial_temperature=1.0).cpu()
    learned = list(gate.parameters())
    assert len(learned) == 1 and learned[0].numel() == 1
    np.testing.assert_allclose(float(gate.temperature.detach()), 1.0, atol=1e-6)
    # With 11/11/3 dimensions, sum(abs(x)) would incorrectly favour 11-D groups.
    equal = gate(torch.ones(1, 11), torch.ones(1, 11), torch.ones(1, 3))
    np.testing.assert_allclose(equal.detach().numpy(), [[1 / 3] * 3], atol=1e-7)
    groups = [torch.full((2, width), float(index + 1))
              for index, width in enumerate((11, 11, 3))]
    for group in groups:
        group[1] *= -1
    weights = gate(*groups, None)
    expected = torch.softmax(torch.tensor([[1.0, 2.0, 3.0]]), dim=-1)
    torch.testing.assert_close(weights, expected.expand(2, -1))
    torch.testing.assert_close(weights[0], weights[1])
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    zero_weights = gate(torch.zeros(1, 11), torch.zeros(1, 11), torch.zeros(1, 3))
    np.testing.assert_allclose(zero_weights.detach().numpy(), [[1 / 3] * 3], atol=1e-7)
    weights[:, 0].sum().backward()
    assert gate.raw_temperature.grad is not None
    assert torch.isfinite(gate.raw_temperature.grad).all()
    assert float(gate.raw_temperature.grad.abs()) > 0
    assert float(gate.temperature.detach()) > 0


def test_gate_rejects_empty_or_nonfinite_feature_groups():
    mope = _mope()
    gate = mope.DeviationGuidedGating().cpu()
    texture, morphology = torch.ones(1, 11), torch.ones(1, 3)
    for invalid in (torch.empty(1, 0), torch.ones(11), torch.full((1, 11), np.nan),
                    torch.full((1, 11), np.inf)):
        with _rejects(ValueError):
            gate(invalid, texture, morphology)


def test_model_dimensions_and_weighted_output():
    mope = _mope()
    model = mope.MoPERanker().cpu()
    assert sum(parameter.numel() for parameter in model.parameters()) == 29
    assert list(model.gating.state_dict()) == ["raw_temperature"]
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    scores, gates, experts = model(torch.zeros((2, 25), dtype=torch.float32))
    assert scores.shape == (2, 1) and gates.shape == (2, 3) and experts.shape == (2, 3)
    torch.testing.assert_close(scores, (gates * experts).sum(dim=1, keepdim=True))
    assert torch.isfinite(scores).all() and torch.all((scores > 0) & (scores < 1))


def test_threshold_uses_class_balanced_training_anchors():
    mope = _mope()
    threshold = mope.compute_vsk_threshold(0.2, 0.8, 0.6, lambda_value=0.25)
    np.testing.assert_allclose(threshold, 0.825, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mope.compute_vsk_threshold(0.2, 0.8, 0.6), threshold)
    np.testing.assert_allclose(mope.compute_vsk_threshold(0.2, 0.8, 0.6, 0), 0.8)
    for position in range(4):
        values = [0.2, 0.8, 0.6, 0.25]
        values[position] = np.nan
        with _rejects(ValueError):
            mope.compute_vsk_threshold(*values)


def test_manuscript_defaults_and_seed_reproducibility():
    mope = _mope()
    assert mope.NUM_EPOCHS == 16 and mope.TRAINING_SEED == 42
    assert mope.CELLPOSE_FIXED_DIAMETER == 95
    assert mope.MODEL_SCHEMA == "fdg_pooled_reference_v1"
    assert mope.NORMALIZATION_REFERENCE == "all_training"
    assert mope.VSK_THRESHOLD_LAMBDA == 0.25
    mope.set_training_seed()
    first = (random.random(), np.random.random(), torch.rand(3))
    first_model = mope.MoPERanker().cpu()
    mope.set_training_seed(42)
    second = (random.random(), np.random.random(), torch.rand(3))
    second_model = mope.MoPERanker().cpu()
    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
    for key, tensor in first_model.state_dict().items():
        torch.testing.assert_close(tensor, second_model.state_dict()[key], rtol=0, atol=0)
    config = mope.build_model_config(first_model, epochs=16, batch_size=128, seed=42)
    assert config["model_schema"] == "fdg_pooled_reference_v1"
    assert config["normalization_reference"] == "all_training"
    assert config["epochs"] == 16 and config["batch_size"] == 128 and config["seed"] == 42
    assert config["cellpose_diameter"] == 95 and config["aggregation_lambda"] == 0.25
    assert config["aggregation_reference_center"] == "class_balanced_anchor_midpoint"


def test_two_epoch_cpu_optimization_has_finite_loss():
    mope = _mope()
    mope.set_training_seed(42)
    model = mope.MoPERanker().cpu()
    rng = np.random.default_rng(42)
    healthy = rng.normal(-0.5, 0.2, size=(4, 25))
    diseased = rng.normal(0.8, 0.2, size=(3, 25))
    h_norm, d_norm, _, _ = mope.normalize_features(healthy, diseased)
    history = mope.PairwiseRankingTrainer(model).train(
        h_norm, d_norm, num_epochs=2, batch_size=8, verbose=False, silent=True
    )
    assert len(history["loss"]) == 2 and np.isfinite(history["loss"]).all()
    assert float(model.gating.temperature.detach()) > 0
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_checkpoint_round_trip_on_cpu():
    mope = _mope()
    model, checkpoint, features = _synthetic_checkpoint()
    mope.validate_checkpoint(checkpoint)
    inputs = torch.as_tensor(features, dtype=torch.float32)
    with torch.no_grad():
        expected = model(inputs)
    with tempfile.TemporaryDirectory(prefix="mope_fdg_contract_") as directory:
        torch.save(checkpoint, Path(directory) / "mope_model.pth")
        with redirect_stdout(io.StringIO()):
            loaded = mope.load_model(directory)
        assert loaded is not None and len(loaded) == 8
        loaded_model, _, mean, denominator, names, *_ = loaded
        assert all(parameter.device.type == "cpu" for parameter in loaded_model.parameters())
        np.testing.assert_array_equal(mean, checkpoint["feat_mean"])
        np.testing.assert_array_equal(denominator, checkpoint["feat_std"])
        assert list(names) == list(checkpoint["feature_names"])
        with torch.no_grad():
            actual = loaded_model(inputs)
        for expected_tensor, actual_tensor in zip(expected, actual):
            torch.testing.assert_close(expected_tensor, actual_tensor, rtol=0, atol=0)


def test_checkpoint_rejects_legacy_gate_and_wrong_normalization():
    mope = _mope()
    _, valid, _ = _synthetic_checkpoint()
    variants = []
    unversioned = copy.deepcopy(valid)
    unversioned.pop("config")
    variants.append(unversioned)
    healthy_only = copy.deepcopy(valid)
    healthy_only["config"]["model_schema"] = "fdg_healthy_reference_v1"
    healthy_only["config"]["normalization_reference"] = "training_healthy_only"
    variants.append(healthy_only)
    wrong_normalization = copy.deepcopy(valid)
    wrong_normalization["config"]["normalization_reference"] = "healthy"
    variants.append(wrong_normalization)
    legacy_linear = copy.deepcopy(valid)
    legacy_linear["model_state_dict"].pop("gating.raw_temperature")
    legacy_linear["model_state_dict"]["gating.fc.weight"] = torch.zeros((3, 25))
    legacy_linear["model_state_dict"]["gating.fc.bias"] = torch.zeros(3)
    variants.append(legacy_linear)
    for temperature in (torch.zeros(2), torch.tensor(float("nan"))):
        invalid = copy.deepcopy(valid)
        invalid["model_state_dict"]["gating.raw_temperature"] = temperature
        variants.append(invalid)
    for scale in (np.ones(24), np.zeros(25), np.full(25, np.nan)):
        invalid = copy.deepcopy(valid)
        invalid["feat_std"] = scale
        variants.append(invalid)
    for invalid in variants:
        with _rejects(ValueError, RuntimeError):
            mope.validate_checkpoint(invalid)
        with tempfile.TemporaryDirectory(prefix="mope_legacy_contract_") as directory:
            path = Path(directory) / "mope_model.pth"
            torch.save(invalid, path)
            before = path.read_bytes()
            with redirect_stdout(io.StringIO()), _rejects(ValueError, RuntimeError):
                mope.load_model(directory)
            assert path.read_bytes() == before
    for field, value in (("aggregation_lambda", 0.75),
                         ("aggregation_reference_center", "kernel_count_weighted")):
        invalid = copy.deepcopy(valid)
        invalid["config"][field] = value
        with tempfile.TemporaryDirectory(prefix="mope_wrong_rule_") as directory:
            torch.save(invalid, Path(directory) / "mope_model.pth")
            with redirect_stdout(io.StringIO()), _rejects(ValueError, RuntimeError):
                mope.load_model(directory)


def test_cellpose_factory_enforces_named_v3_without_downloads():
    mope = _mope()
    constructed = []
    sentinel = object()

    def fake_constructor(**kwargs):
        constructed.append(kwargs)
        return sentinel

    fake_models = SimpleNamespace(Cellpose=fake_constructor)
    fake_cellpose = ModuleType("cellpose")
    fake_cellpose.models = fake_models
    fake_modules = {"cellpose": fake_cellpose, "cellpose.models": fake_models}
    with patch.dict(sys.modules, fake_modules):
        with patch.object(mope.importlib_metadata, "version", return_value="3.1.1.1"):
            assert mope.create_cellpose_model(use_gpu=False) is sentinel
        assert constructed == [{"gpu": False, "model_type": "cyto"}]

        # Reject incompatible or unknown versions before touching construction.
        for version in ("4.0.8", "unversioned"):
            with patch.object(mope.importlib_metadata, "version", return_value=version):
                with _rejects(RuntimeError):
                    mope.create_cellpose_model(use_gpu=False)
            assert len(constructed) == 1
        missing = mope.importlib_metadata.PackageNotFoundError("cellpose")
        with patch.object(mope.importlib_metadata, "version", side_effect=missing):
            with _rejects(RuntimeError):
                mope.create_cellpose_model(use_gpu=False)
        assert len(constructed) == 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-path", type=Path, help="Root or release MoPE module")
    args = parser.parse_args()
    if args.module_path is not None:
        os.environ["MOPE_MODULE_PATH"] = str(args.module_path.resolve())
    torch.set_num_threads(1)
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} CPU synthetic checks passed for {_mope().__file__}")


if __name__ == "__main__":
    main()
