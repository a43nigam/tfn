import pytest

from tfn.model import registry as reg


def test_get_model_config_known() -> None:
    cfg = reg.get_model_config("tfn_classifier")
    assert isinstance(cfg, dict)
    assert cfg["class"].__name__ == "TFN"


def test_get_model_config_unknown() -> None:
    with pytest.raises(ValueError):
        _ = reg.get_model_config("nonexistent_model")


def test_validate_model_task_compatibility() -> None:
    assert reg.validate_model_task_compatibility("tfn_classifier", "classification") is True
    assert reg.validate_model_task_compatibility("tfn_classifier", "vision") is False


def test_required_optional_params_presence() -> None:
    req = reg.get_required_params("tfn_classifier")
    opt = reg.get_optional_params("tfn_classifier")
    assert "vocab_size" in req
    assert "dropout" in opt 