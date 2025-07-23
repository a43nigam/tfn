import json
import subprocess
import sys
from pathlib import Path

import pytest

CLI = [sys.executable, "-m", "tfn.scripts.train", "--dry_run", "--epochs", "1"]

# Representative (task, dataset, model) combos covering each task category
COMBOS = [
    ("classification", "sst2", "tfn_classifier"),
    ("classification", "agnews", "transformer_classifier"),
    ("regression", "stsb", "tfn_regressor"),
    ("time_series", "electricity", "tfn_timeseries_regressor"),
    ("language_modeling", "long_text_synth", "tfn_language_model"),
]

@pytest.mark.parametrize("task,dataset_key,model_key", COMBOS)
def test_cli_dry_run(task: str, dataset_key: str, model_key: str):
    """Spawn the training CLI with --dry_run and ensure zero exit status."""
    cmd = CLI + [
        "--task", task,
        "--dataset", dataset_key,
        "--model", model_key,
        "--model_kwargs", json.dumps({}),
        "--dataset_kwargs", json.dumps({}),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0, f"CLI failed: {result.stderr}\nSTDOUT:\n{result.stdout}" 