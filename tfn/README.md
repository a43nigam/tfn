# Token Field Network (TFN)

A novel architecture that replaces attention with continuous field **projection → evolution → sampling**. TFN scales linearly, preserves spatial inductive bias, and is differentiable end-to-end.

> **Key idea** – every token emits a learnable Gaussian/RBF field on a 1-D (or 2-D) grid. Fields evolve via CNN/PDE operators, optionally interact (interference), and are re-sampled back into token embeddings.

---

## 📦 Repository Layout

```
tfn/
├── core/              # kernels, projection, evolution, sampling, interference
├── model/             # high-level PyTorch modules + baselines
├── tfn_datasets/      # lightweight dataset loaders (text, time-series, vision, physics, …)
├── scripts/           # unified training + benchmark CLIs
├── tests/             # >200 unit tests covering maths + numerics
└── utils/             # metrics, plotting, synthetic generators
```

Every directory is **unit-testable in isolation**; run `pytest -q` for full coverage.

---

## 🚀 Quick Start

```bash
# 1. Install (CPU)
pip install torch pandas numpy datasets

# 2. Train on AG News (text classification)
python -m tfn.scripts.train \
    --task classification \
    --dataset agnews \
    --model tfn_classifier \
    --model_kwargs '{"embed_dim": 128}' \
    --epochs 10
```

TIP : add `--device cuda` for GPU, or omit to let TFN auto-detect.

---

## 🏗 Supported Tasks & Datasets

| Task                   | Dataset keys (📥 `--dataset`)                              |
|------------------------|------------------------------------------------------------|
| Text **classification**| `agnews`, `imdb`, `yelp_full`, plus all GLUE: `sst2`, `mrpc`, `qqp`, `qnli`, `rte`, `cola`, `wnli` |
| Text **regression**    | `stsb`                                                     |
| **Time-series**        | `electricity`, `jena`, `jena_multi`                        |
| **Language modelling** | `pg19`, `long_text_synth`                                  |
| **Vision** (β)         | `cifar10`, `cifar100`, `imagenet32`                        |
| **Physics / PDE**      | `pde_burgers_synthetic`, `pde_wave_synthetic`, `pde_heat_synthetic` *(synthetic generators)* |
| **NER** (β)            | `conll2003`                                               |
| **Synthetic**          | `synthetic_copy`, `synthetic_reverse`                      |

> All loaders live in `tfn/tfn_datasets/registry.py` – pass dataset-specific args via JSON, e.g. `--dataset_kwargs '{"seq_len": 256}'`.

---

## 🧠 Available Models (📦 `--model`)

| Category          | Registry key                | Notes |
|-------------------|-----------------------------|-------|
| **Token Field Networks** | `tfn_classifier`, `tfn_regressor`, `tfn_timeseries_regressor`, `tfn_sequence_regressor`, `tfn_language_model` | Core architecture (field projection + evolution + sampling) |
| **Enhanced TFN**  | `enhanced_tfn_classifier`, `enhanced_tfn_language_model` | Adds field interference, dynamic propagation, physics constraints |
| **Image TFN (2-D)**| `tfn_vision` | 2-D fields, CNN/PDE evolvers; experimental |
| **Baselines**     | `transformer_classifier`, `performer_classifier`, `lstm_classifier`, `cnn_classifier`, … (replace `classifier` with `regressor` / `language_model` for other tasks) | Reference implementations

Default hyper-parameters are stored in the registry (`tfn/model/registry.py`). Override any of them via CLI:

```bash
--embed_dim 256 --num_layers 4 --kernel_type compact --evolution_type pde
```

---

## 🔧 Unified Training CLI

All 1-D models share a **single** entry-point:

```bash
python -m tfn.scripts.train \
    --task <classification|regression|time_series|language_modeling|ner> \
    --dataset <dataset_key> \
    --model <model_key> \
    [common args] [dataset-specific args] [model-specific args]
```

### Common args
`--epochs` (10) • `--batch_size` (32) • `--lr` (1e-3) • `--device` (`cuda`/`cpu`) • `--num_workers` (2) • `--dry_run`

### Examples

1. **Text classification (GLUE – SST-2)**
```bash
python -m tfn.scripts.train \
  --task classification --dataset sst2 --model tfn_classifier \
  --model_kwargs '{"embed_dim": 256, "num_layers": 3}' \
  --epochs 10
```

2. **Time-series forecasting (Electricity)**
```bash
python -m tfn.scripts.train \
  --task time_series --dataset electricity --model tfn_timeseries_regressor \
  --dataset_kwargs '{"seq_len": 168, "step": 1}' \
  --epochs 5
```

3. **Language modelling (PG-19) with Enhanced TFN**
```bash
python -m tfn.scripts.train \
  --task language_modeling --dataset pg19 --model enhanced_tfn_language_model \
  --model_kwargs '{"interference_type": "standard", "propagator_type": "wave"}' \
  --epochs 3
```

4. **Baseline Transformer on IMDB**
```bash
python -m tfn.scripts.train \
  --task classification --dataset imdb --model transformer_classifier --epochs 5
```

> Check the registry or run with `--dry_run` to print all required/optional parameters before training.contin

---

## ⚡ Benchmark Runner

```bash
# Quick 2-dataset sanity sweep
python -m tfn.scripts.benchmark --preset quick
```

Presets live inside `scripts/benchmark.py` (`quick`, `nlp_small`, `nlp_full`, `time_series`). Results are written to `outputs/benchmark_<preset>.json`.

---

## 🔬 Research Features

* **Field Interference** – multi-token constructive/destructive interaction.
* **Dynamic Propagation** – learnable PDE/CNN operators chosen per layer.
* **Physics Constraints** – optional loss enforcing e.g. Burgers/Heat/Wave equations.
* **2-D ImageTFN** – extend fields over (H × W) grid; see `tests/test_tfn_pytorch.py` for usage.

Enable via the Enhanced model registry keys and pass JSON kwargs:
```bash
--model_kwargs '{"interference_type": "standard", "propagator_type": "diffusion", "use_physics_constraints": true}'
```

---

## 🗂 Output Structure

```
outputs/
├── agnews_tfn_classifier/           # dataset_model
│   ├── best_model.pt               # final checkpoint
│   ├── history.json                # train/val loss curves
│   └── cfg.json                    # full CLI args dump (auto-saved)
└── benchmark_quick.json
```

---

## 🛠 Developer Notes

1. **Numerical stability** – all evolvers support FP32/AMP; kernels normalised to avoid blow-up.
2. **Modularity** – every component has a public `forward` with clear typing + docstring.
3. **Testing** – run `pytest -q` for the full suite.
4. **Extending TFN** – add a new kernel/evolution in `core/`, register it, add a unit-test, and update this README.

---

## 🤝 Contributing

Pull requests welcome! Please:
1. Run full test-suite.
2. Include docstrings + typing.
3. Update the registry + this README if you add a new model or dataset. 