# Token Field Network (TFN) Training Guide

This repository contains a **streamlined** implementation of Token Field Networks (TFN).

• **UnifiedTFN** – a single, fully-parameterised PyTorch module that covers *all* 1-D sequence use-cases (classification, regression, time-series) via a simple `task` flag.
• **ImageTFN** – the dedicated 2-D / image variant.
• **EnhancedTFN** – an optional research variant that augments UnifiedTFN with *field interference*, *dynamic propagation*, and *physics-inspired* constraints. Enable via `--model enhanced_tfn_classifier` (classification) or by passing `use_enhanced=True` to `UnifiedTFN`.

Legacy wrappers such as `TFNClassifier`, `TFNRegressor`, and `tfn.model.tfn_2d.*` remain as *thin aliases* for backward compatibility but will be removed in a future release.

TFN replaces attention with continuous field projection, evolution, and sampling; the new unified design eliminates code duplication while preserving full CLI configurability.

## 🚀 Quick Start

### Install Dependencies
```bash
pip install torch datasets pandas numpy
```

### Basic Training Examples
```bash
# Train TFN on AG News (text classification)
python -m tfn.scripts.train --task classification --dataset agnews --model tfn --epochs 10

# Train TFN on SST-2 (GLUE)
python -m tfn.scripts.train --task classification --dataset sst2 --model tfn --epochs 10

# Train TFN on Electricity Transformer Temperature (time-series)
python -m tfn.scripts.train --task time_series --dataset electricity --model tfn --epochs 10
```

## 📊 Available Datasets

### Text Classification Datasets
- **AG News**: 4-class news classification
- **IMDB**: Binary sentiment analysis
- **Yelp Full**: 5-class review classification

### GLUE Benchmark Tasks
- **SST-2**: Sentiment analysis (binary)
- **MRPC**: Paraphrase detection (binary)
- **QQP**: Question similarity (binary)
- **QNLI**: Question-answer entailment (binary)
- **RTE**: Textual entailment (binary)
- **CoLA**: Linguistic acceptability (binary)
- **STS-B**: Semantic similarity (regression)
- **WNLI**: Winograd NLI (binary)

### Climate/Time Series Datasets
- **Electricity Transformer Temperature**: Temperature prediction
- **Jena Climate**: Single-variable climate prediction
- **Jena Climate Multi**: Multi-variable climate prediction

### Other Datasets
- **Arxiv**: Paper classification by subject category
- **PG19**: Long text modeling
- **NER**: Named entity recognition
- **Synthetic**: Various synthetic sequence tasks

## 🤖 Available Models

### Classification Models
- **TFN (Unified)**: Standard Token Field Network
- **Enhanced TFN**: Field-interference variant (`enhanced_tfn_classifier`)
- **Transformer**: Standard Transformer encoder
- **Performer**: Linear attention approximation
- **LSTM**: LSTM-based classifier
- **CNN**: CNN-based classifier

### Regression Models
- **TFN**: Token Field Network for regression
- **Transformer**: Standard Transformer for regression
- **Performer**: Linear attention for regression
- **LSTM**: LSTM-based regressor
- **CNN**: CNN-based regressor

## 🎯 Training Commands

### Text Classification Training

#### Basic Text Classification
```bash
python -m tfn.scripts.train --task classification --dataset <dataset> --model tfn [options]
```

**Parameters:**
- `--dataset`: Dataset name (`agnews`, `imdb`, `yelp_full`)
- `--seq_len`: Sequence length (default: 128)
- `--embed_dim`: Embedding dimension (default: 128)
- `--num_layers`: Number of TFN layers (default: 2)
- `--kernel_type`: Kernel type (`rbf`, `compact`, `fourier`) (default: `rbf`)
- `--evolution_type`: Evolution type (`cnn`, `spectral`, `pde`) (default: `cnn`)
- `--grid_size`: Grid size (default: 64)
- `--time_steps`: Time steps (default: 3)
- `--dropout`: Dropout rate (default: 0.1)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 10)
- `--lr`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--device`: Device (`cuda`, `cpu`, `auto`) (default: `auto`)
- `--num_workers`: DataLoader workers (default: 2)
- `--save_dir`: Save directory (default: `outputs`)
- `--tag`: Optional run tag

**Examples:**
```bash
# Train on AG News with default parameters
python -m tfn.scripts.train --task classification --dataset agnews --model tfn --epochs 10

# Train on IMDB with custom parameters
python -m tfn.scripts.train --task classification --dataset imdb --model tfn \
       --embed_dim 256 --num_layers 3 --batch_size 64 --epochs 20 --lr 3e-4

# Train on Yelp with larger model
python -m tfn.scripts.train --task classification --dataset yelp_full --model tfn \
       --embed_dim 512 --num_layers 4 --seq_len 256 --epochs 15
```

### GLUE Benchmark Training

All GLUE tasks use the same unified CLI; simply specify the dataset key and model.

```bash
# General form
python -m tfn.scripts.train --task classification --dataset <glue_task> --model <model> [options]

# Examples
python -m tfn.scripts.train --task classification --dataset sst2 --model tfn --epochs 10
python -m tfn.scripts.train --task classification --dataset mrpc --model transformer --epochs 15
python -m tfn.scripts.train --task classification --dataset qqp --model performer --embed_dim 256 --num_layers 3 --epochs 20
```

### Climate/Time Series Training

#### Climate Datasets with Model Selection
```bash
python -m tfn.scripts.train_climate_tfn --dataset <dataset> --model <model> [options]
```

**Parameters:**
- `--dataset`: Climate dataset (`electricity`, `jena`, `jena_multi`)
- `--model`: Model architecture (`tfn`, `transformer`, `performer`, `lstm`, `cnn`) (default: `tfn`)
- `--seq_len`: Sequence length (default: 128)
- `--step`: Step size for sliding window (default: 1)
- `--embed_dim`: Embedding dimension (default: 128)
- `--num_layers`: Number of layers (default: 2)
- `--kernel_type`: Kernel type for TFN (`rbf`, `compact`, `fourier`) (default: `rbf`)
- `--evolution_type`: Evolution type for TFN (`cnn`, `spectral`, `pde`) (default: `cnn`)
- `--grid_size`: Grid size for TFN (default: 64)
- `--time_steps`: Time steps for TFN (default: 3)
- `--dropout`: Dropout rate (default: 0.1)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 10)
- `--lr`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--device`: Device (`cuda`, `cpu`, `auto`) (default: `auto`)
- `--num_workers`: DataLoader workers (default: 2)
- `--save_dir`: Save directory (default: `outputs`)
- `--tag`: Optional run tag

**Examples:**
```bash
# Train TFN on Electricity Transformer Temperature
python -m tfn.scripts.train_climate_tfn --dataset electricity --model tfn --epochs 10

# Train Transformer on Jena Climate
python -m tfn.scripts.train_climate_tfn --dataset jena --model transformer --epochs 15

# Train LSTM on Jena Climate Multi-variable
python -m tfn.scripts.train_climate_tfn --dataset jena_multi --model lstm --seq_len 256 --epochs 20

# Train with custom sequence length and step size
python -m tfn.scripts.train_climate_tfn --dataset electricity --model tfn --seq_len 64 --step 2 --epochs 10
```

### Arxiv Training

#### Arxiv Papers Classification
```bash
python -m tfn.scripts.train_arxiv_tfn [options]
```

**Parameters:**
- `--seq_len`: Sequence length (default: 512)
- `--embed_dim`: Embedding dimension (default: 256)
- `--num_layers`: Number of TFN layers (default: 3)
- `--kernel_type`: Kernel type (`rbf`, `compact`, `fourier`) (default: `rbf`)
- `--evolution_type`: Evolution type (`cnn`, `spectral`, `pde`) (default: `cnn`)
- `--grid_size`: Grid size (default: 64)
- `--time_steps`: Time steps (default: 3)
- `--dropout`: Dropout rate (default: 0.1)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 10)
- `--lr`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--device`: Device (`cuda`, `cpu`, `auto`) (default: `auto`)
- `--num_workers`: DataLoader workers (default: 2)
- `--save_dir`: Save directory (default: `outputs`)
- `--tag`: Optional run tag

**Examples:**
```bash
# Train on Arxiv with default parameters
python -m tfn.scripts.train_arxiv_tfn --epochs 10

# Train with custom parameters
python -m tfn.scripts.train_arxiv_tfn --embed_dim 512 --num_layers 4 --batch_size 64 --epochs 20 --lr 3e-4
```

### Other Training Scripts

#### Long Text Training
```bash
python -m tfn.scripts.train_long [options]

```

#### NER Training
```bash
python -m tfn.scripts.train_ner_tfn [options]
```

#### PG19 Training
```bash
python -m tfn.scripts.train_pg19 [options]
```

#### Synthetic Sequence Training
```bash
python -m tfn.scripts.train_synthetic_seq [options]
```

#### CIFAR Training
```bash
python -m tfn.scripts.train_cifar_tfn [options]
python -m tfn.scripts.train_cifar_vit [options]
```

## 🏁 Benchmarking

Run multiple dataset / model configurations programmatically with the new benchmark driver:

```bash
# Quick 2-dataset sanity sweep
python -m tfn.scripts.benchmark --preset quick

# Full NLP benchmark (all GLUE + common text datasets)
python -m tfn.scripts.benchmark --preset nlp_full

# Time-series benchmark
python -m tfn.scripts.benchmark --preset time_series
```

Results are written to `outputs/benchmark_<preset>.json`.

## 🔄 Model Comparison Examples

### Compare All Models on SST-2
```bash
for model in tfn transformer performer lstm cnn; do
  python -m tfn.scripts.train_glue_tfn --task sst2 --model $model --epochs 10
done
```

### Compare TFN vs Transformer on All GLUE Tasks
```bash
for task in sst2 mrpc qqp qnli rte cola stsb wnli; do
  python -m tfn.scripts.train_glue_tfn --task $task --model tfn --epochs 10
  python -m tfn.scripts.train_glue_tfn --task $task --model transformer --epochs 10
done
```

### Compare Models on Climate Data
```bash
for model in tfn transformer lstm cnn; do
  python -m tfn.scripts.train_climate_tfn --dataset electricity --model $model --epochs 10
done
```

## 📁 Output Organization

Results are automatically organized by model and dataset:
```
outputs/
├── sst2_tfn_ed128_L2/
│   ├── best_model.pt
│   └── history.json
├── sst2_transformer_ed128_L2/
│   ├── best_model.pt
│   └── history.json
├── electricity_tfn_ed128_L2/
│   ├── best_model.pt
│   └── history.json
└── ...
```

## 🗂️ Dataset Sources

### Kaggle Datasets
- **AG News**: Available via HuggingFace datasets
- **IMDB**: `/kaggle/input/imdb-dataset-of-50k-movie-reviews/`
- **Yelp**: `/kaggle/input/yelp-dataset-yelp-review-full/`
- **GLUE Tasks**: Available via HuggingFace datasets
- **Electricity**: `/kaggle/input/electricity-transformer-temperature/`
- **Jena Climate**: `/kaggle/input/jena-climate-archive/`
- **Arxiv**: `/kaggle/input/arxiv-papers-2021/`

### HuggingFace Datasets
- **GLUE**: `datasets.load_dataset("glue", task_name)`
- **Electricity**: `datasets.load_dataset("mstz/electricity_transformer_temperature")`
- **Jena Climate**: `datasets.load_dataset("mstz/jena_climate")`
- **Arxiv**: `datasets.load_dataset("arxiv_dataset")`

## ⚙️ Model-Specific Parameters

### TFN Parameters
- `--kernel_type`: Kernel type (`rbf`, `compact`, `fourier`)
- `--evolution_type`: Evolution type (`cnn`, `spectral`, `pde`)
- `--grid_size`: Grid size for field evolution
- `--time_steps`: Number of time steps for evolution

### Transformer Parameters
- `--num_heads`: Number of attention heads (default: 4)
- `--embed_dim`: Embedding dimension
- `--num_layers`: Number of transformer layers

### Performer Parameters
- `--proj_dim`: Projection dimension for linear attention (default: 64)
- `--embed_dim`: Embedding dimension
- `--num_layers`: Number of performer layers

### LSTM Parameters
- `--hidden_dim`: Hidden dimension (default: 128)
- `--bidirectional`: Use bidirectional LSTM (default: True)
- `--num_layers`: Number of LSTM layers

### CNN Parameters
- `--num_filters`: Number of filters (default: 128)
- `--filter_sizes`: Filter sizes (default: [3, 4, 5])

## 🎯 Evaluation Metrics

### Classification Tasks
- **Accuracy**: Percentage of correct predictions
- **Loss**: Cross-entropy loss

### Regression Tasks
- **MSE**: Mean squared error
- **RMSE**: Root mean squared error
- **Loss**: MSE loss

## 🔧 Advanced Usage

### Custom Hyperparameter Search
```bash
# Grid search over learning rates
for lr in 1e-4 3e-4 1e-3 3e-3; do
  python -m tfn.scripts.train_glue_tfn --task sst2 --model tfn --lr $lr --tag lr$lr
done
```

### Model Size Comparison
```bash
# Compare different model sizes
for embed_dim in 64 128 256 512; do
  python -m tfn.scripts.train_glue_tfn --task sst2 --model tfn --embed_dim $embed_dim --tag ed$embed_dim
done
```

### Multi-GPU Training
```bash
# Use specific GPU
python -m tfn.scripts.train_glue_tfn --task sst2 --model tfn --device cuda:0
```

## 📊 Monitoring and Logging

All training runs save:
- **Best model checkpoint**: `best_model.pt`
- **Training history**: `history.json` with loss and metrics
- **Console output**: Real-time training progress

## 🚨 Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce `--batch_size` or `--seq_len`
2. **Dataset not found**: Ensure Kaggle datasets are in correct paths or use HuggingFace
3. **Import errors**: Install required dependencies with `pip install torch datasets pandas numpy`

### Performance Tips
- Use `--device cuda` for GPU training
- Adjust `--num_workers` based on your system
- Use `--batch_size` that fits in your GPU memory
- For long sequences, consider reducing `--seq_len`

## 📚 Additional Resources

- **TFN Architecture**: See `tfn/core/` for implementation details
- **Dataset Loaders**: See `tfn/tfn_datasets/` for dataset implementations
- **Model Definitions**: See `tfn/model/` for model architectures
- **Training Scripts**: See `tfn/scripts/` for all training scripts

## 🤝 Contributing

To add new datasets or models:
1. Add dataset loader in `tfn/tfn_datasets/`
2. Add model definition in `tfn/model/`
3. Update training script with new options
4. Update this README with new commands 

## ⚠️ Mixed-Precision Limitation
Currently, the ImageTFN model does not fully support mixed-precision (AMP) training due to dtype issues. If you attempt to use torch.cuda.amp or similar features, you may encounter runtime errors. This is a known limitation and will be addressed in a future release.

**Tip:** Always run scripts as modules using `python -m tfn.scripts.train ...` to ensure robust imports and package resolution, especially on platforms like Kaggle or Colab. 