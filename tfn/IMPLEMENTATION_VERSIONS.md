# TFN Implementation Versions

This document explains the distinction between the two TFN implementations in this codebase.

## 🕰️ **Previous Implementation: Token-Based 1D TFN**

### **Purpose**
- **Domain**: 1D time series forecasting (ETT, Jena Climate, arXiv text)
- **Architecture**: Token-based field projection with complex token-field conversions
- **Key Use Cases**: Time series forecasting, text classification, sequence modeling

### **Core Components**
```python
# Field Projection (1D)
FieldProjector(embed_dim=64, pos_dim=1, kernel_type="rbf")

# Field Evolution (1D)  
FieldEvolver(embed_dim=64, pos_dim=1, evolution_type="pde")

# Grid Generation (1D)
UniformFieldGrid(pos_dim=1, grid_size=100, bounds=(0.0, 1.0))

# Time Series Regressors
TFNTimeSeriesRegressor(
    input_dim=features,
    embed_dim=128,
    output_len=1,  # Single-step forecasting
    num_layers=2,
    kernel_type="rbf",
    evolution_type="cnn",
    grid_size=64,
    time_steps=3
)
```

### **Key Features**
- **Temporal Causality**: Causal interference patterns respect time ordering
- **Token-Based**: Converts discrete tokens to continuous 1D fields
- **PDE Evolution**: Diffusion/wave equations for field dynamics
- **Multi-Scale**: Handles different time scales and frequencies
- **Complex Pipeline**: Token → Field → Evolution → Sampling → Token

### **Training Scripts**
```bash
# Climate time series
python train_climate_tfn.py --dataset jena --embed_dim 128 --num_layers 3

# ArXiv text classification  
python train_arxiv_tfn.py --embed_dim 256 --num_layers 4

# Long sequence tasks
python train_long.py --model tfn1d --seq_len 512
```

---

## 🆕 **New Implementation: PyTorch ImageTFN**

### **Purpose**
- **Domain**: 2D image classification (CIFAR-10/100, ImageNet)
- **Architecture**: Direct image processing with streamlined field operations
- **Key Use Cases**: Image classification, computer vision tasks

### **Core Components**
```python
# Image Field Emission (2D)
ImageFieldEmitter(in_channels=3, out_channels=64)

# Image Field Interference (2D)
ImageFieldInterference(num_heads=8)

# Image Field Propagation (2D)
ImageFieldPropagator(steps=4)

# Complete ImageTFN Model
ImageTFN(
    in_ch=3,  # RGB images
    num_classes=10
)
```

### **Key Features**
- **Direct Processing**: No token-field conversions, works directly on images
- **Computational Efficiency**: Low-rank interference + optimized convolutions
- **Physics-Aware**: Real PDE evolution with finite-difference methods
- **Adaptive Fields**: Dynamic field centers that learn optimal locations
- **Modular Design**: Each component can be tested independently
- **Production-Ready**: Mixed precision, CLI, benchmarking, checkpointing

### **Training Scripts**
```bash
# Train ImageTFN on CIFAR-10
python train_tfn_pytorch.py --dataset cifar10 --epochs 50 --batch_size 128

# Benchmark against baselines
python benchmark_tfn_pytorch.py --dataset cifar10 --epochs 50 --models tfn resnet vit

# Use mixed precision for faster training
python train_tfn_pytorch.py --dataset cifar10 --mixed_precision
```

---

## 📊 **Comparison Table**

| Aspect | Previous 1D TFN | New ImageTFN |
|--------|-----------------|-------------|
| **Domain** | Time series (1D) | Images (2D) |
| **Field Type** | Token-based projection | Direct image processing |
| **Interference** | Causal patterns | Low-rank mixing |
| **Evolution** | PDE diffusion/wave | Finite-difference Laplacian |
| **Efficiency** | O(N²) operations | O(N·rank) with low-rank |
| **Architecture** | Complex token-field conversions | Streamlined image pipeline |
| **Use Cases** | ETT, Jena Climate, arXiv | CIFAR, ImageNet |
| **Causality** | Temporal causality | Spatial relationships |
| **Production** | Research prototype | Enterprise-ready |

---

## 🎯 **When to Use Which Implementation**

### **Use Previous 1D TFN for:**
- Time series forecasting (ETT, Jena Climate)
- Text classification (arXiv papers)
- Sequence modeling tasks
- Tasks requiring temporal causality
- Research on token-based field dynamics

### **Use New ImageTFN for:**
- Image classification (CIFAR, ImageNet)
- Computer vision tasks
- 2D spatial data processing
- Production image models
- Tasks requiring spatial field dynamics

---

## 🔧 **File Organization**

### **Previous 1D TFN Files**
```
tfn/core/
├── field_projection.py      # Token-based field projection
├── field_evolution.py       # 1D field evolution
├── field_sampling.py        # Field sampling back to tokens
├── kernels.py              # 1D kernels (RBF, Compact, Fourier)
└── grid_utils.py           # 1D grid generation

tfn/model/
├── tfn_regressors.py       # Time series regressors
├── tfn_classifiers.py      # Text classifiers
└── tfn_base.py            # Base 1D TFN components

tfn/scripts/
├── train_climate_tfn.py    # Climate time series training
├── train_arxiv_tfn.py      # ArXiv text training
└── train_long.py          # Long sequence training
```

### **New ImageTFN Files**
```
tfn/core/
├── field_emitter.py           # ImageFieldEmitter for 2D
├── field_interference_block.py # ImageFieldInterference for 2D
└── field_propagator.py        # ImageFieldPropagator for 2D

tfn/model/
└── tfn_pytorch.py            # ImageTFN for 2D images

tfn/scripts/
├── train_tfn_pytorch.py      # ImageTFN training
└── benchmark_tfn_pytorch.py  # ImageTFN benchmarking

tfn/tests/
└── test_tfn_pytorch.py       # ImageTFN validation tests
```

---

## 🚀 **Migration Guide**

### **From 1D TFN to ImageTFN**
```python
# Old: Token-based approach
from tfn.model.tfn_regressors import TFNTimeSeriesRegressor
model = TFNTimeSeriesRegressor(input_dim=32, embed_dim=128)

# New: Direct image processing
from tfn.model.tfn_pytorch import ImageTFN
model = ImageTFN(in_ch=3, num_classes=10)
```

### **Key Differences**
1. **Input**: Tokens vs Images
2. **Field Processing**: Token→Field→Token vs Direct Image→Field
3. **Dimensionality**: 1D vs 2D
4. **Causality**: Temporal vs Spatial
5. **Efficiency**: O(N²) vs O(N·rank)

---

## 📈 **Performance Comparison**

### **Previous 1D TFN**
- **Complexity**: High (token-based conversions)
- **Memory**: O(N²) for token interactions
- **Speed**: Slower due to token-field conversions
- **Accuracy**: Good for time series, text

### **New ImageTFN**
- **Complexity**: Low (direct image processing)
- **Memory**: O(N·rank) with low-rank interference
- **Speed**: Faster due to optimized convolutions
- **Accuracy**: Good for images, scalable

---

## 🎉 **Summary**

The codebase now contains **two distinct TFN implementations**:

1. **Previous 1D TFN**: Token-based, time series focused, research prototype
2. **New ImageTFN**: Direct image processing, production-ready, optimized for 2D

Both implementations share the core **field-based philosophy** but are optimized for their respective domains and use cases. 