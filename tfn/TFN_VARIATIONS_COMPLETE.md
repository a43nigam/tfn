# Complete TFN Variations Analysis

## 🎯 **Overview**
This document provides a comprehensive analysis of ALL Token Field Network (TFN) variations implemented in the repository, organized by architecture type, task specialization, and implementation details.

---

## 📊 **TFN Variations Summary**

| Category | Count | Variations |
|----------|-------|------------|
| **Core Architectures** | 4 | 1D TFN, 2D TFN, ImageTFN, Enhanced TFN |
| **Task-Specific Models** | 8 | Classifiers, Regressors, Taggers, Language Models |
| **Field Components** | 12 | Emitters, Interference, Propagation, Evolution |
| **Total Variations** | **24+** | Complete TFN ecosystem |

---

## 🏗️ **1. CORE TFN ARCHITECTURES**

### **1.1 1D TFN (Token-Based)**
**Location**: `tfn/model/tfn_base.py` → `TrainableTFNLayer`

**Architecture**:
```
Tokens → Field Projection → Field Evolution → Field Sampling → Updated Tokens
```

**Key Features**:
- **1D spatial domain**: Linear grid for sequence processing
- **Token-centric**: Each token emits a field and samples back
- **Configurable kernels**: RBF, Compact, Fourier
- **Multiple evolution types**: CNN, Spectral, PDE
- **Learnable parameters**: Sigma, evolution coefficients

**Use Cases**: Text classification, time series, sequence modeling

---

### **1.2 2D TFN (Lattice-Based)**
**Location**: `tfn/model/tfn_2d.py` → `TrainableTFNLayer2D`

**Architecture**:
```
Tokens → 2D Field Projection → 2D Field Evolution → 2D Field Sampling → Updated Tokens
```

**Key Features**:
- **2D spatial domain**: H×W lattice for spatial processing
- **Gaussian field emission**: Each token emits 2D Gaussian patches
- **Depth-wise convolution**: 2D field evolution
- **Multiscale processing**: Optional up/down sampling
- **Kernel mixing**: Two-component Gaussian mixtures
- **Global context**: Optional global field context

**Use Cases**: Image classification, spatial sequence modeling

---

### **1.3 ImageTFN (Direct Image Processing)**
**Location**: `tfn/model/tfn_pytorch.py` → `ImageTFN`

**Architecture**:
```
Images → Field Emission → Field Interference → Field Propagation → Classification
```

**Key Features**:
- **Direct image processing**: No tokenization required
- **Field emission**: Convolutional field generation
- **Field interference**: Multi-head attention on fields
- **PDE propagation**: Physics-inspired field evolution
- **End-to-end**: Direct image → classification

**Use Cases**: Image classification, computer vision tasks

---

### **1.4 Enhanced TFN (Advanced Field Dynamics)**
**Location**: `tfn/model/tfn_enhanced.py` → `EnhancedTFNLayer`

**Architecture**:
```
Tokens → Field Projection → Field Interference → Field Propagation → Field Operators → Field Evolution → Field Sampling → Enhanced Tokens
```

**Key Features**:
- **Multi-stage processing**: 6-stage field pipeline
- **Field interference**: Token-centric attention mechanisms
- **Dynamic propagation**: Adaptive field evolution
- **Interaction operators**: Fractal, causal, meta operators
- **Physics constraints**: Energy conservation, symmetry

**Use Cases**: Research, advanced sequence modeling

---

## 🎯 **2. TASK-SPECIFIC TFN MODELS**

### **2.1 Classification Models**

#### **TFNClassifier** (`tfn/model/tfn_classifiers.py`)
```python
class TFNClassifier(nn.Module):
    """Simple classifier using TFN layers."""
```
- **Purpose**: Sequence classification
- **Architecture**: TFN layers + global pooling + MLP head
- **Tasks**: GLUE, text classification, sentiment analysis
- **Output**: [B, num_classes] logits

#### **TFNClassifier2D** (`tfn/model/tfn_2d.py`)
```python
class TFNClassifier2D(nn.Module):
    """A stack of 2-D TFN layers followed by pooling + MLP head."""
```
- **Purpose**: 2D sequence classification
- **Architecture**: 2D TFN layers + spatial pooling + MLP head
- **Tasks**: Image classification, spatial sequence classification
- **Output**: [B, num_classes] logits

#### **VisionTFN** (`tfn/scripts/train_cifar_tfn.py`)
```python
class VisionTFN(nn.Module):
    """Vision-specific TFN for image classification."""
```
- **Purpose**: Image classification with patch tokens
- **Architecture**: Patch embedding + TFNClassifier2D
- **Tasks**: CIFAR-10/100, ImageNet
- **Output**: [B, num_classes] logits

---

### **2.2 Regression Models**

#### **TFNRegressor** (`tfn/model/tfn_classifiers.py`)
```python
class TFNRegressor(nn.Module):
    """Simple regressor using TFN layers."""
```
- **Purpose**: Sequence regression
- **Architecture**: TFN layers + output projection
- **Tasks**: Time series forecasting, value prediction
- **Output**: [B, N, output_dim] or [B, output_dim]

#### **TFNTimeSeriesRegressor** (`tfn/model/tfn_regressors.py`)
```python
class TFNTimeSeriesRegressor(nn.Module):
    """TFN Regressor specifically designed for time series forecasting."""
```
- **Purpose**: Time series forecasting
- **Architecture**: TFN layers + forecasting head
- **Tasks**: ETT, Jena Climate, weather forecasting
- **Output**: [B, output_len] predictions

#### **TFNMultiStepRegressor** (`tfn/model/tfn_regressors.py`)
```python
class TFNMultiStepRegressor(nn.Module):
    """TFN Regressor for multi-step forecasting with sequence output."""
```
- **Purpose**: Multi-step sequence forecasting
- **Architecture**: TFN layers + sequence output projection
- **Tasks**: Long-term forecasting, sequence prediction
- **Output**: [B, output_len] sequence predictions

#### **TFNSequenceRegressor** (`tfn/model/tfn_regressors.py`)
```python
class TFNSequenceRegressor(nn.Module):
    """TFN Regressor for sequence-to-sequence regression."""
```
- **Purpose**: Sequence-to-sequence regression
- **Architecture**: TFN layers + sequence projection
- **Tasks**: Signal processing, sequence transformation
- **Output**: [B, N, output_dim] sequence outputs

---

### **2.3 Tagging Models**

#### **TFNTagger** (`tfn/scripts/train_ner_tfn.py`)
```python
class TFNTagger(nn.Module):
    """1-D TFN encoder + per-token linear head for NER."""
```
- **Purpose**: Named Entity Recognition (NER)
- **Architecture**: 1D TFN layers + per-token classification
- **Tasks**: CoNLL-2003, NER tasks
- **Output**: [B, L, num_tags] token-level predictions

#### **TFNTagger2D** (`tfn/scripts/train_ner_tfn.py`)
```python
class TFNTagger2D(nn.Module):
    """2-D TFN encoder for token-level tagging (row-major layout)."""
```
- **Purpose**: 2D NER with spatial layout
- **Architecture**: 2D TFN layers + per-token classification
- **Tasks**: Spatial NER, 2D sequence tagging
- **Output**: [B, L, num_tags] token-level predictions

---

### **2.4 Language Models**

#### **TFNLanguageModel** (`tfn/scripts/train_pg19.py`)
```python
class TFNLanguageModel(nn.Module):
    """TFN-based language model for PG-19."""
```
- **Purpose**: Language modeling
- **Architecture**: TFN layers + vocabulary projection
- **Tasks**: PG-19, text generation, language modeling
- **Output**: [B, L, vocab_size] token probabilities

#### **TFNSeqModel** (`tfn/model/seq_baselines.py`)
```python
class TFNSeqModel(nn.Module):
    """Simplified 1-D TFN sequence model for token-level prediction."""
```
- **Purpose**: Sequence-to-sequence modeling
- **Architecture**: Single TFN layer + vocabulary projection
- **Tasks**: Synthetic sequence tasks (copy, reverse)
- **Output**: [B, L, vocab_size] token predictions

---

## 🔬 **3. FIELD COMPONENTS**

### **3.1 Field Emission**

#### **ImageFieldEmitter** (`tfn/core/field_emitter.py`)
```python
class ImageFieldEmitter(nn.Module):
    """Emit continuous fields from image features."""
```
- **Purpose**: Convert images to continuous fields
- **Method**: Convolutional field generation
- **Use**: ImageTFN pipeline

#### **FieldProjector** (`tfn/core/field_projection.py`)
```python
class FieldProjector(nn.Module):
    """Projects token embeddings into continuous fields using kernels."""
```
- **Purpose**: Token → field projection
- **Kernels**: RBF, Compact, Fourier
- **Use**: 1D/2D TFN pipelines

---

### **3.2 Field Interference**

#### **ImageFieldInterference** (`tfn/core/field_interference_block.py`)
```python
class ImageFieldInterference(nn.Module):
    """Multi-head field interference for image processing."""
```
- **Purpose**: Field-level attention for images
- **Method**: Multi-head field interference
- **Use**: ImageTFN pipeline

#### **TokenFieldInterference** (`tfn/core/field_interference.py`)
```python
class TokenFieldInterference(nn.Module):
    """Token-centric field interference mechanisms."""
```
- **Purpose**: Token-centric field attention
- **Variants**: Standard, Causal, MultiScale, Physics
- **Use**: Enhanced TFN pipeline

---

### **3.3 Field Propagation**

#### **ImageFieldPropagator** (`tfn/core/field_propagator.py`)
```python
class ImageFieldPropagator(nn.Module):
    """PDE-inspired field propagation for images."""
```
- **Purpose**: Physics-inspired field evolution
- **Methods**: Diffusion, Wave, Schrödinger
- **Use**: ImageTFN pipeline

#### **DynamicFieldPropagator** (`tfn/core/dynamic_propagation.py`)
```python
class DynamicFieldPropagator(nn.Module):
    """Dynamic field propagation with adaptive evolution."""
```
- **Purpose**: Adaptive field evolution
- **Variants**: Standard, Adaptive, Causal
- **Use**: Enhanced TFN pipeline

---

### **3.4 Field Evolution**

#### **FieldEvolver** (`tfn/core/field_evolution.py`)
```python
class FieldEvolver(nn.Module):
    """Base class for field evolution strategies."""
```
- **Purpose**: Field evolution strategies
- **Variants**: CNN, Spectral, PDE
- **Use**: All TFN pipelines

#### **CNNFieldEvolver** (`tfn/core/field_evolution.py`)
```python
class CNNFieldEvolver(nn.Module):
    """CNN-based field evolution."""
```
- **Purpose**: Convolutional field evolution
- **Method**: 1D convolutions with learnable filters
- **Use**: Standard TFN pipelines

#### **SpectralFieldEvolver** (`tfn/core/field_evolution.py`)
```python
class SpectralFieldEvolver(nn.Module):
    """Spectral domain field evolution."""
```
- **Purpose**: Frequency domain evolution
- **Method**: FFT → filter → IFFT
- **Use**: Audio, signal processing

#### **PDEFieldEvolver** (`tfn/core/field_evolution.py`)
```python
class PDEFieldEvolver(nn.Module):
    """Physics-inspired PDE field evolution."""
```
- **Purpose**: Physics-constrained evolution
- **Methods**: Diffusion, Wave, Schrödinger equations
- **Use**: Scientific computing, physics tasks

---

### **3.5 Field Sampling**

#### **FieldSampler** (`tfn/core/field_sampling.py`)
```python
class FieldSampler(nn.Module):
    """Sample field values at specific positions."""
```
- **Purpose**: Field → token sampling
- **Methods**: Linear interpolation, nearest neighbor
- **Use**: All TFN pipelines

---

### **3.6 Field Interaction Operators**

#### **FieldInteractionOperators** (`tfn/core/interaction_operators.py`)
```python
class FieldInteractionOperators(nn.Module):
    """Field interaction operators for advanced dynamics."""
```
- **Purpose**: Advanced field interactions
- **Variants**: Standard, Fractal, Causal, Meta
- **Use**: Enhanced TFN pipeline

---

## 🎯 **4. TASK-SPECIFIC IMPLEMENTATIONS**

### **4.1 Text Classification**
- **Models**: TFNClassifier, TFNClassifier2D
- **Tasks**: GLUE, AG News, Yelp, IMDB
- **Scripts**: `train_glue_tfn.py`, `train_long.py`, `train_arxiv_tfn.py`

### **4.2 Time Series Forecasting**
- **Models**: TFNTimeSeriesRegressor, TFNMultiStepRegressor
- **Tasks**: ETT, Jena Climate, weather data
- **Scripts**: `train_climate_tfn.py`

### **4.3 Named Entity Recognition**
- **Models**: TFNTagger, TFNTagger2D
- **Tasks**: CoNLL-2003 NER
- **Scripts**: `train_ner_tfn.py`

### **4.4 Image Classification**
- **Models**: VisionTFN, ImageTFN
- **Tasks**: CIFAR-10/100, ImageNet
- **Scripts**: `train_cifar_tfn.py`, `train_tfn_pytorch.py`

### **4.5 Language Modeling**
- **Models**: TFNLanguageModel, TFNSeqModel
- **Tasks**: PG-19, synthetic sequence tasks
- **Scripts**: `train_pg19.py`, `train_synthetic_seq.py`

---

## 🔧 **5. CONFIGURATION OPTIONS**

### **5.1 Kernel Types**
- **RBF**: Radial Basis Function (default)
- **Compact**: Compact support kernel
- **Fourier**: Frequency domain kernel

### **5.2 Evolution Types**
- **CNN**: Convolutional Neural Network (default)
- **Spectral**: Spectral/Fourier domain
- **PDE**: Physics-based PDE evolution

### **5.3 Architecture Variants**
- **1D**: Linear sequence processing
- **2D**: Spatial lattice processing
- **Image**: Direct image processing
- **Enhanced**: Advanced field dynamics

---

## 📊 **6. PERFORMANCE CHARACTERISTICS**

| Model Type | Memory | Speed | Accuracy | Use Case |
|------------|--------|-------|----------|----------|
| **1D TFN** | Low | Fast | Good | Text, time series |
| **2D TFN** | Medium | Medium | Better | Spatial sequences |
| **ImageTFN** | High | Slow | Best | Images |
| **Enhanced TFN** | High | Slow | Research | Advanced tasks |

---

## 🎯 **7. USAGE RECOMMENDATIONS**

### **For Text Classification**
```python
# Use TFNClassifier for GLUE tasks
model = TFNClassifier(
    vocab_size=vocab_size,
    embed_dim=128,
    num_classes=num_classes,
    kernel_type="rbf",
    evolution_type="cnn"
)
```

### **For Time Series Forecasting**
```python
# Use TFNTimeSeriesRegressor for forecasting
model = TFNTimeSeriesRegressor(
    input_dim=input_dim,
    embed_dim=128,
    output_len=24,
    kernel_type="rbf",
    evolution_type="pde"
)
```

### **For Image Classification**
```python
# Use ImageTFN for direct image processing
model = ImageTFN(
    in_ch=3,
    num_classes=10
)
```

### **For NER**
```python
# Use TFNTagger for token-level tagging
model = TFNTagger(
    vocab_size=vocab_size,
    num_tags=num_tags,
    kernel_type="compact",
    evolution_type="cnn"
)
```

---

## 🚀 **8. CONCLUSION**

The TFN repository contains **24+ distinct variations** of Token Field Networks, covering:

- **4 Core Architectures**: 1D, 2D, Image, Enhanced
- **8 Task-Specific Models**: Classifiers, Regressors, Taggers, Language Models
- **12 Field Components**: Emission, Interference, Propagation, Evolution
- **Multiple Domains**: Text, Time Series, Images, Sequences

Each variation is optimized for specific tasks and domains, providing a comprehensive toolkit for field-based neural network research and applications! 🎯 