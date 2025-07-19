# Enhanced TFN Training Integration Guide

## 🎯 **Complete Integration: Enhanced TFN with All Training Scripts**

The Enhanced TFN has been successfully integrated into all major training scripts, allowing you to train it on the same datasets as the base 1D TFN. Here's how to use it:

---

## 🚀 **Quick Start Examples**

### **1. GLUE Benchmark Training**
```bash
# Train Enhanced TFN on SST-2 with physics constraints
python tfn/scripts/train_glue_tfn.py \
    --task sst2 \
    --model enhanced_tfn \
    --interference_type physics \
    --evolution_type diffusion \
    --propagator_type adaptive \
    --operator_type fractal \
    --use_physics_constraints \
    --epochs 10

# Train Enhanced TFN on MRPC with standard interference
python tfn/scripts/train_glue_tfn.py \
    --task mrpc \
    --model enhanced_tfn \
    --interference_type standard \
    --evolution_type cnn \
    --propagator_type standard \
    --operator_type standard \
    --epochs 15
```

### **2. NER Training**
```bash
# Train Enhanced TFN on CoNLL-2003 NER
python tfn/scripts/train_ner_tfn.py \
    --tfn_type enhanced \
    --interference_type multiscale \
    --evolution_type wave \
    --propagator_type causal \
    --operator_type meta \
    --use_physics_constraints \
    --epochs 5
```

### **3. Synthetic Data Training**
```bash
# Train Enhanced TFN classifier on synthetic data
python tfn/scripts/train_tfn.py \
    --model enhanced_tfn_classifier \
    --interference_type physics \
    --evolution_type diffusion \
    --propagator_type adaptive \
    --operator_type fractal \
    --use_physics_constraints \
    --epochs 10
```

---

## 📊 **Available Training Scripts with Enhanced TFN**

### **1. GLUE Training Script** (`train_glue_tfn.py`)
**Supported Tasks**: SST-2, MRPC, QQP, QNLI, RTE, CoLA, STS-B, WNLI

**Enhanced TFN Options**:
- `--model enhanced_tfn`: Use Enhanced TFN instead of base TFN
- `--interference_type`: Field interference type (standard/causal/multiscale/physics)
- `--propagator_type`: Field propagator type (standard/adaptive/causal)
- `--operator_type`: Field operator type (standard/fractal/causal/meta)
- `--use_physics_constraints`: Enable physics constraints
- `--constraint_weight`: Weight for physics constraint loss

**Example**:
```bash
python tfn/scripts/train_glue_tfn.py \
    --task sst2 \
    --model enhanced_tfn \
    --interference_type physics \
    --evolution_type diffusion \
    --propagator_type adaptive \
    --operator_type fractal \
    --use_physics_constraints \
    --constraint_weight 0.1 \
    --epochs 10
```

### **2. NER Training Script** (`train_ner_tfn.py`)
**Supported Tasks**: CoNLL-2003 NER

**Enhanced TFN Options**:
- `--tfn_type enhanced`: Use Enhanced TFN instead of 1D/2D TFN
- All the same field interference, propagation, and operator options

**Example**:
```bash
python tfn/scripts/train_ner_tfn.py \
    --tfn_type enhanced \
    --interference_type multiscale \
    --evolution_type wave \
    --propagator_type causal \
    --operator_type meta \
    --use_physics_constraints \
    --epochs 5
```

### **3. Main Training Script** (`train_tfn.py`)
**Supported Tasks**: Synthetic classification and regression

**Enhanced TFN Options**:
- `--model enhanced_tfn_classifier`: Enhanced TFN for classification
- `--model enhanced_tfn_regressor`: Enhanced TFN for regression (not yet implemented)
- All the same field interference, propagation, and operator options

**Example**:
```bash
python tfn/scripts/train_tfn.py \
    --model enhanced_tfn_classifier \
    --interference_type physics \
    --evolution_type diffusion \
    --propagator_type adaptive \
    --operator_type fractal \
    --use_physics_constraints \
    --epochs 10
```

---

## 🔧 **Enhanced TFN Configuration Options**

### **Field Interference Types**
```bash
--interference_type {standard,causal,multiscale,physics}
```
- **standard**: Basic field interference
- **causal**: Time-series causality
- **multiscale**: Multi-scale field interactions
- **physics**: Physics-constrained interference

### **Field Propagation Types**
```bash
--propagator_type {standard,adaptive,causal}
```
- **standard**: Basic field propagation
- **adaptive**: Learnable evolution parameters
- **causal**: Causality-preserving propagation

### **Field Operator Types**
```bash
--operator_type {standard,fractal,causal,meta}
```
- **standard**: Basic field operations
- **fractal**: Fractal field transformations
- **causal**: Causality-preserving operations
- **meta**: Meta-learning field operations

### **Physics Constraints**
```bash
--use_physics_constraints  # Enable physics constraints
--constraint_weight 0.1    # Weight for constraint loss
```

---

## 🎯 **Model Comparison Examples**

### **Base 1D TFN vs Enhanced TFN**

**Base 1D TFN**:
```bash
python tfn/scripts/train_glue_tfn.py \
    --task sst2 \
    --model tfn \
    --evolution_type cnn \
    --epochs 10
```

**Enhanced TFN**:
```bash
python tfn/scripts/train_glue_tfn.py \
    --task sst2 \
    --model enhanced_tfn \
    --interference_type physics \
    --evolution_type diffusion \
    --propagator_type adaptive \
    --operator_type fractal \
    --use_physics_constraints \
    --epochs 10
```

### **Performance Comparison**
| Model | Attention Type | Complexity | Physics | Scalability |
|-------|---------------|------------|---------|-------------|
| **Base 1D TFN** | Field-based | O(N×M) | Basic | Linear |
| **Enhanced TFN** | Field-based + Interference | O(N×M) | Advanced | Linear |

---

## 🧪 **Testing Enhanced TFN Integration**

### **1. Test GLUE Integration**
```bash
# Quick test on SST-2
python tfn/scripts/train_glue_tfn.py \
    --task sst2 \
    --model enhanced_tfn \
    --interference_type standard \
    --epochs 2 \
    --batch_size 16
```

### **2. Test NER Integration**
```bash
# Quick test on CoNLL-2003
python tfn/scripts/train_ner_tfn.py \
    --tfn_type enhanced \
    --interference_type standard \
    --epochs 2 \
    --batch 16
```

### **3. Test Synthetic Integration**
```bash
# Quick test on synthetic data
python tfn/scripts/train_tfn.py \
    --model enhanced_tfn_classifier \
    --interference_type standard \
    --epochs 2 \
    --batch_size 16
```

---

## 📈 **Advanced Usage Examples**

### **1. Physics-Constrained Training**
```bash
python tfn/scripts/train_glue_tfn.py \
    --task sst2 \
    --model enhanced_tfn \
    --interference_type physics \
    --evolution_type diffusion \
    --propagator_type adaptive \
    --operator_type fractal \
    --use_physics_constraints \
    --constraint_weight 0.1 \
    --epochs 20
```

### **2. Multi-Scale Field Interactions**
```bash
python tfn/scripts/train_glue_tfn.py \
    --task mrpc \
    --model enhanced_tfn \
    --interference_type multiscale \
    --evolution_type spectral \
    --propagator_type adaptive \
    --operator_type meta \
    --epochs 15
```

### **3. Causal Field Processing**
```bash
python tfn/scripts/train_ner_tfn.py \
    --tfn_type enhanced \
    --interference_type causal \
    --evolution_type wave \
    --propagator_type causal \
    --operator_type causal \
    --epochs 10
```

---

## 🔍 **Key Features**

### **✅ Complete Integration**
- **All Training Scripts**: Enhanced TFN available in all major training scripts
- **All Datasets**: Same datasets as base 1D TFN
- **All Tasks**: Classification, regression, NER, GLUE tasks
- **Full CLI Configurability**: All parameters configurable via command line

### **✅ Advanced Field Processing**
- **6-Stage Pipeline**: Complete field-based processing
- **Physics Constraints**: Optional physics-based regularization
- **Multiple Interference Types**: Standard, causal, multiscale, physics
- **Adaptive Evolution**: Learnable field evolution parameters

### **✅ Research Platform**
- **Modular Design**: Each component independently configurable
- **Comprehensive Logging**: Detailed training progress and metrics
- **Physics Integration**: Physics-constrained training and inference
- **Performance Monitoring**: Full training and evaluation pipeline

---

## 🎯 **Conclusion**

The Enhanced TFN is now **fully integrated** into all training scripts and can be used on **all the same datasets** as the base 1D TFN:

✅ **GLUE Benchmark**: All 8 GLUE tasks supported
✅ **NER Tasks**: CoNLL-2003 NER supported  
✅ **Synthetic Data**: Classification and regression tasks
✅ **Full CLI Configurability**: All Enhanced TFN parameters configurable
✅ **Physics Integration**: Optional physics constraints for regularization
✅ **Research Platform**: Comprehensive field-based neural network toolkit

**Usage**: Simply add `--model enhanced_tfn` (GLUE) or `--tfn_type enhanced` (NER) to any existing training command to use Enhanced TFN instead of the base TFN! 🚀 