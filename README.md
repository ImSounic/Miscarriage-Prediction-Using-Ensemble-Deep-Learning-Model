# Pregnancy Complication Prediction Using Ensemble Deep Learning**

## **: All files will be created once you run the two python notebooks from scratch. All notebooks were downloaded from Google Colab which was used to implement and test the code. TabNet SHAP Results were inaccurate due to less compute being available for calculating hence was not involved in the report,  Also due to unavailability of high computational power, no of samples in SHAP Analysis were decreased across all models.



## Project Overview

This project develops an ensemble deep learning approach for predicting pregnancy complications, specifically miscarriages, using the National Family Health Survey-5 (NFHS-5) dataset. The system combines three specialized tabular deep learning architectures and provides interpretable predictions through SHAP (SHapley Additive exPlanations) analysis.


## Setup and Installation

### Environment Setup

This project requires Python 3.8+ and the following key libraries:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- PyTorch (1.9+)
- tab_transformer_pytorch 
- pytorch_tabnet
- scikit-learn
- imbalanced-learn
- shap
- pandas
- numpy
- matplotlib
- seaborn

### Hardware Requirements

The model training and SHAP analysis were performed on hardware with the following specifications:
- GPU: NVIDIA A100 (40GB VRAM)
- Memory: 32GB RAM
- Storage: 100GB available disk space

For inference only, the hardware requirements are less demanding.

## Implementation Details

### Pipeline Overview

The implementation follows these key steps:

1. **Data Preprocessing**: Cleaning, encoding categorical variables, handling missing values, and feature scaling
2. **Feature Selection**: LASSO-based feature selection to reduce dimensionality while preserving predictive power
3. **Class Balancing**: A novel NearSMOTE technique combining SMOTE and NearMiss for handling class imbalance
4. **Model Training**: Implementation of three deep learning architectures
   - TabTransformer: Contextual embeddings with self-attention layers
   - FT-Transformer: Feature tokenization with transformer blocks
   - TabNet: Sequential feature selection via attention masks
5. **Ensemble Integration**: Weighted ensemble with prediction averaging
6. **Model Explainability**: SHAP analysis for interpretable predictions

### Key Components

#### NearSMOTE Implementation

The NearSMOTE algorithm represents a novel contribution of this project, combining Synthetic Minority Over-sampling Technique (SMOTE) with selective undersampling via NearMiss to address class imbalance while preserving decision boundary clarity.

```python
# Pipeline implementation of NearSMOTE
nearsmote = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('nearmiss', NearMiss(version=1, n_neighbors=3)),
    ('smotetomek', SMOTETomek(random_state=42))
])
```

#### Model Architecture

The TabTransformer model uses contextual embeddings for categorical features:

```python
model_tab = TabTransformer(
    categories=cat_cardinalities,
    num_continuous=X_train_num.shape[1],
    dim=64, dim_out=1, depth=4, heads=4,
    attn_dropout=0.1, ff_dropout=0.1
).to(device)
```

The FT-Transformer integrates numerical and categorical features through a tokenized approach:

```python
model_ft = FTTransformer(
    categories=cat_cardinalities,
    num_continuous=X_train_num.shape[1],
    dim=32, dim_out=1, depth=2, heads=2,
    attn_dropout=0.2, ff_dropout=0.2
).to(device)
```

TabNet enables interpretable feature selection through sequential attention:

```python
model_tabnet = TabNetClassifier(
    n_d=16, n_a=16, n_steps=4, gamma=1.3,
    cat_idxs=[], cat_dims=[], cat_emb_dim=1,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=1e-3),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='sparsemax'
)
```

#### Ensemble Integration

The ensemble combines predictions from all three models:

```python
ensemble_probs = (tab_preds + ft_preds + tn_preds) / 3
ensemble_preds = (ensemble_probs > 0.5).astype(int)
```


### Using Pre-trained Models

To load and use pre-trained models:

```python
import torch
import joblib
from tab_transformer_pytorch import TabTransformer
from pytorch_tabnet.tab_model import TabNetClassifier

# Load feature names and cardinalities
feature_names = joblib.load('models/feature_names.pkl')
cat_cardinalities = joblib.load('models/cat_cardinalities.pkl')

# Load TabTransformer
model_tab = TabTransformer(
    categories=cat_cardinalities,
    num_continuous=len(num_feature_names),
    dim=64, dim_out=1, depth=4, heads=4,
    attn_dropout=0.1, ff_dropout=0.1
)
model_tab.load_state_dict(torch.load('models/tabtransformer_model.pth'))

# Load TabNet
model_tabnet = TabNetClassifier()
model_tabnet.load_model('models/tabnet_model')

```

## Results

The ensemble model demonstrates superior performance compared to individual models:

| Model          | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------|----------|-----------|--------|----------|---------|
| TabTransformer | 0.8308   | 0.8426    | 0.8135 | 0.8278   | 0.9080  |
| FT-Transformer | 0.8420   | 0.7651    | 0.9871 | 0.8620   | 0.8977  |
| TabNet         | 0.6782   | 0.6722    | 0.6957 | 0.6838   | 0.7521  |
| Ensemble       | 0.8557   | 0.8270    | 0.8996 | 0.8618   | 0.9311  |

SHAP analysis revealed the following top predictive features:
1. Prenatal care (protective)
2. Total children born (risk factor)
3. Respondent health check (protective)
4. Pregnancy parasitic drug administration (protective)
5. Private delivery place (protective)


## Acknowledgments

This project uses the following third-party libraries:
- [tab_transformer_pytorch](https://github.com/lucidrains/tab-transformer-pytorch) (MIT License)
- [pytorch_tabnet](https://github.com/dreamquark-ai/tabnet) (MIT License)
- [SHAP](https://github.com/slundberg/shap) (MIT License)
- [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) (MIT License)

## Contact

Soa39@aber.ac.uk

## License

This project is licensed under the MIT License - see the LICENSE file for details.
