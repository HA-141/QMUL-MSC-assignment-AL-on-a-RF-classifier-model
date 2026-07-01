# Active Learning for Random Forest Classification

Active-learning-driven Random Forest classifiers applied to two biomedical prediction tasks, evaluating how much labeled data is actually needed to reach strong performance.

## Problem & Motivation
Labeling data is expensive, especially in biomedical contexts where labels require lab work or expert annotation. This project investigates whether an active learning loop — where the model selectively queries the most informative samples rather than training on everything at once — can reach strong classification performance using a fraction of the full labeled dataset, tested across two different domains:

1. **Blood-brain barrier (BBB) permeability** prediction for drug candidates
2. **Breast cancer (BC)** identification from patient data

## Approach
- Trained separate Random Forest classifiers per dataset, starting from just 5 labeled samples
- Used **least confidence uncertainty sampling** to select the next 20 samples to label at each iteration, stratified by class label proportion
- Ran the loop across 5 random seeds (1, 10, 42, 50, 100) to test robustness to initialization
- Training stopped once the labeled set reached 80% of the total dataset
- Tracked MCC and F1 score at every iteration, plus feature importance, comparing the model's first-5-sample performance against the mean of 5 randomly-selected stratified samples, with t-tests to assess significance

**Versions:** Python 3.13.0 · Pandas 2.2.3 · NumPy 2.1.2 · Matplotlib 3.9.2 · Seaborn 0.13.2 · Scikit-learn 1.5.2 · SciPy 1.14.1 · RDKit 2024.09.5

## Results

### Exploratory Data Analysis
Both datasets showed class imbalance: benign samples outnumbered malignant in the BC dataset, and BBB-permeable (BBB+) samples outnumbered impermeable (BBB-) samples — both risking a model bias toward the majority class.

| Blood-Brain Barrier | Breast Cancer |
|---|---|
| ![BBB label distribution](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/raw/main/NB_Images/BBB_Label_dist.png) | ![Breast Cancer label distribution](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/raw/main/NB_Images/Breast_Cancer_Label_dist.png) |

Feature-level analysis showed a stark contrast between datasets: in the BC dataset, features like `perimeter_worst`, `area_worst`, and `concave points_worst` showed clearly separable distributions between classes, while in the BBBP dataset most features clustered near zero with minimal separation between BBB+ and BBB- samples.

![BBB feature distribution](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/raw/main/NB_Images/BBB_Feature_dist.png)
![Breast Cancer feature distribution](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/raw/main/NB_Images/Breast_Cancer_Feature_dist.png)

PCA reinforced this: the BC dataset formed two distinguishable clusters (a dense benign cluster and a more sparsely spaced malignant cluster), while the BBBP dataset showed both classes congregating into a single group with no clear decision boundary at low dimensions.

| BBB PCA | Breast Cancer PCA |
|---|---|
| ![BBB PCA](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/raw/main/NB_Images/BBB_PCA.png) | ![Breast Cancer PCA](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/raw/main/NB_Images/Breast_Cancer_PCA.png) |

### Active Learning Performance
For the **BC dataset**, MCC and F1 scores rose sharply and plateaued after the 5th active learning iteration, with no significant difference between the first-5-sample scores and the mean random-stratified-sample scores (t-test not significant for either metric) — indicating the first 5 samples were already informative enough for the model to learn key classification features.

For the **BBBP dataset**, both scores rose sharply early on but never reached 1, and were noticeably more erratic — MCC dropped sharply between iterations 15-20. Unlike the BC dataset, the t-test showed a **significant decrease** in MCC between the first-5 and random-mean samples (p = 0.0008), though F1 showed no significant difference.

![BBB MCC](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/raw/main/NB_Images/Blood_Brain_Barrier_MCC.png)
![BBB F1](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/raw/main/NB_Images/Blood_Brain_Barrier_F1.png)
![Breast Cancer MCC](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/raw/main/NB_Images/Breast_Cancer_MCC.png)
![Breast Cancer F1](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/raw/main/NB_Images/Breast_Cancer_F1.png)

**Cross-dataset comparison** confirmed the BC dataset performed significantly better overall: steeper score increases, earlier plateau, and lower variance than BBBP — backed by unpaired t-tests showing significant differences in both MCC (p = 3.7e-6) and F1 (p = 0.0077) between the two datasets' first-5-sample performance, and even stronger significance (p < 0.001) when comparing mean random-sample performance.

![MCC comparison](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/raw/main/NB_Images/MCC_Comparison.png)
![F1 comparison](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/raw/main/NB_Images/F1_Comparison.png)

Feature importance analysis explained the gap: the BC dataset showed a sharp importance drop-off after its top 5 features (matching the clearly-separable features from Figure 2), while the BBBP dataset showed only one major important feature with a much smaller overall importance scale — consistent with its weaker feature separability.

![BBB top features](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/raw/main/NB_Images/Blood_Brain_Barrier_Top_Features.png)
![Breast Cancer top features](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/raw/main/NB_Images/Breast_Cancer_Top_Features.png)

## Outcome
The active learning model performed well on both datasets but significantly better on the BC dataset, which benefited from clearly separated, high-importance features and well-defined decision boundaries — producing a sharp accuracy increase followed by an early plateau. The BBBP dataset also reached strong classification accuracy, but its lack of highly separable features resulted in a more gradual, less stable performance curve and a statistically significant performance gap versus BC across nearly every comparison tested. This demonstrates that active learning's benefit is data-dependent: it works best when informative, well-separated features already exist for the model to exploit early on.

## Setup & Usage

Open `Code.ipynb` in Jupyter and run all cells. Requires the package versions listed above.

## License
BSD-3-Clause
