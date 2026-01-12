*Contains a RF classifier models using active learning that predicts:*

1) Blood bain barrier permeability for select drugs
2) Breast Cancer identification of a patient

Version of python and supporting libraries:
Python version: 3.13.0 
Pandas library version: 2.2.3
NumPy library version: 2.1.2
Matplotlib library version: 3.9.2
Seaborn library version: 0.13.2
Scikit-learn library version: 1.5.2
SciPy library version: 1.14.1
RDKit library version: 2024.09.5


## Sections of the code for both datasets  include:

### Basic EDA and data cleaning
### Advanced EDA
- *Label distribution*
![Label dist for Blood brain barrier permeability](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/blob/main/NB_Images/BBB_Label_dist.png)
  ![Label dist for Breast cancer](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/blob/main/NB_Images/Breast_Cancer_Label_dist.png)
- *Feature distribution*

  ![Feature dist for Blood brain barrier permeability](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/blob/main/NB_Images/BBB_Feature_dist.png)

  ![Feature dist for Breast Cancer](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/blob/main/NB_Images/Breast_Cancer_Feature_dist.png)
- *PCA (2 components)*
  ![PCA for Blood brain barrier permeability](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/blob/main/NB_Images/BBB_PCA.png)
  ![PCA for Breast cancer](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/blob/main/NB_Images/Breast_Cancer_PCA.png)
### Active Learning Random forest classifier training
- *Training samples start with first five samples, then five randomly selected samples stratified by class label proportion*
- *Random states for random sample selection: 1, 10, 42, 50, 100*
- *Least confidence uncertainty sampling (20 least confident samples)*
- *Acquire MCC and F1 scores and the model's feature importance.*
- *Tranining loop stops when training size is more than 80% of the total sample size*
### Individual statistical analysis
### Feature importance extraction
- *For Blood Brain Barrier Permeability: *
  ![Blood Brain Barrier MCC scores](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/blob/main/NB_Images/Blood_Brain_Barrier_MCC.png)
  ![Blood Brain Barrier F1 scores](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/blob/main/NB_Images/Blood_Brain_Barrier_F1.png)
  ![Blood Brain Barrier top feature importances](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/blob/main/NB_Images/Blood_Brain_Barrier_Top_Features.png)
- *For Breast Cancer:*
  ![Breast Cancer MCC scores](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/blob/main/NB_Images/Breast_Cancer_MCC.png)
  ![Breast Cancer F1 scores](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/blob/main/NB_Images/Breast_Cancer_F1.png)
  ![Breast Cancer top feature importances](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/blob/main/NB_Images/Breast_Cancer_Top_Features.png)
*Comparative analysis for both datasets:*
  ![MCC score comparison](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/blob/main/NB_Images/MCC_Comparison.png)
  ![F1 score comparison](https://github.com/HA-141/QMUL-MSC-assignment-AL-on-a-RF-classifier-model/blob/main/NB_Images/F1_Comparison.png)
