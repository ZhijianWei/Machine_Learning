# <div align="center">Warfarin Sensitivity Discrimination and Dosage Prediction Model
## <div align="center"><b><a href="[README.md](README.md)">English</a> | <a href=[READMEzh.md](READMEzh.md)>简体中文</a></b></div>

#### Author: Wei Zhijian (Nanjing Agricultural University), if you have any questions, please feel free to contact me at ``18151936092@163.com``📧
**The dataset is highly relevant to clinical pharmacy. The original data's indicators might be difficult for non-professionals to understand, but it is not necessary to fully comprehend them for practical machine learning. If there is enough interest, I will further improve the README.<br>
   If this set of algorithms is helpful to you, you can give this project a Star ⭐, or recommend it to your friends, thank you!😊**

## 💻Introduction
#### This is a real project from a hospital in Nanjing, including classification and regression tasks. The original data consists of various physical indicators and dosage statistics of warfarin for more than two hundred patients (normalized). The task of this project is to clean and analyze the data, select the best machine learning model for modeling and validation, compare it with models of different ensemble methods, and ultimately construct the optimal warfarin sensitivity discrimination model and warfarin dosage model.

* **Warfarin Sensitivity Discrimination Model**: Naive Bayes, Logistic Regression
  <br><br>
* **Drug Dosage Prediction Model**: RFR, SVR, XGBoost, Stacking (SOTA)
<br>

<br>

## ⚡Data Description

Includes various indicators and average warfarin dosage of 247 mechanical heart valve postoperative patients (normalized)

    1204all.xlsx

Data extracted separately related to sensitivity (normalized), this table can be directly used for drug sensitivity analysis model construction

    normalized_data.xlsx

This script can be used for SHAP value calculation and visualization (example based on RFR)

    SHAP_mapping(RF).py

## 👀Results Display

### * **Naive Bayes Warfarin Sensitivity Analysis Model**:
![img.png](NB_ROC.png)
![NB_accuracy.png](NB_accuracy.png)

### * **Logistic Regression Warfarin Sensitivity Analysis Model**:
![LR_ROC.jpg](LR_ROC.jpg)
![LR_accuracy.png](LR_accuracy.png)

<br><br><br>

### * **Warfarin Drug Dosage Prediction Model**:
Different variable SHAP value distribution:

![SHAP(RF).jpg](SHAP%28RF%29.jpg)
<br><br>
<br>
<br>
Comparison of predictive accuracy of different models (model selection and tuning are omitted)
![R2.png](R2.png)
![RMSE.png](RMSE.png)
![MAE.png](MAE.png)

<br><br><br>
