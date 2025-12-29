# Income Prediction Using Decision Tree - Project Results

## Executive Summary
Successfully built and optimized a Decision Tree classifier to predict income levels (≤$50K vs >$50K) from the Adult dataset with **85.09% accuracy**.

---

## 1. Data Overview

### Dataset Information
- **Total Records**: 32,561 entries
- **Total Columns**: 15 features + 1 target
- **Data Types**: 6 numeric (int64) + 9 categorical (object)
- **Memory Usage**: 3.7+ MB

### Key Features
- **Numeric Features**: age, fnlwgt, education.num, capital.gain, capital.loss, hours.per.week
- **Categorical Features**: workclass, education, marital.status, occupation, relationship, race, sex, native.country
- **Target Variable**: income (≤50K or >50K)

---

## 2. Data Preprocessing

### Missing Values Handling
- **Rows with '?' in workclass**: 1,836 (5% of data) → **Removed**
- **Rows with '?' in occupation**: 7 → **Removed**
- **Rows with '?' in native.country**: 556 → **Removed**
- **Final Clean Dataset**: 30,162 records

### Categorical Encoding
- Applied **LabelEncoder** from sklearn.preprocessing
- All 9 categorical variables converted to numeric format (0-indexed)
- Target variable converted to category type
- Result: All columns now int64 type (except target which is category)

### Train-Test Split
- **Training Set**: 21,113 samples (70%)
- **Test Set**: 9,049 samples (30%)
- **Random State**: 99 (for reproducibility)

---

## 3. Model Development & Evaluation

### Baseline Model (max_depth=5)
**Performance Metrics:**
```
Accuracy: 85.05%

Classification Report:
              precision    recall  f1-score   support
          0       0.86      0.95      0.91      6867
          1       0.78      0.52      0.63      2182

Confusion Matrix:
[[6553  314]
 [1039 1143]]
```

**Interpretation:**
- True Negatives (TN): 6,553 - Correctly predicted ≤50K
- False Positives (FP): 314 - Incorrectly predicted >50K
- False Negatives (FN): 1,039 - Missed >50K predictions
- True Positives (TP): 1,143 - Correctly predicted >50K

---

## 4. Hyperparameter Tuning Results

### A. Max Depth Tuning
**Range Tested**: 1 to 40
**Key Findings:**
- Optimal value: **max_depth = 9**
- Training accuracy increases continuously (overfitting pattern)
- Test accuracy peaks around max_depth = 9-10, then plateaus
- After depth 10, test accuracy remains stable (~85%)

**Graph Insight**: Clear overfitting visible - training accuracy continues rising while test accuracy plateaus

### B. Min Samples Leaf Tuning
**Range Tested**: 5 to 195 (step 20)
**Key Findings:**
- At low values (5-25): Training accuracy ~92%, Test accuracy ~84% (overfitting)
- At optimal values (45-100): Both converge to ~85%
- At high values (100+): Model becomes stable but slightly underfits
- **Best value**: min_samples_leaf = 45

**Graph Insight**: Lower values cause overfitting; convergence improves at higher values

### C. Min Samples Split Tuning
**Range Tested**: 5 to 195 (step 20)
**Key Findings:**
- At low values (5): Training ~97%, Test ~81% (severe overfitting)
- Increasing min_samples_split reduces overfitting gap
- At values >50: Gap becomes minimal (~1%)
- **Best value**: min_samples_split = 50+

**Graph Insight**: Higher values force simpler trees with better generalization

---

## 5. Grid Search Results

### Parameter Grid Tested
```
max_depth: [5, 10]
min_samples_leaf: [50, 100]
min_samples_split: [50, 100]
criterion: ["entropy", "gini"]
```

### Grid Search Results
**Best Parameters Found:**
- **Criterion**: gini
- **Max Depth**: 10
- **Min Samples Leaf**: 50
- **Min Samples Split**: 50

**Best Cross-Validation Accuracy**: 85.10%

---

## 6. Final Optimized Model

### Model Configuration
```python
DecisionTreeClassifier(
    criterion="gini",
    max_depth=10,
    min_samples_leaf=50,
    min_samples_split=50,
    random_state=100
)
```

### Final Model Performance
**Test Accuracy: 85.09%**

**Classification Report:**
```
              precision    recall  f1-score   support
          0       0.88      0.93      0.90      6867
          1       0.73      0.60      0.66      2182

accuracy                           0.85      9049
macro avg       0.81      0.77      0.78      9049
weighted avg    0.84      0.85      0.85      9049
```

**Confusion Matrix:**
```
[[6383  484]
 [ 865 1317]]
```

### Performance Analysis
| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Precision (≤50K)** | 0.88 | 88% of predicted ≤50K are correct |
| **Recall (≤50K)** | 0.93 | 93% of actual ≤50K correctly identified |
| **Precision (>50K)** | 0.73 | 73% of predicted >50K are correct |
| **Recall (>50K)** | 0.60 | 60% of actual >50K correctly identified |
| **Overall Accuracy** | 85.09% | 85% of all predictions correct |

---

## 7. Simplified Model (max_depth=3)

### Purpose
Create an interpretable tree while maintaining reasonable accuracy.

### Performance
**Accuracy: 83.93%**

**Classification Report:**
```
              precision    recall  f1-score   support
          0       0.85      0.96      0.90      6867
          1       0.77      0.47      0.59      2182

accuracy                           0.84      9049
macro avg       0.81      0.71      0.74      9049
weighted avg    0.83      0.84      0.82      9049
```

**Trade-off**: Lost 1.2% accuracy but gained interpretability (3 levels vs 10 levels)

---

## 8. Key Insights & Conclusions

### What Worked Well
✅ **Data Cleaning**: Successfully removed 5.6% missing values without significant impact  
✅ **Categorical Encoding**: LabelEncoder effectively converted categorical to numeric  
✅ **Hyperparameter Optimization**: Grid search identified optimal parameters  
✅ **Model Stability**: Good balance between training and test accuracy (no severe overfitting)  

### Model Strengths
✅ **Good at identifying low-income earners**: 93% recall for ≤50K class  
✅ **High precision for negatives**: 88% precision for ≤50K predictions  
✅ **Stable performance**: Similar performance across different test sets  

### Model Weaknesses
⚠️ **Lower recall for high-income earners**: Only 60% recall for >50K class  
⚠️ **Class imbalance**: Dataset has 3.15x more ≤50K samples than >50K  
⚠️ **Moderate precision for positives**: 73% precision for >50K predictions  

### Recommendations for Improvement
1. **Address Class Imbalance**: Use SMOTE or adjust class weights
2. **Feature Engineering**: Create interaction features (e.g., age × hours.per.week)
3. **Try Other Algorithms**: Random Forest, Gradient Boosting for better recall
4. **Feature Selection**: Identify most important features for prediction
5. **Threshold Tuning**: Adjust decision threshold for better recall/precision balance

---

## 9. Project Completion Checklist

✅ Data loading and exploration  
✅ Missing value detection and removal  
✅ Categorical variable encoding  
✅ Train-test split (70-30)  
✅ Baseline model building  
✅ Hyperparameter tuning (max_depth, min_samples_leaf, min_samples_split)  
✅ Grid search for optimal parameters  
✅ Final model training and evaluation  
✅ Simplified model for interpretability  
✅ Performance metrics and visualization  
✅ Results analysis and insights  

---

## 10. Technical Stack

| Component | Library | Version |
|-----------|---------|---------|
| Data Processing | pandas, numpy | Latest |
| Visualization | matplotlib, seaborn | Latest |
| ML Model | scikit-learn | 0.20+ |
| Hyperparameter Tuning | GridSearchCV (sklearn) | Latest |

---

## Final Statistics

| Metric | Value |
|--------|-------|
| **Dataset Size** | 30,162 records |
| **Feature Count** | 14 features |
| **Final Accuracy** | 85.09% |
| **Training Samples** | 21,113 |
| **Test Samples** | 9,049 |
| **Optimal Max Depth** | 10 |
| **Optimal Min Samples Leaf** | 50 |
| **Optimal Min Samples Split** | 50 |

---

## Conclusion

The Decision Tree classifier successfully predicts income levels with **85.09% accuracy** on the test set. The model performs particularly well at identifying individuals earning ≤50K (93% recall) but moderately at identifying high-income earners (60% recall). The hyperparameter tuning process revealed optimal settings that balance model complexity with generalization ability, resulting in minimal overfitting.

The project demonstrates complete ML workflow: data preprocessing, exploratory analysis, model building, hyperparameter optimization, and comprehensive evaluation.
