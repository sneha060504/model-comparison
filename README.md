#  Assignment 2: Model Comparison

##  Objective
The objective of this assignment is to compare different supervised machine learning algorithms and evaluate their performance on a dataset.

---

##  Resources Used

- Scikit-learn: https://scikit-learn.org/stable/supervised_learning.html  
- NumPy: https://numpy.org/doc/stable/  
- Gradient Descent Cheatsheet: https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html  
- Iris Dataset: https://archive.ics.uci.edu/ml/datasets/iris  
- Titanic Dataset: https://www.kaggle.com/c/titanic  

---

##  Dataset Used

### Iris Dataset
- Total Samples: 150  
- Features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width  
- Target: Flower species

---

##  Algorithms Implemented

1. Logistic Regression  
2. K-Nearest Neighbors (KNN)  
3. Decision Tree  

---

##  Approach

1. Loaded dataset using Scikit-learn  
2. Performed preprocessing:
   - Train-test split (80/20)
   - Feature scaling using StandardScaler  
3. Trained models:
   - Logistic Regression
   - KNN (k=5)
   - Decision Tree (max_depth=3)  
4. Evaluated using accuracy score  
5. Compared performance  

---

##  Accuracy Comparison

| Model                  | Accuracy |
|----------------------|----------|
| Logistic Regression  | ~0.96    |
| KNN (k=5)            | ~0.97    |
| Decision Tree        | ~0.95    |

*(Results may vary slightly depending on execution)*

---

##  Best Performing Model

**K-Nearest Neighbors (KNN)**

### Reason:
- Works well on small datasets  
- Captures local patterns effectively  
- No assumptions about data distribution  

---

##  Difficulties Faced

- Choosing optimal K value in KNN  
- Understanding gradient descent concept  
- Scaling issues affecting KNN performance  
- Overfitting in Decision Tree  

---

##  Resolutions

- Applied StandardScaler  
- Tested multiple K values  
- Limited depth of Decision Tree  
- Referred documentation and ML cheatsheets  

---

##  Results

- All models achieved high accuracy (>95%)  
- KNN performed best  
- Logistic Regression was stable  
- Decision Tree was interpretable  

---

##  Learnings

- Importance of preprocessing  
- Role of hyperparameter tuning  
- Model performance varies with dataset  
- Simpler models can perform effectively  

---

##  How to Run

### 1. Clone Repository
```bash
git clone https://github.com/your-username/model-comparison.git
cd model-comparison
