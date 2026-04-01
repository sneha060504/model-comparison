# Assignment 2: Model Comparison (Iris Dataset)


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


data = load_iris()
X = data.data
y = data.target

print("Dataset loaded successfully!")
print("Features shape:", X.shape)
print("Target shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


log_reg = LogisticRegression(max_iter=200)
knn = KNeighborsClassifier(n_neighbors=5)
decision_tree = DecisionTreeClassifier(max_depth=3, random_state=42)


log_reg.fit(X_train, y_train)
knn.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)


lr_pred = log_reg.predict(X_test)
knn_pred = knn.predict(X_test)
dt_pred = decision_tree.predict(X_test)


lr_acc = accuracy_score(y_test, lr_pred)
knn_acc = accuracy_score(y_test, knn_pred)
dt_acc = accuracy_score(y_test, dt_pred)


print("\n=== Accuracy Comparison ===")
print(f"Logistic Regression Accuracy: {lr_acc:.2f}")
print(f"KNN Accuracy: {knn_acc:.2f}")
print(f"Decision Tree Accuracy: {dt_acc:.2f}")


best_model = max(
    [("Logistic Regression", lr_acc),
     ("KNN", knn_acc),
     ("Decision Tree", dt_acc)],
    key=lambda x: x[1]
)

print(f"\nBest Performing Model: {best_model[0]} with accuracy {best_model[1]:.2f}")


print("\n=== Classification Report ===")
if best_model[0] == "Logistic Regression":
    print(classification_report(y_test, lr_pred))
elif best_model[0] == "KNN":
    print(classification_report(y_test, knn_pred))
else:
    print(classification_report(y_test, dt_pred))