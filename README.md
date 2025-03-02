# 🤖 Machine Learning-Based Employee Attrition Prediction

## 📌 Introduction
Employee attrition is a critical concern for organizations, and **Machine Learning (ML)** can help predict which employees are at risk of leaving. This project explores various **ML models and hyperparameter tuning techniques** to find the best-performing model.

- 📂 **Dataset**: The data used in this project is the **IBM HR Analytics Employee Attrition Dataset**, downloaded from **Kaggle**.
- 🏆 **Goal**: Predict employee attrition based on various factors like salary, work-life balance, overtime, and job satisfaction.
- 🛠️ **Models Used**:
  - **RandomForestClassifier**
  - **Hyperparameter Tuning** with GridSearchCV & RandomizedSearchCV
  - **Bayesian Optimization** using Optuna
  - **Feature Importance Analysis**

---

## 🔄 Binary Encoding for Categorical Features
To convert categorical features into numerical values, I applied **One-Hot Encoding**:

```python
# Binary: Attrition, Gender, Over18, Overtime
data_frame["Attrition"] = data_frame["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)
data_frame["Gender"] = data_frame["Gender"].apply(lambda x: 1 if x == "Male" else 0)
data_frame["Over18"] = data_frame["Over18"].apply(lambda x: 1 if x == "Y" else 0)
data_frame["OverTime"] = data_frame["OverTime"].apply(lambda x: 1 if x == "Yes" else 0)
```

---

## 🔀 Using `pandas.get_dummies` for Certain Columns
I used `pd.get_dummies()` to transform categorical variables into multiple binary columns:

```python
# One-hot encoding categorical variables
data_frame = data_frame.join(pd.get_dummies(data_frame["BusinessTravel"])).drop("BusinessTravel", axis=1)
data_frame = data_frame.join(pd.get_dummies(data_frame["Department"], prefix="Department")).drop("Department", axis=1)
data_frame = data_frame.join(pd.get_dummies(data_frame["EducationField"], prefix="Education")).drop("EducationField", axis=1)
data_frame = data_frame.join(pd.get_dummies(data_frame["JobRole"], prefix="Role")).drop("JobRole", axis=1)
data_frame = data_frame.join(pd.get_dummies(data_frame["MaritalStatus"], prefix="Status")).drop("MaritalStatus", axis=1)
```

---

## 🌲 Model Training with `RandomForestClassifier`
I trained a **RandomForestClassifier** on the dataset:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

x, y = data_frame.drop("Attrition", axis=1), data_frame["Attrition"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier(n_jobs=-1)
model.fit(x_train, y_train)
```

---

## 🛠️ Hyperparameter Tuning with `GridSearchCV`
GridSearchCV is a powerful method for hyperparameter tuning as it systematically evaluates all possible parameter combinations. This approach ensures that the model achieves optimal performance by exhaustively searching predefined parameter values. While computationally expensive, it is ideal when accuracy is a higher priority than speed, making it a robust choice for refining RandomForestClassifier models.
I optimized the RandomForestClassifier by searching for the best hyperparameters:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

rf = RandomForestClassifier(n_jobs=-1)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring="accuracy")
grid_search.fit(x_train, y_train)

print("Best Parameters", grid_search.best_params_)
print("Test Set Accuracy", grid_search.best_estimator_.score(x_test, y_test))
```

---

## 🎯 Hyperparameter Tuning with `RandomizedSearchCV`
I used **RandomizedSearchCV** to explore a wider range of hyperparameters efficiently:

```python
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

param_dist = {
    "n_estimators": np.arange(50, 300, 50),
    "max_depth": [None, 10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6],
    "bootstrap": [True, False]
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1,
    scoring="accuracy"
)
random_search.fit(x_train, y_train)
print("Best Parameters", random_search.best_params_)
print("Test Accuracy: {:.4f}".format(random_search.best_estimator_.score(x_test, y_test)))
```

---

## 🔍 Bayesian Optimization with `Optuna`
Bayesian Optimization is a more efficient approach to hyperparameter tuning compared to GridSearchCV and RandomizedSearchCV. Unlike Grid Search, which exhaustively searches all possible parameter combinations, and Random Search, which selects parameters randomly, Bayesian Optimization builds a probabilistic model of the function being optimized. It strategically explores the search space, focusing on promising areas and reducing the number of evaluations needed to find optimal parameters. This results in faster convergence and improved model performance with fewer trials.
I leveraged **Bayesian Optimization** using `Optuna` to fine-tune the model:

```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
    max_depth = trial.suggest_int("max_depth", 10, 50, step=10)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 6)
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        n_jobs=-1
    )
    
    return cross_val_score(rf, x_train, y_train, cv=5, scoring="accuracy").mean()
```

---

## 📊 Feature Importance Analysis with Seaborn
Feature importance analysis is crucial in **model interpretability**, especially in HR analytics. It allows organizations to identify the most influential factors affecting employee attrition, enabling data-driven decision-making. Understanding these factors helps HR teams develop **targeted retention strategies**, optimize workplace policies, and enhance employee satisfaction while maintaining productivity.
I visualized feature importance to **understand key drivers of attrition**:

```python
import matplotlib.pyplot as plt
import numpy as np

feature_importances = best_rf_optuna.feature_importances_
feature_names = x_train.columns
sorted_indices = np.argsort(feature_importances)[::-1]
plt.figure(figsize=(14, 10))
colors = ["red" if i < 5 else "blue" for i in range(len(sorted_indices))]
plt.barh(range(len(sorted_indices)), feature_importances[sorted_indices], color=colors)
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Top Features Influencing Employee Attrition")
plt.xticks(fontsize=12)
plt.yticks(range(len(sorted_indices)), np.array(feature_names)[sorted_indices], fontsize=10)
plt.gca().invert_yaxis()
plt.show()
```

---

## 🏁 Conclusion
Through multiple ML techniques, **GridSearchCV (89.46%)** was the most accurate, while **Optuna & RandomizedSearchCV (88.44%)** found efficient models. **Overtime, Monthly Income, and Age** were the top factors driving attrition. 🚀

