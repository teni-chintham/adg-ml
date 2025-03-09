import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.svm import SVC
import time

# Load dataset
data = pd.read_csv(r"/Users/tenichintham/adg-ml/data/Titanic-Dataset.csv")

data = data.drop(["Name", "Ticket", "Cabin"], axis=1)

categorical_cols = ["Sex", "Embarked"] 
for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

x = data.drop(['Survived'], axis=1).values
y = data['Survived'].values

mask = ~np.isnan(x).any(axis=1)
x = x[mask]
y = y[mask]

shape = data.shape

print("Here is a Description Table of your data:\n\n", data.describe())

print("\nDropping duplicates and handling null values...")
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

print("\nUpdated dataset info:\n")
print("Null values per column:\n", data.isnull().sum())
print("\nTotal duplicate rows removed:", data.duplicated().sum())


temp = "Survived"


X = data.drop(columns=[temp]) 
y = data[temp] 


#HEATMAP
data_x = data.select_dtypes(include=["number"])
plt.figure(figsize=(20, 8))
sns.heatmap(data_x.corr(), annot=True)
plt.show()


#BAR CORR AGAINST SURVIVED
c = ['black', 'purple', 'orange', 'grey']
plt.figure(figsize=(12, 8))
data_x.corrwith(data_x[temp]).plot.bar(fontsize=15, title='Survivor Correlation', rot=45, grid=True, color=c)
plt.show()

#PCLASS VS SURVIVAL
data.groupby("Pclass")[temp].mean().plot(kind='bar')
plt.title(f"Bar Plot of Passenger Class against Survivors")
print(plt.show())

pclass_survival = data.groupby("Pclass")["Survived"].mean()
plt.figure(figsize=(15, 6))
plt.bar(pclass_survival.index, pclass_survival.values, color='skyblue', label="Survival Rate")
z = np.polyfit(pclass_survival.index, pclass_survival.values, 2)
p = np.poly1d(z)
plt.plot(pclass_survival.index, p(pclass_survival.index), color='red', linestyle='-', linewidth=2, label="Trendline")
plt.title("Bar Plot of Passenger Class vs Survivors with Trendline")
print(plt.show())

sns.scatterplot(data=data, x="Pclass",y=temp, hue=temp)
plt.title(f"Scatter Plot of Passenger Class against Survivors")
print(plt.show())

sns.regplot(data=data, x="Pclass",y=temp)
plt.title(f"Regression Plot of Passenger Class against Survivors")
print(plt.show())









#AGE VS SURVIVAL
data.groupby("Age")[temp].mean().plot(kind='bar')
plt.title(f"Bar Plot of Age against Survivors")
print(plt.show())

age_survival = data.groupby("Age")["Survived"].mean()
plt.figure(figsize=(15, 6))
plt.bar(age_survival.index, age_survival.values, color='skyblue', label="Survival Rate")
z = np.polyfit(age_survival.index, age_survival.values, 2)
p = np.poly1d(z)
plt.plot(age_survival.index, p(age_survival.index), color='red', linestyle='-', linewidth=2, label="Trendline")
plt.title("Bar Plot of Age vs Survivors with Trendline")
print(plt.show())

sns.scatterplot(data=data, x="Age",y=temp, hue=temp)
plt.title(f"Scatter Plot of Age against Survivors")
print(plt.show())

sns.regplot(data=data, x="Age",y=temp)
plt.title(f"Regression Plot of Age against Survivors")
print(plt.show())









#SEX VS SURVIVAL
data.groupby("Sex")[temp].mean().plot(kind='bar')
plt.title(f"Bar Plot of Gender against Survivors")
print(plt.show())

sns.scatterplot(data=data, x="Sex",y=temp, hue=temp)
plt.title(f"Scatter Plot of Gender against Survivors")
print(plt.show())









#SIBSP VS SURVIVAL
data.groupby("SibSp")[temp].mean().plot(kind='bar')
plt.title(f"Bar Plot of SibSp against Survivors")
print(plt.show())

SibSp_survival = data.groupby("SibSp")["Survived"].mean()
plt.figure(figsize=(15, 6))
plt.bar(SibSp_survival.index, SibSp_survival.values, color='skyblue', label="Survival Rate")
z = np.polyfit(SibSp_survival.index, SibSp_survival.values, 2)
p = np.poly1d(z)
plt.plot(SibSp_survival.index, p(SibSp_survival.index), color='red', linestyle='-', linewidth=2, label="Trendline")
plt.title("Bar Plot of SibSp vs Survivors with Trendline")
print(plt.show())

sns.scatterplot(data=data, x="SibSp",y=temp, hue=temp)
plt.title(f"Scatter Plot of SibSp against Survivors")
print(plt.show())

sns.regplot(data=data, x="SibSp",y=temp)
plt.title(f"Regression Plot of SibSp against Survivors")
print(plt.show())









#FARE VS SURVIVAL
data.groupby("Fare")[temp].mean().plot(kind='bar')
plt.title(f"Bar Plot of Fare against Survivors")
print(plt.show())

Fare_survival = data.groupby("Fare")["Survived"].mean()
plt.figure(figsize=(15, 6))
plt.bar(Fare_survival.index, Fare_survival.values, color='skyblue', label="Survival Rate")
z = np.polyfit(Fare_survival.index, Fare_survival.values, 2)
p = np.poly1d(z)
plt.plot(Fare_survival.index, p(Fare_survival.index), color='red', linestyle='-', linewidth=2, label="Trendline")
plt.title("Bar Plot of Fare vs Survivors with Trendline")
print(plt.show())

sns.scatterplot(data=data, x="Fare",y=temp, hue=temp)
plt.title(f"Scatter Plot of Fare against Survivors")
print(plt.show())

sns.regplot(data=data, x="Fare",y=temp)
plt.title(f"Regression Plot of Fare against Survivors")
print(plt.show())









def preprocess_data(x, y):
    smote = SMOTE(sampling_strategy="auto", random_state=0)  
    x_os, y_os = smote.fit_resample(x, y)
    return train_test_split(x_os, y_os, test_size=0.3, random_state=0)

x_train, x_test, y_train, y_test = preprocess_data(x, y)


scaler = StandardScaler().fit(x_train)
x_train_sc = scaler.transform(x_train)
x_test_sc = scaler.transform(x_test)

# Logistic Regression with Cross-Validation
def logisticreg(x_train, x_test, y_train, y_test, sol):
    model = LogisticRegression(solver=sol, random_state=0)
    model.fit(x_train, y_train)

    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(model, x_train, y_train, cv=kfold)

    y_pred = model.predict(x_test)

    return {
        "Solver": sol,
        "CV Accuracy": results.mean() * 100,
        "Test Accuracy": accuracy_score(y_test, y_pred) * 100,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Model": model
    }

# XGBoost Classifier
def xgboost_class(x_train, x_test, y_train, y_test, estimators):
    model = XGBClassifier(n_estimators=estimators, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=sum(y_train == 0) / sum(y_train == 1), 
                          random_state=0, n_jobs=-1)
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "Precision": precision_score(y_test, y_pred) * 100,
        "Recall": recall_score(y_test, y_pred) * 100,
        "F1 Score": f1_score(y_test, y_pred) * 100,
        "Model": model
    }

# SVM Classifier
def svc_class(x_train, x_test, y_train, y_test, ker):
    model = SVC(kernel=ker, random_state=0)
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "Precision": precision_score(y_test, y_pred) * 100,
        "Recall": recall_score(y_test, y_pred) * 100,
        "F1 Score": f1_score(y_test, y_pred) * 100,
        "Model": model
    }

start_time = time.time()










# Best Logistic Regression Model Selection
logisticsol = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
best_logistic = max(
    (logisticreg(x_train_sc, x_test_sc, y_train, y_test, sol) for sol in logisticsol), 
    key=lambda m: m["F1 Score"]
)

print(f"Best Logistic Regression Model: {best_logistic['Solver']}")
print(f"Cross-Validation Accuracy: {best_logistic['CV Accuracy']:.2f}%")
print(f"Test Accuracy: {best_logistic['Test Accuracy']:.2f}%")
print(f"Precision: {best_logistic['Precision']:.2f}")
print(f"Recall: {best_logistic['Recall']:.2f}")
print(f"F1 Score: {best_logistic['F1 Score']:.2f}\n\n")

# Best XGBoost Model Selection
best_xgboost = max(
    (xgboost_class(x_train, x_test, y_train, y_test, est) for est in range(1, 11, 1)),
    key=lambda m: m["F1 Score"]
)

print(f"Best XGBoost Model (n_estimators={best_xgboost['Model'].n_estimators}):")
print(f"Accuracy: {best_xgboost['Accuracy']:.2f}%")
print(f"Precision: {best_xgboost['Precision']:.2f}")
print(f"Recall: {best_xgboost['Recall']:.2f}")
print(f"F1 Score: {best_xgboost['F1 Score']:.2f}\n\n")

# Best SVC Model Selection
svr_kernels = ["linear", "poly", "rbf", "sigmoid"]
best_svc = max(
    (svc_class(x_train_sc, x_test_sc, y_train, y_test, kernel) for kernel in svr_kernels), 
    key=lambda m: m["F1 Score"]
)

print(f"Best SVC Model (Kernel='{best_svc['Model'].kernel}'):")
print(f"Accuracy: {best_svc['Accuracy']:.2f}%")
print(f"Precision: {best_svc['Precision']:.2f}")
print(f"Recall: {best_svc['Recall']:.2f}")
print(f"F1 Score: {best_svc['F1 Score']:.2f}")

end_time = time.time()
print(f"\nTotal Processing Time: {end_time - start_time:.2f} seconds")