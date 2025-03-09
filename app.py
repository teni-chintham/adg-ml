from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import scipy as sc
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
import matplotlib
import matplotlib.pyplot as plt
import pickle
import joblib
import os
import matplotlib
from matplotlib.cm import viridis


plt.switch_backend('Agg')
matplotlib.use('Agg')

app = Flask(__name__)

data = pd.read_csv(r"/Users/tenichintham/adg-ml/data/Titanic-Dataset.csv")
data = data.drop(["Name", "Ticket", "Cabin"], axis=1)
categorical_cols = ["Sex", "Embarked"]
for col in categorical_cols:
    data[col] = data[col].astype(str).fillna("Unknown")
    data[col] = data[col].astype("category").cat.codes

data.dropna(inplace=True) 

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/task-menu", methods=["GET", "POST"])
def link():
    return render_template("linkr.html")

@app.route("/task-1-1", methods=["GET", "POST"])
def task_1_1_():
    return render_template("task1_1.html")


@app.route('/heatmap')
def heatmap():
    plt.figure(figsize=(15, 8))
    sns.heatmap(data.corr(), annot=True, cmap='viridis')
    plt.title("Correlation Heatmap")
    static_dir = os.path.join(os.getcwd(), 'static')
    heatmap_path = os.path.join(static_dir, 'heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('heatmap.html', heatmap_url='/static/images/heatmap.png')


@app.route('/correlation-bar')
def correlation_bar():
    temp = 'Survived'
    colors = ['black', 'purple', 'orange', 'grey']
    plt.figure(figsize=(15, 8))
    data.corrwith(data[temp]).plot.bar(fontsize=15, title='Survivor Correlation', rot=45, grid=True, color=colors)
    static_dir = os.path.join(os.getcwd(), 'static')
    chart_path = os.path.join(static_dir, 'correlation_bar.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('correlation_bar.html', chart_url='/static/images/correlation_bar.png')


@app.route("/link-fare", methods=["GET", "POST"])
def link_fare():
    return render_template("link_fare_survival.html")


@app.route('/fare-survival-bar')
def fare_survival_bar():
    temp = 'Survived'
    data[temp] = pd.to_numeric(data[temp], errors='coerce')
    plt.figure(figsize=(15, 8))
    data.groupby("Fare")[temp].mean().plot(kind='bar', color='blue')
    plt.title("Bar Plot of Fare against Survivors")
    plt.xlabel("Fare")
    plt.ylabel("Average Survival Rate")
    plt.grid(axis='y')
    static_dir = os.path.join(os.getcwd(), 'static')
    plot_path = os.path.join(static_dir, 'fare_survival_bar.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('fare_survival_bar.html', chart_url='/static/images/fare_survival_bar.png')

@app.route('/fare_survival_trend')
def fare_survival():
    Fare_survival = data.groupby("Fare")["Survived"].mean()
    plt.figure(figsize=(15, 6))
    plt.bar(Fare_survival.index, Fare_survival.values, color='skyblue', label="Survival Rate")
    z = np.polyfit(Fare_survival.index, Fare_survival.values, 2)
    p = np.poly1d(z)
    plt.plot(Fare_survival.index, p(Fare_survival.index), color='red', linestyle='-', linewidth=2, label="Trendline")
    plt.title("Fare vs. Survival Rate with Trendline")
    plt.xlabel("Fare")
    plt.ylabel("Survival Rate")
    plt.legend()
    static_dir = os.path.join(os.getcwd(), 'static')
    fare_plot_path = os.path.join(static_dir, 'fare_survival.png')
    plt.savefig(fare_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('fare_survival.html', fare_plot_url='/static/images/fare_survival.png')



@app.route('/fare_scatter')
def fare_scatter():
    plt.figure(figsize=(12, 6))
    scatter = sns.scatterplot(data=data, x="Fare", y="Survived", hue="Survived", palette="coolwarm")
    plt.title("Scatter Plot of Fare vs. Survival")
    plt.xlabel("Fare")
    plt.ylabel("Survival (0 = No, 1 = Yes)")
    plt.legend(title="Survival", labels=["Did Not Survive", "Survived"])
    static_dir = os.path.join(os.getcwd(), 'static')
    scatter_plot_path = os.path.join(static_dir, 'fare_scatter.png')
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('fare_scatter.html', scatter_plot_url='/static/images/fare_scatter.png')


@app.route('/fare_regression')
def fare_regression():
    plt.figure(figsize=(12, 6))
    reg_plot = sns.regplot(data=data, x="Fare", y="Survived", logistic=True, ci=None, scatter_kws={'alpha':0.5})
    plt.title("Regression Plot of Fare vs. Survival")
    plt.xlabel("Fare")
    plt.ylabel("Survival Probability (0 = No, 1 = Yes)")
    static_dir = os.path.join(os.getcwd(), 'static')
    reg_plot_path = os.path.join(static_dir, 'fare_regression.png')
    plt.savefig(reg_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('fare_regression.html', reg_plot_url='/static/images/fare_regression.png')










@app.route("/link-pclass", methods=["GET", "POST"])
def link_pclass():
    return render_template("link_pclass_survival.html")


@app.route('/pclass-survival-bar')
def pclass_survival_bar():
    temp = 'Survived'
    data[temp] = pd.to_numeric(data[temp], errors='coerce')
    plt.figure(figsize=(15, 8))
    data.groupby("Pclass")[temp].mean().plot(kind='bar', color='blue')
    plt.title("Bar Plot of Pclass against Survivors")
    plt.xlabel("Pclass")
    plt.ylabel("Average Survival Rate")
    plt.grid(axis='y')
    static_dir = os.path.join(os.getcwd(), 'static')
    plot_path = os.path.join(static_dir, 'pclass_survival_bar.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('pclass_survival_bar.html', chart_url='/static/images/pclass_survival_bar.png')

@app.route('/pclass_survival_trend')
def pclass_survival():
    pclass_survival = data.groupby("Pclass")["Survived"].mean()
    plt.figure(figsize=(15, 6))
    plt.bar(pclass_survival.index, pclass_survival.values, color='skyblue', label="Survival Rate")
    z = np.polyfit(pclass_survival.index, pclass_survival.values, 2)
    p = np.poly1d(z)
    plt.plot(pclass_survival.index, p(pclass_survival.index), color='red', linestyle='-', linewidth=2, label="Trendline")
    plt.title("Pclass vs. Survival Rate with Trendline")
    plt.xlabel("Pclass")
    plt.ylabel("Survival Rate")
    plt.legend()
    static_dir = os.path.join(os.getcwd(), 'static')
    pclass_plot_path = os.path.join(static_dir, 'pclass_survival.png')
    plt.savefig(pclass_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('pclass_survival.html', pclass_plot_url='/static/images/pclass_survival.png')



@app.route('/pclass_scatter')
def pclass_scatter():
    plt.figure(figsize=(12, 6))
    scatter = sns.scatterplot(data=data, x="Pclass", y="Survived", hue="Survived", palette="coolwarm")
    plt.title("Scatter Plot of Pclass vs. Survival")
    plt.xlabel("Pclass")
    plt.ylabel("Survival (0 = No, 1 = Yes)")
    plt.legend(title="Survival", labels=["Did Not Survive", "Survived"])
    static_dir = os.path.join(os.getcwd(), 'static')
    scatter_plot_path = os.path.join(static_dir, 'pclass_scatter.png')
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('pclass_scatter.html', scatter_plot_url='/static/images/pclass_scatter.png')


@app.route('/pclass_regression')
def pclass_regression():
    plt.figure(figsize=(12, 6))
    reg_plot = sns.regplot(data=data, x="Pclass", y="Survived", logistic=True, ci=None, scatter_kws={'alpha':0.5})
    plt.title("Regression Plot of Pclass vs. Survival")
    plt.xlabel("Pclass")
    plt.ylabel("Survival Probability (0 = No, 1 = Yes)")
    static_dir = os.path.join(os.getcwd(), 'static')
    reg_plot_path = os.path.join(static_dir, 'pclass_regression.png')
    plt.savefig(reg_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('pclass_regression.html', reg_plot_url='/static/images/pclass_regression.png')









@app.route("/link-age", methods=["GET", "POST"])
def link_age():
    return render_template("link_age_survival.html")


@app.route('/age-survival-bar')
def age_survival_bar():
    temp = 'Survived'
    data[temp] = pd.to_numeric(data[temp], errors='coerce')
    plt.figure(figsize=(15, 8))
    data.groupby("Age")[temp].mean().plot(kind='bar', color='blue')
    plt.title("Bar Plot of Age against Survivors")
    plt.xlabel("Age")
    plt.ylabel("Average Survival Rate")
    plt.grid(axis='y')
    static_dir = os.path.join(os.getcwd(), 'static')
    plot_path = os.path.join(static_dir, 'age_survival_bar.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('age_survival_bar.html', chart_url='/static/images/age_survival_bar.png')

@app.route('/age_survival_trend')
def age_survival():
    age_survival = data.groupby("Age")["Survived"].mean()
    plt.figure(figsize=(15, 6))
    plt.bar(age_survival.index, age_survival.values, color='skyblue', label="Survival Rate")
    z = np.polyfit(age_survival.index, age_survival.values, 2)
    p = np.poly1d(z)
    plt.plot(age_survival.index, p(age_survival.index), color='red', linestyle='-', linewidth=2, label="Trendline")
    plt.title("Age vs. Survival Rate with Trendline")
    plt.xlabel("Age")
    plt.ylabel("Survival Rate")
    plt.legend()
    static_dir = os.path.join(os.getcwd(), 'static')
    age_plot_path = os.path.join(static_dir, 'age_survival.png')
    plt.savefig(age_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('age_survival.html', age_plot_url='/static/images/age_survival.png')



@app.route('/age_scatter')
def age_scatter():
    plt.figure(figsize=(12, 6))
    scatter = sns.scatterplot(data=data, x="Age", y="Survived", hue="Survived", palette="coolwarm")
    plt.title("Scatter Plot of Age vs. Survival")
    plt.xlabel("Age")
    plt.ylabel("Survival (0 = No, 1 = Yes)")
    plt.legend(title="Survival", labels=["Did Not Survive", "Survived"])
    static_dir = os.path.join(os.getcwd(), 'static')
    scatter_plot_path = os.path.join(static_dir, 'age_scatter.png')
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('age_scatter.html', scatter_plot_url='/static/images/age_scatter.png')


@app.route('/age_regression')
def age_regression():
    plt.figure(figsize=(12, 6))
    reg_plot = sns.regplot(data=data, x="Age", y="Survived", logistic=True, ci=None, scatter_kws={'alpha':0.5})
    plt.title("Regression Plot of Age vs. Survival")
    plt.xlabel("Age")
    plt.ylabel("Survival Probability (0 = No, 1 = Yes)")
    static_dir = os.path.join(os.getcwd(), 'static')
    reg_plot_path = os.path.join(static_dir, 'age_regression.png')
    plt.savefig(reg_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('age_regression.html', reg_plot_url='/static/images/age_regression.png')










@app.route("/link-sibsp", methods=["GET", "POST"])
def link_sibsp():
    return render_template("link_sibsp_survival.html")


@app.route('/sibsp-survival-bar')
def sibsp_survival_bar():
    temp = 'Survived'
    data[temp] = pd.to_numeric(data[temp], errors='coerce')
    plt.figure(figsize=(15, 8))
    data.groupby("SibSp")[temp].mean().plot(kind='bar', color='blue')
    plt.title("Bar Plot of Number of Siblings/Spouses against Survivors")
    plt.xlabel("SibSp")
    plt.ylabel("Average Survival Rate")
    plt.grid(axis='y')
    static_dir = os.path.join(os.getcwd(), 'static')
    plot_path = os.path.join(static_dir, 'sibsp_survival_bar.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('sibsp_survival_bar.html', chart_url='/static/images/sibsp_survival_bar.png')

@app.route('/sibsp_survival_trend')
def sibsp_survival():
    sibsp_survival = data.groupby("SibSp")["Survived"].mean()
    plt.figure(figsize=(15, 6))
    plt.bar(sibsp_survival.index, sibsp_survival.values, color='skyblue', label="Survival Rate")
    z = np.polyfit(sibsp_survival.index, sibsp_survival.values, 2)
    p = np.poly1d(z)
    plt.plot(sibsp_survival.index, p(sibsp_survival.index), color='red', linestyle='-', linewidth=2, label="Trendline")
    plt.title("Number of Siblings/Spouses vs. Survival Rate with Trendline")
    plt.xlabel("SibSp")
    plt.ylabel("Survival Rate")
    plt.legend()
    static_dir = os.path.join(os.getcwd(), 'static')
    sibsp_plot_path = os.path.join(static_dir, 'sibsp_survival.png')
    plt.savefig(sibsp_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('sibsp_survival.html', sibsp_plot_url='/static/images/sibsp_survival.png')


@app.route('/sibsp_scatter')
def sibsp_scatter():
    plt.figure(figsize=(12, 6))
    scatter = sns.scatterplot(data=data, x="SibSp", y="Survived", hue="Survived", palette="coolwarm")
    plt.title("Scatter Plot of Number of Siblings/Spouses vs. Survival")
    plt.xlabel("SibSp")
    plt.ylabel("Survival (0 = No, 1 = Yes)")
    plt.legend(title="Survival", labels=["Did Not Survive", "Survived"])
    static_dir = os.path.join(os.getcwd(), 'static')
    scatter_plot_path = os.path.join(static_dir, 'sibsp_scatter.png')
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('sibsp_scatter.html', scatter_plot_url='/static/images/sibsp_scatter.png')


@app.route('/sibsp_regression')
def sibsp_regression():
    plt.figure(figsize=(12, 6))
    reg_plot = sns.regplot(data=data, x="SibSp", y="Survived", logistic=True, ci=None, scatter_kws={'alpha':0.5})
    plt.title("Regression Plot of Number of Siblings/Spouses vs. Survival")
    plt.xlabel("SibSp")
    plt.ylabel("Survival Probability (0 = No, 1 = Yes)")
    static_dir = os.path.join(os.getcwd(), 'static')
    reg_plot_path = os.path.join(static_dir, 'sibsp_regression.png')
    plt.savefig(reg_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('sibsp_regression.html', reg_plot_url='/static/images/sibsp_regression.png')










@app.route("/link-sex", methods=["GET", "POST"])
def link_sex():
    return render_template("link_sex_survival.html")


@app.route('/sex-survival-bar')
def sex_survival_bar():
    temp = 'Survived'
    data[temp] = pd.to_numeric(data[temp], errors='coerce')
    plt.figure(figsize=(15, 8))
    data.groupby("Sex")[temp].mean().plot(kind='bar', color='blue')
    plt.title("Bar Plot of Number of Siblings/Spouses against Survivors")
    plt.xlabel("Sex")
    plt.ylabel("Average Survival Rate")
    plt.grid(axis='y')
    static_dir = os.path.join(os.getcwd(), 'static')
    plot_path = os.path.join(static_dir, 'sex_survival_bar.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('sex_survival_bar.html', chart_url='/static/images/sex_survival_bar.png')


@app.route('/sex_scatter')
def sex_scatter():
    plt.figure(figsize=(12, 6))
    scatter = sns.scatterplot(data=data, x="Sex", y="Survived", hue="Survived", palette="coolwarm")
    plt.title("Scatter Plot of Number of Siblings/Spouses vs. Survival")
    plt.xlabel("Sex")
    plt.ylabel("Survival (0 = No, 1 = Yes)")
    plt.legend(title="Survival", labels=["Did Not Survive", "Survived"])
    static_dir = os.path.join(os.getcwd(), 'static')
    scatter_plot_path = os.path.join(static_dir, 'sex_scatter.png')
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return render_template('sex_scatter.html', scatter_plot_url='/static/images/sex_scatter.png')


















x = data.drop(['Survived'], axis=1).values
y = data['Survived'].values

mask = ~np.isnan(x).any(axis=1)
x = x[mask]
y = y[mask]


def preprocess_data(x, y):
    smote = SMOTE(sampling_strategy="auto", random_state=0)
    x_os, y_os = smote.fit_resample(x, y)
    return train_test_split(x_os, y_os, test_size=0.3, random_state=0)

x_train, x_test, y_train, y_test = preprocess_data(x, y)
scaler = StandardScaler().fit(x_train)
x_train_sc = scaler.transform(x_train)
x_test_sc = scaler.transform(x_test)


def logisticreg(sol):
    model = LogisticRegression(solver=sol, random_state=0)
    cv_scores = cross_val_score(model, x_train_sc, y_train, cv=5, scoring='accuracy')
    model.fit(x_train_sc, y_train)
    y_pred = model.predict(x_test_sc)
    return {
        "Solver": sol,
        "F1 Score": f1_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "CV Accuracy": np.mean(cv_scores) * 100,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Model": model
    }

def xgboost_class(estimators):
    model = XGBClassifier(n_estimators=estimators, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
                          random_state=0, n_jobs=-1)
    cv_scores = cross_val_score(model, x_train_sc, y_train, cv=5, scoring='accuracy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return {
        "Solver": f"n_estimators={estimators}",
        "F1 Score": f1_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "CV Accuracy": np.mean(cv_scores) * 100,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Model": model
    }

def svc_class(ker):
    model = SVC(kernel=ker, random_state=0)
    cv_scores = cross_val_score(model, x_train_sc, y_train, cv=5, scoring='accuracy')
    model.fit(x_train_sc, y_train)
    y_pred = model.predict(x_test_sc)
    return {
        "Solver": ker,
        "F1 Score": f1_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "CV Accuracy": np.mean(cv_scores) * 100,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Model": model
    }


logistic_solvers = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
logistic_models = [logisticreg(sol) for sol in logistic_solvers]
best_logistic = max(logistic_models, key=lambda m: m["F1 Score"])

xgb_estimators = range(1, 51, 5)
xgb_models = [xgboost_class(est) for est in xgb_estimators]
best_xgb = max(xgb_models, key=lambda m: m["F1 Score"])

svm_kernels = ["linear", "poly", "rbf", "sigmoid"]
svm_models = [svc_class(ker) for ker in svm_kernels]
best_svc = max(svm_models, key=lambda m: m["F1 Score"])


all_models = logistic_models + xgb_models + svm_models
best_overall = max(all_models, key=lambda m: m["Accuracy"])

@app.route('/task-1-2')
def task_1_2_():
    return render_template('task1_2.html', best_logistic=best_logistic, solver_results_list=logistic_models)

@app.route('/task-1-3')
def task_1_3_():
    if isinstance(best_overall.get('Model'), object):
        best_overall['Model'] = best_overall['Model'].__class__.__name__

    return render_template(
        'task1_3.html',
        logistic_models=logistic_models,
        xgb_models=xgb_models,
        svm_models=svm_models,
        best_overall=best_overall
    )


if __name__ == '__main__':
    app.run(debug=True)

