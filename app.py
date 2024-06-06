from flask import Flask, render_template, request

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
df = pd.read_csv(("stackoverflow_full.csv"))
df.drop(columns = 'Unnamed: 0', inplace=True)
df.drop(columns = 'HaveWorkedWith', inplace=True)
def segment_country(country):
    if country in ['United States of America', 'Canada', 'Mexico']:
        return 'NorthAmerica'
    elif country in ['United Kingdom of Great Britain and Northern Ireland', 'France', 'Germany', 'Spain', 'Italy', 'Portugal', 'Belgium', 'Netherlands', 'Austria', 'Switzerland', 'Denmark', 'Ireland', 'Norway', 'Sweden', 'Finland', 'Greece', 'Czech Republic', 'Slovakia', 'Hungary', 'Poland']:
        return 'Europe'
    elif country in ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela, Bolivarian Republic of...', 'Bolivia']:
        return 'South America'
    elif country in ['China', 'Japan', 'South Korea', 'Viet Nam', 'India', 'Sri Lanka', 'Pakistan', 'Bangladesh', 'Indonesia', 'Malaysia', 'Philippines', 'Taiwan', 'Thailand', 'Cambodia', 'Myanmar', 'Laos', 'Singapore', 'Hong Kong (S.A.R.)']:
        return 'Asia'
    elif country in ['Australia', 'New Zealand', 'Fiji', 'Papua New Guinea', 'Solomon Islands', 'Vanuatu', 'Samoa', 'Tonga']:
        return 'Australia'
    else:
        return 'Others' 

# Создаем новый столбец Continent
df['Continent'] = df['Country'].apply(segment_country)
continent_counts = df['Continent'].value_counts()
df.drop(columns = 'Country', inplace=True)
# Также удаляем столбец YearsCodePro, поскольку у нас есть столбец YearsCode
df.drop(columns = 'YearsCodePro', inplace=True)
# Сделаем функцию для удаления выбросов с помощью IQR
df_copy = df.copy()

# Используем енкодер
label_encoder = LabelEncoder()
categorical_columns = ['Age', 'Accessibility', 'EdLevel', 'Gender', 'MentalHealth', 'MainBranch', 'Continent']
for col in categorical_columns:
    df_copy[col] = label_encoder.fit_transform(df[col])
def remove_outliers_iqr(data, column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
    return data

df_copy = remove_outliers_iqr(df_copy, 'YearsCode')
df_copy = remove_outliers_iqr(df_copy, 'PreviousSalary')
df_copy = remove_outliers_iqr(df_copy, 'ComputerSkills')
X = df_copy.drop("Employed", axis=1)  # фичи
y = df_copy["Employed"]  # таргет

# Делаем выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'max_depth': [10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

app = Flask(__name__)

results = []

@app.route('/')
def index():
    button_text = "Выставить гиперпараметр"
    return render_template('index.html', button_text=button_text)

def makeRegression():
    decision_tree = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

    grid_search.fit(X_train, y_train)

    # Лучшие гиперпараметры
    best_params = grid_search.best_params_
    best_decision_tree = DecisionTreeClassifier(random_state=42, **best_params)
    best_decision_tree.fit(X_train, y_train)

    y_pred_dt = best_decision_tree.predict(X_test)

    y_scores_dt = best_decision_tree.predict_proba(X_train)[:, 1]
    fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_train, y_scores_dt)
    auc_dt = auc(fpr_dt, tpr_dt)
    return auc_dt

@app.route('/submit', methods=['POST'])
def submit():
    # button_text = str(request.form['min_samples_split'])
    min_samples_split = [int(x) for x in str(request.form['min_samples_split']).split(',')]
    max_depth = [int(x) for x in str(request.form['max_depth']).split(',')]
    # min_samples_leaf = [int(x) for x in str(request.form['min_samples_leaf']).split(',')]
    param_grid['min_samples_split'] = min_samples_split
    param_grid['max_depth'] = max_depth
    # param_grid['min_samples_leaf'] = min_samples_leaf
    results.insert(0, makeRegression())
    return render_template('index.html', roc=makeRegression(), l=results)

if __name__ == '__main__':
    app.run(debug=True)

