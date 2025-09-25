import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

try:
    df = pd.read_csv('insurance.csv')
except FileNotFoundError:
    print("Erro: O arquivo 'insurance.csv' não foi encontrado. Certifique-se de que ele está na mesma pasta do script.")
    exit()

tercis = df['charges'].quantile([1/3, 2/3]).tolist()
bins = [df['charges'].min() - 1, tercis[0], tercis[1], df['charges'].max() + 1]
labels = ['Baixo Custo', 'Médio Custo', 'Alto Custo']
df['cost_category'] = pd.cut(df['charges'], bins=bins, labels=labels, include_lowest=True)

le = LabelEncoder()
df['cost_category_encoded'] = le.fit_transform(df['cost_category'])
y = df['cost_category_encoded']

X = df.drop(['charges', 'cost_category', 'cost_category_encoded'], axis=1)
X = pd.get_dummies(X, columns=['sex', 'smoker', 'region'], drop_first=True)

scaler = StandardScaler()
numerical_cols = ['age', 'bmi', 'children']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

classifiers = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'KNeighbors': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(random_state=42)
}

resultados_finais = {clf: {'Acuracia': [], 'F1-Score': []} for clf in classifiers}
NUM_REPETICOES = 20

for i in range(NUM_REPETICOES):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    for nome, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        resultados_finais[nome]['Acuracia'].append(acc)
        resultados_finais[nome]['F1-Score'].append(f1)

dados_csv = []
nome_dataset = 'insurance'

for classificador in classifiers.keys():
    linha_acc = {
        'nome_dataset': nome_dataset,
        'nome_classificador': classificador,
        'metrica': 'Acuracia',
    }
    for i, resultado in enumerate(resultados_finais[classificador]['Acuracia'], 1):
        linha_acc[f'resultado_{i}'] = resultado
    dados_csv.append(linha_acc)

    linha_f1 = {
        'nome_dataset': nome_dataset,
        'nome_classificador': classificador,
        'metrica': 'F1-Score',
    }
    for i, resultado in enumerate(resultados_finais[classificador]['F1-Score'], 1):
        linha_f1[f'resultado_{i}'] = resultado
    dados_csv.append(linha_f1)

df_resultados = pd.DataFrame(dados_csv)
df_resultados.to_csv('resultados_classificacao_real.csv', index=False)

print("Arquivo 'resultados_classificacao_real.csv' gerado com sucesso!")