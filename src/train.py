#%%
import numpy as np
import pandas as pd
import mlflow
import importlib
import yaml


from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
#%%

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_id=207252348785056863)



# %%
df = pd.read_csv('../data/rclientes.csv')

cols_to_drop = ['RowNumber','Surname', 'Geography', 'Gender', 'CustomerId']

df = df.drop(columns=cols_to_drop)
df.head()
# %%

# %%
X = df.drop('Exited', axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# %%
# Carregar configuração do modelo do YAML
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_class_path = config['model']['class']
model_params = config['model'].get('params', {})

# Importar classe do modelo dinamicamente
module_name, class_name = model_class_path.rsplit('.', 1)
module = importlib.import_module(module_name)
ModelClass = getattr(module, class_name)

with mlflow.start_run():
    mlflow.sklearn.autolog()
    model = ModelClass(**model_params)

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    print(y_pred)

    precision_train = precision_score(y_train,y_pred_train)
    recall_train = recall_score(y_train,y_pred_train)
    acc_train = accuracy_score(y_train,y_pred_train)
    f1_train = f1_score(y_train,y_pred_train)

    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    acc = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)

    print('Precision Train: ', precision_train)
    print('Recall Train: ', recall_train)
    print('Acc Train: ', acc_train)
    print('F1 Train: ', f1_train)

    # Log das métricas de treino
    mlflow.log_metric("precision_train", precision_train)
    mlflow.log_metric("recall_train", recall_train)
    mlflow.log_metric("accuracy_train", acc_train)
    mlflow.log_metric("f1_train", f1_train)

    print('===========================================')
    print('Precision Test: ', precision)
    print('Recall Test: ', recall)
    print('Acc Test: ', acc)
    print('F1 Test: ', f1)

    # Log das métricas de teste
    mlflow.log_metric("precision_test", precision)
    mlflow.log_metric("recall_test", recall)
    mlflow.log_metric("accuracy_test", acc)
    mlflow.log_metric("f1_test", f1)
# %%