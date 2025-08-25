import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

class ChurnModelUtils:
    def __init__(self, num_clientes, salario_medio, churn_rate, taxa_retencao=0.10, custo_retencao=100):
        self.num_clientes = num_clientes
        self.salario_medio = salario_medio
        self.churn_rate = churn_rate
        self.taxa_retencao = taxa_retencao
        self.custo_retencao = custo_retencao

    @staticmethod
    def retorno_bruto_churn(num_clientes, churn_rate, recall, taxa_retencao, salario_medio):
        churn_total = num_clientes * churn_rate
        tp = churn_total * recall
        clientes_retidos = tp * taxa_retencao
        retorno_bruto = clientes_retidos * salario_medio
        return {
            "Clientes_que_iriam_sair": churn_total,
            "Churns_previstos_com_sucesso": tp,
            "Clientes_retidos": clientes_retidos,
            "Retorno_bruto": retorno_bruto
        }

    @staticmethod
    def calcular_roi(num_clientes, salario_medio, churn_rate, recall, precisao, taxa_retencao, custo_retencao):
        churn_total = num_clientes * churn_rate
        tp = churn_total * recall
        total_pred_churn = tp / precisao if precisao > 0 else 0
        fp = total_pred_churn - tp

        custo_total = total_pred_churn * custo_retencao
        clientes_retidos = tp * taxa_retencao
        receita_preservada = clientes_retidos * salario_medio

        roi = (receita_preservada - custo_total) / custo_total if custo_total > 0 else 0
        return {
            "Receita_Preservada": receita_preservada,
            "Custo_Retencao": custo_total,
            "Clientes_Retidos": clientes_retidos,
            "ROI": roi
        }

    @staticmethod
    def train_and_evaluate_cv(model, X, y, n_splits=5, plot_roc=True):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import (
            precision_score, recall_score, accuracy_score, f1_score,
            roc_curve, auc
        )
        import numpy as np
        import matplotlib.pyplot as plt

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        metrics = {'precision': [], 'recall': [], 'accuracy': [], 'f1': []}
        
        # Armazena predições OOF
        oof_preds = np.zeros(len(X))
        oof_probas = np.zeros(len(X))  # para curva ROC
        oof_true = np.zeros(len(X))

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]  # probabilidade da classe positiva

            # Métricas fold a fold
            metrics['precision'].append(precision_score(y_test, y_pred))
            metrics['recall'].append(recall_score(y_test, y_pred))
            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['f1'].append(f1_score(y_test, y_pred))

            # Armazena OOF
            oof_preds[test_idx] = y_pred
            oof_probas[test_idx] = y_proba
            oof_true[test_idx] = y_test

        # Calcula médias das métricas
        results = {m: (np.mean(metrics[m]), np.std(metrics[m])) for m in metrics}

        # Curva ROC (baseado nas probabilidades OOF)
        if plot_roc:
            fpr, tpr, _ = roc_curve(oof_true, oof_probas)
            roc_auc = auc(fpr, tpr)
            # Adiciona a AUC como tupla para manter padrão
            results['roc_auc'] = (roc_auc, 0.0)

            # Título customizado
            model_name = model.__class__.__name__
            prec = results['precision'][0]
            rec = results['recall'][0]
            f1 = results['f1'][0]
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC - {model_name}\nPrecision: {prec:.2f} | Recall: {rec:.2f} | F1: {f1:.2f}')
            plt.legend()
            plt.grid(True)
            plt.show()

        return results


    def track_model_cv(self, result_cv, model_name=""):
        import mlflow
        with mlflow.start_run(run_name=model_name):
            for m in result_cv:
                mlflow.log_metric(f"{m}_mean", result_cv[m][0])
                mlflow.log_metric(f"{m}_std", result_cv[m][1])
            mlflow.log_param("algoritmo", model_name)
            
            # Retorno Bruto (simplificado)
            recall_cv = result_cv["recall"][0]      # média do recall
            retorno = self.retorno_bruto_churn(
                num_clientes=self.num_clientes,
                churn_rate=self.churn_rate,
                recall=recall_cv,
                taxa_retencao=self.taxa_retencao,
                salario_medio=self.salario_medio
            )
            mlflow.log_metric("retorno_bruto", retorno["Retorno_bruto"])
            mlflow.log_metric("clientes_retidos_bruto", retorno["Clientes_retidos"])
            
            # ROI completo (considerando precisão e custos)
            precisao_cv = result_cv["precision"][0] # média da precisão
            roi_dict = self.calcular_roi(
                num_clientes=self.num_clientes,
                salario_medio=self.salario_medio,
                churn_rate=self.churn_rate,
                recall=recall_cv,
                precisao=precisao_cv,
                taxa_retencao=self.taxa_retencao,
                custo_retencao=self.custo_retencao
            )
            mlflow.log_metric("receita_preservada", roi_dict["Receita_Preservada"])
            mlflow.log_metric("custo_retencao", roi_dict["Custo_Retencao"])
            mlflow.log_metric("clientes_retidos", roi_dict["Clientes_Retidos"])
            mlflow.log_metric("roi", roi_dict["ROI"])
            print(f"Tracking done for {model_name}")

    def track_model(self, result, model_name=""):
        import mlflow
        with mlflow.start_run(run_name=model_name):
            mlflow.sklearn.autolog()
            mlflow.log_metric("precision_train", result["metrics_train"]["precision"])
            mlflow.log_metric("recall_train", result["metrics_train"]["recall"])
            mlflow.log_metric("accuracy_train", result["metrics_train"]["accuracy"])
            mlflow.log_metric("f1_train", result["metrics_train"]["f1"])
            mlflow.log_metric("precision_test", result["metrics_test"]["precision"])
            mlflow.log_metric("recall_test", result["metrics_test"]["recall"])
            mlflow.log_metric("accuracy_test", result["metrics_test"]["accuracy"])
            mlflow.log_metric("f1_test", result["metrics_test"]["f1"])
            
            # Retorno Bruto (simplificado)
            retorno = self.retorno_bruto_churn(
                num_clientes=self.num_clientes,
                churn_rate=self.churn_rate,
                recall=result["metrics_test"]["recall"],
                taxa_retencao=self.taxa_retencao,
                salario_medio=self.salario_medio
            )
            mlflow.log_metric("retorno_bruto", retorno["Retorno_bruto"])
            mlflow.log_metric("clientes_retidos_bruto", retorno["Clientes_retidos"])
            
            # ROI completo (considerando precisão e custos)
            roi_dict = self.calcular_roi(
                num_clientes=self.num_clientes,
                salario_medio=self.salario_medio,
                churn_rate=self.churn_rate,
                recall=result["metrics_test"]["recall"],
                precisao=result["metrics_test"]["precision"],
                taxa_retencao=self.taxa_retencao,
                custo_retencao=self.custo_retencao
            )
            mlflow.log_metric("receita_preservada", roi_dict["Receita_Preservada"])
            mlflow.log_metric("custo_retencao", roi_dict["Custo_Retencao"])
            mlflow.log_metric("clientes_retidos", roi_dict["Clientes_Retidos"])
            mlflow.log_metric("roi", roi_dict["ROI"])
            print(f"Tracking done for {model_name}")

    def avaliar_modelos_cv(self, modelos, X, y, n_splits=5, verbose=True):
        """
        Executa cross-validation em múltiplos modelos e retorna DataFrame de métricas, incluindo retorno financeiro.
        """
        import time
        resultados = []
        for nome, modelo in modelos.items():
            if verbose:
                print(f"Treinando: {nome}")
            start = time.time()
            result = self.train_and_evaluate_cv(modelo, X, y, n_splits=n_splits, plot_roc=False)
            resultado_alg = {'algoritmo': nome}
            for m, (mean, std) in result.items():
                resultado_alg[f'{m}_mean'] = mean
                resultado_alg[f'{m}_std'] = std
            resultado_alg['tempo_execucao'] = time.time() - start

            # Cálculo financeiro
            recall = result['recall'][0]
            precision = result['precision'][0]
            retorno = self.retorno_bruto_churn(
                num_clientes=self.num_clientes,
                churn_rate=self.churn_rate,
                recall=recall,
                taxa_retencao=self.taxa_retencao,
                salario_medio=self.salario_medio
            )
            roi = self.calcular_roi(
                num_clientes=self.num_clientes,
                salario_medio=self.salario_medio,
                churn_rate=self.churn_rate,
                recall=recall,
                precisao=precision,
                taxa_retencao=self.taxa_retencao,
                custo_retencao=self.custo_retencao
            )
            resultado_alg['retorno_bruto'] = retorno['Retorno_bruto']
            resultado_alg['roi'] = roi['ROI']
            resultados.append(resultado_alg)
        df_resultados = pd.DataFrame(resultados)
        return df_resultados

    @staticmethod
    def ranking_modelos(df_resultados, top_n=5):
        """
        Exibe ranking dos melhores modelos por F1, ROC AUC, Precision e Recall.
        """
        print("\n🏆 RANKING DOS MELHORES MODELOS")
        for metric in ['f1_mean', 'roc_auc_mean', 'precision_mean', 'recall_mean']:
            if metric in df_resultados.columns:
                print(f"\nTop {top_n} por {metric.replace('_mean','').upper()}:")
                display(df_resultados.nlargest(top_n, metric)[['algoritmo', metric, metric.replace('mean','std')]])

    @staticmethod
    def plot_comparativo_barras(df_resultados):
        """
        Plota gráficos de barras comparando F1, Precision, Recall, Accuracy, ROC AUC, Retorno Bruto e ROI.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        metrics = ['f1_mean', 'precision_mean', 'recall_mean', 'accuracy_mean']
        if 'roc_auc_mean' in df_resultados.columns:
            metrics.append('roc_auc_mean')
        if 'retorno_bruto' in df_resultados.columns:
            metrics.append('retorno_bruto')
        if 'roi' in df_resultados.columns:
            metrics.append('roi')
        plt.figure(figsize=(4*len(metrics), 7))
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, len(metrics), i)
            order = df_resultados.sort_values(metric, ascending=False)
            sns.barplot(y='algoritmo', x=metric, data=order, palette='viridis')
            plt.title(metric.replace('_mean','').replace('_',' ').upper())
            plt.xlabel('Score' if metric not in ['retorno_bruto','roi'] else metric.replace('_',' ').title())
            plt.ylabel('')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_radar_top_n(df_resultados, n=5):
        """
        Plota gráfico radar dos top N modelos por F1 Score.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        top_n = df_resultados.nlargest(n, 'f1_mean')
        metrics = ['precision_mean', 'recall_mean', 'f1_mean', 'accuracy_mean']
        labels = ['Precision', 'Recall', 'F1', 'Accuracy']
        if 'roc_auc_mean' in df_resultados.columns:
            metrics.append('roc_auc_mean')
            labels.append('ROC AUC')
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for i, (_, row) in enumerate(top_n.iterrows()):
            values = [row[m] for m in metrics]
            values += values[:1]
            ax.plot(angles, values, label=row['algoritmo'])
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        plt.title(f'Radar Top {n} Modelos')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.show()

    def plot_roc_top_n(self, modelos, X, y, df_resultados, n=3, n_splits=5):
        """
        Plota curvas ROC dos top N modelos por F1 Score.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_curve, auc
        top_n = df_resultados.nlargest(n, 'f1_mean')
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10.colors
        for i, (_, row) in enumerate(top_n.iterrows()):
            nome = row['algoritmo']
            model = modelos[nome]
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            tprs, aucs = [], []
            mean_fpr = np.linspace(0, 1, 100)
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                model.fit(X_train, y_train)
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_proba = model.decision_function(X_test)
                else:
                    continue
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(roc_auc)
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, color=colors[i],
                     label=f'{nome} (AUC = {mean_auc:.3f} ± {std_auc:.3f})', linewidth=2)
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[i], alpha=0.15)
        plt.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC=0.5)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Falsos Positivos')
        plt.ylabel('Verdadeiros Positivos')
        plt.title(f'Curvas ROC - Top {n} Modelos')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.show()