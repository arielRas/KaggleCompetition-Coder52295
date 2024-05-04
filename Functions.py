import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve, classification_report, roc_auc_score

class Correlations():
    def __init__(self, df:pd.DataFrame, target_var:str, vars:list, vars_type:str):
        self.df = df
        self.target_var = target_var
        self.vars = vars
        self.vars_type = vars_type

    #CREA UN DATAFRAME VACIO PARA ALOJAR LAS CORRELACIONES
    def _get_correlation_dataframe(self) -> pd.DataFrame:
        df_corr = pd.DataFrame(index=self.vars, columns=['correlation'])
        return df_corr    

    #DEVUELVE EL NUMERO MULTIPLO DE 10 SUPERIOR INMEDIATO AL VALOR MAXIMO DE LA SERIE
    def _get_limit_value(self, values:pd.Series) -> float:
        limit_value = values.abs().max()
        limit_value = np.ceil(limit_value*10)/10
        return limit_value

    #FUNCION PARA PLOTEAR CORRELACION DE PUNTO BISERIAL
    def plot_biserial_point(self):
        df_corr = self._get_correlation_dataframe()
        corr_list = [stats.pointbiserialr(self.df[self.target_var], self.df[var]) for var in self.vars]
        for var, corr in zip(self.vars, corr_list):
            df_corr.loc[var, 'correlation'] = corr.statistic
        self._plot_correlation(df_corr)

    #FUNCION PARA PLOTEAR COEFICIENTE DE CONTIGENCIA
    def plot_coef_contingency(self):
        df_corr = self._get_correlation_dataframe()
        for var in self.vars:
            contingency_table = pd.crosstab(self.df[var], self.df[self.target_var])
            chi_2 = stats.chi2_contingency(contingency_table)
            df_corr.loc[var, 'correlation'] = math.sqrt(chi_2.statistic/(self.df.shape[0]+chi_2.statistic))
        self._plot_correlation(df_corr)

    #FUNCION PARA PLOTEAR COEFICIENTE DE CONTINGENCIA
    def plot_contingency_coef(self):
        n = self.df.shape[0]
        df_corr = self._get_correlation_dataframe()
        for var in self.vars:
            contingency_table = pd.crosstab(self.df[self.target_var], self.df[var])
            chi_2 = stats.chi2_contingency(contingency_table)
            df_corr.loc[var, 'coef'] = math.sqrt(chi_2.statistic/(n + chi_2.statistic))            
        self._plot_correlation(df_corr)

    #FUNCION PARA PLOTEAR EL GRAFICO DE BARRAS
    def _plot_correlation(self, df_corr:pd.DataFrame):
        colors = ['#000E9E' if corr < 0 else '#3DB2DA' for corr in df_corr.correlation]
        limit_value = self._get_limit_value(df_corr.correlation)
        plt.style.use("bmh")
        plt.figure(figsize=(8,5))  
        sns.barplot(x=df_corr.correlation, y=df_corr.index, hue=df_corr.index, palette=colors)
        plt.title(f'Correlacion de variable objetivo con variables {self.vars_type}', fontsize=14)
        plt.xlabel('Coeficiente de correlacion')
        plt.ylabel(f'Variables {self.vars_type}')
        plt.xlim(-1*limit_value, limit_value)
        plt.axvline(x=0, color='black', linestyle='-')
        plt.tight_layout()
        plt.show()
        plt.style.use("default")

class Distributions():
    def __init__(self, x:pd.Series, hue:pd.Series, var_name:str, label:str):
        self.x = x
        self.hue = hue
        self.var_name = var_name
        self.label = label

    #FUNCION PARA PLOTEAR HISTOGRAMA Y BOXPLOT DE UNA VARIABLE
    def plot_distribution(self, bins:int):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
        sns.histplot(x=self.x, hue=self.hue, multiple='stack',bins=bins, kde=True, ax=ax[0])
        ax[0].set_title(f'Distribucion de la variable {self.var_name}')
        ax[0].set_xlabel(self.label)
        ax[0].get_legend().set_title('Salió del banco')
        sns.boxplot(x=self.x, hue=self.hue, ax=ax[1])
        ax[1].set_title(f'Distribucion de la variable {self.var_name}')
        ax[1].set_xlabel(self.label)
        ax[1].set_ylabel(f'Variable "{self.var_name}"')
        ax[1].get_legend().set_title('Salió del banco')
        plt.tight_layout()
        plt.show()

    #FUNCION PARA PLOTEAR DOS GRAFICOS DE BARRAS DE VARIABLE CUALITATIVA(FRECUENCIA Y PROPORCION)
    def plot_distribution_cualit(self):
        mpl.style.use('bmh')
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12,5))
        sns.countplot(x=self.x , hue=self.hue, ax=ax[0])
        ax[0].set_title(f'Distribucion de la variable {self.var_name}')
        ax[0].set_xlabel(self.label)
        ax[0].set_ylabel('Frecuencia')
        ax[0].get_legend().set_title('Salió del banco')
        sns.countplot(x=self.x , hue=self.hue, stat='proportion', ax=ax[1])
        ax[1].set_title(f'Distribucion de la variable {self.var_name}')
        ax[1].set_xlabel(self.label)
        ax[1].set_ylabel('Proporcion')
        ax[1].get_legend().set_title('Salió del banco')
        plt.tight_layout()
        plt.show()
        mpl.style.use('default')

class Metrics():
    def __init__(self) -> None:
        pass

    def plot_confusion_matrix_dicotomic(self, y_true, y_pred, label_1, label_2):
        matrix = confusion_matrix(y_true, y_pred)
        sns.heatmap(matrix, fmt='.0f', annot=True, cmap='inferno')
        plt.title('Matriz de confusion')
        plt.xlabel('Predict data')        
        plt.ylabel('True data')
        if label_1 and label_2:
            plt.xticks(ticks=[0.5,1.5], labels=[label_1, label_2])
            plt.yticks(ticks=[0.5,1.5], labels=[label_1, label_2])
        plt.show()

    def plot_roc_curve(self, y_true, y_prob):
        mpl.style.use('ggplot')
        false_positive_rate, true_positive_rate, _threshold = roc_curve(y_true, y_prob)
        sns.lineplot(x=false_positive_rate, y=true_positive_rate)
        plt.plot([0, 1], ls="--")
        plt.title('Curva ROC')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.tight_layout()
        plt.show()
        mpl.style.use('default')

    def get_metrics(self, model:str, sample:str, y_true, y_pred, y_prob)->dict:
        report = classification_report(y_true, y_pred, output_dict=True)
        precision_0 = report.get('0', {}).get('precision', None)
        precision_1 = report.get('1', {}).get('precision', None)
        recall_0 = report.get('0', {}).get('recall', None)
        recall_1 = report.get('1', {}).get('recall', None)
        f1_0 = report.get('0', {}).get('f1-score', None)
        f1_1 = report.get('1', {}).get('f1-score', None)
        accuracy = report.get('accuracy', None)
        rog_score = roc_auc_score(y_true, y_prob)
        keys = ['model','sample','accuracy','rog_score','precision_0','precision_1','recall_0','recall_1','f1_score_0','f1_score_1']
        values =  [model, sample, accuracy, rog_score, precision_0, precision_1, recall_0, recall_1, f1_0, f1_1]
        metrics = {}
        for key, value in zip(keys, values):
            metrics[key] = value
        return metrics