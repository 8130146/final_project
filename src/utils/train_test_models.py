import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import lime.lime_tabular
import pickle
import graphviz
import os
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import BorderlineSMOTE

# Funcao para importar o dataset


@st.cache_resource
def load_dataset():
    data = pd.read_csv('../datasets/dataset_short.csv')
    return data

# Funcao para calcular e guardar as metricas de um modelo


def calc_save_metrics(Y_test, y_pred, nome):
    metrics = {}
    # Calcular a accuracy, precision, recall e F1-score
    accuracy = accuracy_score(Y_test, y_pred)
    metrics["accuracy"] = accuracy
    precision = precision_score(Y_test, y_pred, average='binary')
    metrics["precision"] = precision
    recall = recall_score(Y_test, y_pred, average='binary')
    metrics["recall"] = recall
    f1 = f1_score(Y_test, y_pred, average='binary')
    metrics["f1"] = f1
    # Guardar o dicionario das metricas na pasta metrics
    pasta = '../metrics'
    os.makedirs(pasta, exist_ok=True)
    file_path = os.path.join(pasta, nome)
    with open(file_path, 'wb') as file:
        pickle.dump(metrics, file)

# Funcao para guardar um modelo na pasta models


def save_model(model, name):
    models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    file_path_models = os.path.join(models_dir, name)
    # Guardar o modelo usando o pickle
    with open(file_path_models, 'wb') as f:
        pickle.dump(model, f)


# Funcao para gerar o grafico lime para um modelo
def generate_lime_plot(best_model, X_train, X_test, X, nome):
    # Definir o numero de carateristicas a serem explicadas
    num_features = 8
    # Definir os nomes das características (features)
    feature_names = X.columns.tolist()

    # Converter X_train e X_test para DataFrames para usar replace e fillna
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # substituir valores infinitos e NaN nos dados de treino
    X_train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train_df.fillna(X_train.mean(), inplace=True)

    # substituir valores infinitos e NaN nos dados de teste
    X_test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test_df.fillna(X_test.mean(), inplace=True)

    # Gerar explicação com LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_df.values, mode='classification', feature_names=feature_names)
    # primeira instancioa dos dados de teste
    instance = X_test_df.values[0]
    try:
        explanation = explainer.explain_instance(
            instance, best_model.predict_proba, num_features=num_features)
        # Guardar a imagem do LIME em formato PNG
        fig = explanation.as_pyplot_figure()
        output_dir = '../images'
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, nome),
                    bbox_inches='tight', dpi=150)
        plt.close(fig)  # Fechar a figura após guardar
    except ValueError as e:
        print(f"Error in Lime explanation: {e}")

# Funcao para gerar a arvore de decisao


def generate_dtree_graph(best_model, X):
    # Gerar ficheiro .dot
    img = export_graphviz(best_model, out_file=None,
                          feature_names=X.columns,
                          filled=True, rounded=True,
                          special_characters=True)
    # Visualizar a árvore de decisão
    graph = graphviz.Source(img)
    # Guardar o grafico
    folder_path = "../images"
    graph.render(os.path.join(folder_path, "grafico_decision_tree"),
                 format='png')

# Funcao para gerar o grafico shap para a decisoon tree


def generate_shap_plot_dtree(best_model, X_train, X_test):
    # Calcular os valores SHAP para as previsões do conjunto de teste
    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer.shap_values(X_test, check_additivity=False)
    # Plotar os valores SHAP
    shap.summary_plot(shap_values, X_test, show=False)
    fig = plt.gcf()  # Obtém a figura atual
    fig.set_size_inches(24, 8)
    output_dir = '../images'
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f'shap_decision_tree.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)

# Funcao para treinar, exportar metricas, gerar arvore, gerar graficos SHAP e LIME para a decision tree


def decisiontree():
    dados = load_dataset()
    X = dados.drop(["DDoS"], axis=1)
    Y = dados['DDoS']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 20],
        # Usar todas as features disponíveis - none
        'max_features': ['sqrt', 'log2', None],
        'min_samples_leaf': [1, 2, 5, 10]
    }

    random_search = RandomizedSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_distributions=params,
        n_iter=100,
        cv=5,
        scoring='recall',
        n_jobs=-1
    )

    random_search.fit(X_train, Y_train)
    best_model = random_search.best_estimator_
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    y_pred = best_model.predict(X_test_df)
    calc_save_metrics(Y_test, y_pred, 'metrics_decision_tree.pkl')
    save_model(best_model, 'decision_tree_model.pkl')
    generate_dtree_graph(best_model, X)
    generate_shap_plot_dtree(best_model, X_train, X_test)
    generate_lime_plot(best_model, X_train, X_test, X, 'lime_decisionTree.png')

# Funcao para gerar graficos de densidade para cada carateristica separados por classe


def plot_feature_distributions(X_train, Y_train):
    # Distribuições das características por classe
    for feature_name in X_train.columns:
        fig, ax = plt.subplots()
        for class_idx, class_name in enumerate(np.unique(Y_train)):
            sns.kdeplot(X_train[Y_train == class_idx]
                        [feature_name], label=class_name, ax=ax)
        ax.set_title(f'Distribuição da Característica: {feature_name}')
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Densidade')
        ax.legend(title='Classe')
        fig.tight_layout()
        fig.savefig(f'../images/distribuicao_{feature_name}.png')
        plt.close(fig)

# Funcao para gerar graficos da contribuicao media de cada carateristica para a probabilidade predita de cada classe


def analyze_feature_contributions(X_test, y_proba, means, variances, Y):
    # Analisar a contribuição de cada característica usando y_proba
    for class_index, class_name in enumerate(np.unique(Y)):
        fig, ax = plt.subplots()
        contributions = np.zeros(X_test.shape)

        for j, feature_name in enumerate(X_test.columns):
            # Para cada característica, calcular a contribuição média da probabilidade predita
            contributions[:, j] = y_proba * (X_test[feature_name] -
                                             means[class_index, j])**2 / (2 * variances[class_index, j])

        ax.bar(X_test.columns, contributions.mean(axis=0))
        ax.set_title(
            f'Contribuição das Características para a Classe {class_name}')
        ax.set_xlabel('Características')
        ax.set_ylabel('Contribuição Média')
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        fig.savefig(
            f'../images/contribuicao_caracteristicas_classe_{class_name}.png')
        plt.close(fig)

# Funcao para gerar o grafico shap para o naive bayes


def generate_shap_plot_nb(best_model, X_train, X_test, feature_names):
    # Resumir conjunto de treino em 5 clusters para acelarar calculo
    X_train_summary = shap.kmeans(X_train, 5)
    # Calcular os valores SHAP para as previsões do conjunto de teste
    explainer = shap.KernelExplainer(best_model.predict_proba, X_train_summary)
    shap_values = explainer.shap_values(X_test, nsamples='auto')
    # Se for uma lista
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    # Grafico resumo
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names, show=False)
    fig = plt.gcf()
    fig.set_size_inches(24, 8)
    output_dir = '../images'
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'shap_naive_bayes.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)


# Funcao para treinar, exportar metricas, gerar graficos SHAP e LIME para o naive bayes


def naivebayes():
    dados = load_dataset()
    dados.dropna(inplace=True)
    X = dados.drop(["DDoS"], axis=1)
    Y = dados['DDoS']

    # Aplicar BorderlineSMOTE para balancear as classes
    borderline_smote = BorderlineSMOTE(random_state=42, k_neighbors=5)
    X_resampled, Y_resampled = borderline_smote.fit_resample(X, Y)

    # Aplicar SMOTEENN para limpar o ruido
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, Y_resampled = smote_enn.fit_resample(X_resampled, Y_resampled)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_resampled, Y_resampled, test_size=0.2, random_state=42, stratify=Y_resampled)

    param_dist = {'var_smoothing': np.logspace(-9, 0, num=100)}
    random_search = RandomizedSearchCV(
        estimator=GaussianNB(),
        param_distributions=param_dist,
        n_iter=100, cv=5,
        scoring='recall',
        random_state=42)
    random_search.fit(X_train, Y_train)

    nb = random_search.best_estimator_
    nb.fit(X_train, Y_train)

    # Extrair medias e variancias
    means = nb.theta_
    variances = nb.var_
    y_proba = nb.predict_proba(X_test)
    # Probabilidade da classe 1
    y_proba = y_proba[:, 1]
    # Ajuste o limite de decisão (threshold) para aumentar recall
    threshold = 0.1
    y_pred_adjusted = (y_proba >= threshold).astype(int)

    # Guardar tudo
    save_model(nb, 'naive_bayes_model.pkl')
    calc_save_metrics(Y_test, y_pred_adjusted, 'metrics_naive_bayes.pkl')
    file_path_data = os.path.join('../metrics', 'data_naive_bayes.pkl')
    with open(file_path_data, 'wb') as f:
        pickle.dump((means, variances, X.columns, np.unique(Y)), f)

    # Distribuições das características por classe
    plot_feature_distributions(X_train, Y_train)

    # Analisar a contribuição de cada característica usando y_proba
    analyze_feature_contributions(X_test, y_proba, means, variances, Y)

    # Usar os nomes das colunas no shap
    feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{
        i}' for i in range(X.shape[1])]

    generate_shap_plot_nb(nb, X_train, X_test, feature_names)
    generate_lime_plot(nb, X_train, X_test, X, 'lime_naiveBayes.png')


# Funcao main
if __name__ == "__main__":
    decisiontree()
    naivebayes()
