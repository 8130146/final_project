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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE


@st.cache_resource
def load_dataset():
    data = pd.read_csv('../datasets/dataset_short.csv')
    return data


def decisiontree():
    dados = load_dataset()
    X = dados.drop(["DDoS"], axis=1)
    Y = dados['DDoS']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 20],
        # Usar todas as features disponíveis
        'max_features': ['sqrt', 'log2', None],
        'min_samples_leaf': [1, 2, 5, 10]
    }

    random_search = RandomizedSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=100,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    random_search.fit(X_train, Y_train)

    best_model = random_search.best_estimator_

    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    y_pred = best_model.predict(X_test_df)

    models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    file_path_models = os.path.join(models_dir, 'decision_tree_model.pkl')
    # Guardar o modelo usando o pickle
    with open(file_path_models, 'wb') as f:
        pickle.dump(best_model, f)

    metrics_decision_tree = {}

    # Calcular a precisão
    accuracy = accuracy_score(Y_test, y_pred)
    metrics_decision_tree["accuracy"] = accuracy

    # Calcular a precisão (precision)
    precision = precision_score(Y_test, y_pred, average='binary')
    metrics_decision_tree["precision"] = precision

    # Calcular o recall
    recall = recall_score(Y_test, y_pred, average='binary')
    metrics_decision_tree["recall"] = recall

    # Calcular o F1-score
    f1 = f1_score(Y_test, y_pred, average='binary')
    metrics_decision_tree["f1"] = f1

    metrics_dir = '../metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    file_path = os.path.join(metrics_dir, 'metrics_decision_tree.pkl')

    with open(file_path, 'wb') as file:
        pickle.dump(metrics_decision_tree, file)

    generate_dtree_graph(best_model, X)
    generate_shap_plot_dtree(best_model, X_train, X_test)
    generate_lime_plot_dtree(best_model, X_train, X_test, X)


def generate_dtree_graph(best_model, X):
    # Gerar ficheiro .dot
    img = export_graphviz(best_model, out_file=None,
                          feature_names=X.columns,
                          filled=True, rounded=True,
                          special_characters=True)
    # Visualizar a árvore de decisão
    graph = graphviz.Source(img)
    folder_path = "../images"
    graph.render(os.path.join(folder_path, "grafico_decision_tree"),
                 format='png')  # Guardar o gráfico


def generate_shap_plot_dtree(best_model, X_train, X_test):
    # Calcular os valores SHAP para as previsões do conjunto de teste
    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer.shap_values(X_test)
    # Plotar os valores SHAP
    shap.summary_plot(shap_values, X_test, show=False)
    fig = plt.gcf()  # Obtém a figura atual
    fig.set_size_inches(24, 8)
    output_dir = '../images'
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f'shap_decision_tree.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)


def generate_lime_plot_dtree(best_model, X_train, X_test, X):
    num_features = 8
    # Definir os nomes das características (features)
    feature_names = X.columns

    # Replace inf and NaN values in training data
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(X_train.mean(), inplace=True)

    # Replace inf and NaN values in test data
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

    # Gerar explicação com LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values, mode='classification', feature_names=feature_names)
    instance = X_test.values[0]
    try:
        explanation = explainer.explain_instance(
            instance, best_model.predict_proba, num_features=num_features)
        # Guardar a imagem do LIME em formato PNG
        fig = explanation.as_pyplot_figure()
        output_dir = '../images'
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'lime_decisionTree.png'),
                    bbox_inches='tight', dpi=150)
        plt.close(fig)  # Fechar a figura após guardar
    except ValueError as e:
        print(f"Error in Lime explanation: {e}")


def naivebayes():
    dados = load_dataset()
    dados.dropna(inplace=True)
    X = dados.drop(["DDoS"], axis=1)
    Y = dados['DDoS']

    # Aplicar SMOTE para balancear as classes
    smote = SMOTE(random_state=42)
    X_resampled, Y_resampled = smote.fit_resample(X, Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_resampled, Y_resampled, test_size=0.2, random_state=42, stratify=Y_resampled)

    param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}

    grid_search = GridSearchCV(
        estimator=GaussianNB(), param_grid=param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, Y_train)

    nb = grid_search.best_estimator_
    nb.fit(X_train, Y_train)

    # Extraindo probabilidades
    class_priors = nb.class_prior_
    means = nb.theta_
    variances = nb.var_

    y_pred = nb.predict(X_test)
    y_proba = nb.predict_proba(X_test)

    models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    file_path_models = os.path.join(models_dir, 'naive_bayes_model.pkl')
    # Guardar o modelo usando o pickle
    with open(file_path_models, 'wb') as f:
        pickle.dump(nb, f)

    metrics_naive_bayes = {}

    # Calcular a precisão
    accuracy = accuracy_score(Y_test, y_pred)
    metrics_naive_bayes["accuracy"] = accuracy

    # Calcular a precisão (precision)
    precision = precision_score(Y_test, y_pred, average='binary')
    metrics_naive_bayes["precision"] = precision

    # Calcular o recall
    recall = recall_score(Y_test, y_pred, average='binary')
    metrics_naive_bayes["recall"] = recall

    # Calcular o F1-score
    f1 = f1_score(Y_test, y_pred, average='binary')
    metrics_naive_bayes["f1"] = f1

    metrics_dir = '../metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    file_path = os.path.join(metrics_dir, 'metrics_naive_bayes.pkl')

    with open(file_path, 'wb') as file:
        pickle.dump(metrics_naive_bayes, file)
    print(metrics_naive_bayes)

    file_path_data = os.path.join(metrics_dir, 'data_naive_bayes.pkl')
    with open(file_path_data, 'wb') as f:
        pickle.dump((X_test, Y_test, y_pred, y_proba, class_priors,
                    means, variances, X.columns, np.unique(Y)), f)

    # Guargar graficos
    # Probabilidades a priori
    fig, ax = plt.subplots()
    ax.bar(np.unique(Y), class_priors)
    ax.set_title('Probabilidades a Priori')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Probabilidade')
    fig.savefig('../images/probabilidades_a_priori.png')

    # Distribuições das características por classe
    for feature_name in X.columns:
        fig, ax = plt.subplots()
        for class_idx, class_name in enumerate(np.unique(Y)):
            sns.kdeplot(X_train[Y_train == class_idx]
                        [feature_name], label=class_name, ax=ax)
        ax.set_title(f'Distribuição da Característica: {feature_name}')
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Densidade')
        ax.legend(title='Classe')
        fig.savefig(f'../images/distribuicao_{feature_name}.png')

    # Analisar a contribuição de cada característica usando y_proba
    for i, class_name in enumerate(np.unique(Y)):
        fig, ax = plt.subplots()
        feature_contributions = []
        for j, feature_name in enumerate(X.columns):
            # Para cada característica, calcular a média da probabilidade predita para a classe
            contributions = y_proba[:, i] * \
                (X_test[feature_name] - means[i, j])**2 / (2 * variances[i, j])
            feature_contributions.append(contributions.mean())
        feature_contributions = np.array(feature_contributions)
        ax.bar(X.columns, feature_contributions)
        ax.set_title(
            f'Contribuição das Características para a Classe: {class_name}')
        ax.set_xlabel('Características')
        ax.set_ylabel('Contribuição Média')
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        fig.savefig(f'../images/contribuicao_caracteristicas_{class_name}.png')

    # Use original column names for SHAP plots
    feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{
        i}' for i in range(X.shape[1])]

    generate_shap_plot_nb(nb, X_train, X_test, feature_names)

    generate_lime_plot_nb(nb, X_train, X_test, X)


def generate_shap_plot_nb(best_model, X_train, X_test, feature_names):

    # Summarize the background data to reduce memory usage
    # Adjust the sample size as needed
    X_train_summary = shap.kmeans(X_train, 2)

    # Calcular os valores SHAP para as previsões do conjunto de teste
    explainer = shap.KernelExplainer(best_model.predict_proba, X_train_summary)
    shap_values = explainer.shap_values(X_test, nsamples='auto')

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Ensure shap_values is aligned with the feature names
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names, show=False)
    fig = plt.gcf()
    fig.set_size_inches(24, 8)

    output_dir = '../images'
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'shap_naive_bayes.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)


def generate_lime_plot_nb(best_model, X_train, X_test, X):
    num_features = 8
    # Definir os nomes das características (features)
    feature_names = X.columns.tolist()

    # Convert X_train and X_test to DataFrames to use replace and fillna
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Replace inf and NaN values in training data
    X_train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train_df.fillna(X_train.mean(), inplace=True)

    # Replace inf and NaN values in test data
    X_test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test_df.fillna(X_test.mean(), inplace=True)

    # Gerar explicação com LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_df.values, mode='classification', feature_names=feature_names)
    instance = X_test_df.values[0]
    try:
        explanation = explainer.explain_instance(
            instance, best_model.predict_proba, num_features=num_features)
        # Guardar a imagem do LIME em formato PNG
        fig = explanation.as_pyplot_figure()
        output_dir = '../images'
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'lime_naiveBayes.png'),
                    bbox_inches='tight', dpi=150)
        plt.close(fig)  # Fechar a figura após guardar
    except ValueError as e:
        print(f"Error in Lime explanation: {e}")


if __name__ == "__main__":
    # decisiontree()
    naivebayes()
