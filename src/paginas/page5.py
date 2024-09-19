import streamlit as st


def main():
    st.title("Analise Shap para Naive Bayes")

    st.header("Sumário dos valores SHAP para a classe 0 (Não ataque)")
    st.image('images/shap_naiveBayes/shap_naiveBayes_class0.png')

    st.header("Sumário dos valores SHAP para a classe 1 (Ataque DDoS)")
    st.image('images/shap_naiveBayes/shap_naiveBayes_class1.png')

    feature_names = {
        0: 'dl_bitrate',
        1: 'ul_bitrate',
        2: 'cel_x_dl_retx',
        3: 'cel_x_dl_tx',
        4: 'cel_x_ul_retx',
        5: 'cel_x_ul_tx',
        6: 'ul_total_bytes_non_incr',
        7: 'dl_total_bytes_non_incr'
    }

    st.header("Graficos SHAP de dependencia para cada feature da classe 1")
    selected_feature_index = st.selectbox('Escolha a Característica', list(
        feature_names.keys()), format_func=lambda x: feature_names[x])
    selected_feature_name = feature_names[selected_feature_index]
    st.subheader(f'Distribuição da Característica: {selected_feature_name}')
    st.image(
        f'images/shap_naiveBayes/shap_naiveBayes_class1_dependence_feature{selected_feature_index}.png')
