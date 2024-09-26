import streamlit as st


def main():
    st.title("Shap - Decision Tree - dataset_short")

    st.header("Sumário dos valores SHAP para a classe 1(Ataque DDoS)")
    st.image('images/shap_dTree/shap_dTree_class1.png')

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
        f'images/shap_dTree/shap_dTree_class1_dependence_feature{selected_feature_index}.png')

    st.title("Shap - Decision Tree - dataset_series")

    st.header("Sumário dos valores SHAP para a classe 1(Ataque DDoS)")
    st.image('images/shap_dTree_series/shap_dTree_series_class1.png')

    feature_names_series = {
        0: 'dl_bitrate',
        1: 'ul_bitrate',
        2: 'cel_x_dl_retx',
        3: 'cel_x_dl_tx',
        4: 'cel_x_ul_retx',
        5: 'cel_x_ul_tx',
        6: 'ul_total_bytes_non_incr',
        7: 'dl_total_bytes_non_incr',
        8: 'dl_bitrate-1',
        9: 'ul_bitrate-1',
        10: 'cel_x_dl_retx-1',
        11: 'cel_x_dl_tx-1',
        12: 'cel_x_ul_retx-1',
        13: 'cel_x_ul_tx-1',
        14: 'ul_total_bytes_non_incr-1',
        15: 'dl_total_bytes_non_incr-1',
        16: 'dl_bitrate-2',
        17: 'ul_bitrate-2',
        18: 'cel_x_dl_retx-2',
        19: 'cel_x_dl_tx-2',
        20: 'cel_x_ul_retx-2',
        21: 'cel_x_ul_tx-2',
        22: 'ul_total_bytes_non_incr-2',
        23: 'dl_total_bytes_non_incr-2',
    }

    st.header("Graficos SHAP de dependencia para cada feature da classe 1")
    selected_feature_index = st.selectbox('Escolha a Característica', list(
        feature_names_series.keys()), format_func=lambda x: feature_names_series[x])
    selected_feature_name = feature_names_series[selected_feature_index]
    st.subheader(f'Distribuição da Característica: {selected_feature_name}')
    st.image(
        f'images/shap_dTree_series/shap_dTree_series_class1_dependence_feature{selected_feature_index}.png')
