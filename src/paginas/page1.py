import pandas as pd
import streamlit as st
import altair as alt
import os


@st.cache_data
def load_separate_by_imei(file):
    if file is not None:
        df = pd.read_csv(file)
        imeis_list = df['imeisv'].unique()
        imeis_dataframes = {}
        for imei in imeis_list:
            imeis_dataframes[imei] = df[df['imeisv'] == imei]
        return imeis_dataframes
    else:
        return {}

# Função para criar gráficos


def create_chart(df, x, y, title):
    return alt.Chart(df).mark_line().encode(
        x=x,
        y=y,
    ).properties(
        width=600,
        height=400,
        title=title
    ).interactive()


def main():
    st.title('Carregar dataset')
    st.write("Faça upload de um dataset(csv)")

    df = st.file_uploader("Selecione o dataset.", type=["csv"])
    imeis_dataframes = load_separate_by_imei(df)

    if imeis_dataframes:
        st.write("Dataset carregado com sucesso!")

        imeis_list = list(imeis_dataframes.keys())
        selected_imei = st.selectbox('Select an IMEI', imeis_list)

        if selected_imei:
            df_selected_ue = imeis_dataframes[selected_imei]
            st.write("Selected Imei: " + str(selected_imei))

        st.write(df_selected_ue.head(20))

        df_selected_ue['_time'] = pd.to_datetime(
            df_selected_ue['_time'], format='ISO8601')

        st.markdown('### Analise de trafego 5G')

        start_date, end_date = st.date_input('Selecione um intervalo de datas', [
            df_selected_ue['_time'].min().date(), df_selected_ue['_time'].max().date()])
        df_selected_ue = df_selected_ue[(df_selected_ue['_time'].dt.date >= start_date) & (
            df_selected_ue['_time'].dt.date <= end_date)]

        # Organizar gráficos em duas colunas
        col1, col2 = st.columns(2)

        with col1:
            chart = create_chart(df_selected_ue, '_time:T',
                                 'dl_bitrate:Q', 'Taxa de bits de transmissoes Downlink')
            st.altair_chart(chart)

            chart = create_chart(
                df_selected_ue, '_time:T', 'cell_x_dl_retx:Q', 'Número de blocos retransmitidos no Downlink')
            st.altair_chart(chart)

            chart = create_chart(
                df_selected_ue, '_time:T', 'cell_x_dl_tx:Q', 'Blocos de dados enviados com sucesso da rede para os dispositivos conectados a essa célula')
            st.altair_chart(chart)

            chart = create_chart(
                df_selected_ue, '_time:T', 'dl_total_bytes_non_incr:Q', 'Bytes de downlink que não foram transmitidos com sucesso.')
            st.altair_chart(chart)

        with col2:
            chart = create_chart(df_selected_ue, '_time:T',
                                 'ul_bitrate:Q', 'Taxa de Bits de tranmissões Uplink')
            st.altair_chart(chart)

            chart = create_chart(
                df_selected_ue, '_time:T', 'cell_x_ul_retx:Q', 'Número de blocos retransmitidos no Uplink')
            st.altair_chart(chart)

            chart = create_chart(df_selected_ue, '_time:T',
                                 'cell_x_ul_tx:Q', 'Blocos de dados transmitidos com sucesso do UE para a rede')
            st.altair_chart(chart)

            chart = create_chart(
                df_selected_ue, '_time:T', 'ul_total_bytes_non_incr:Q', 'Bytes de Uplink que não foram transmitidos com sucesso.')
            st.altair_chart(chart)


if __name__ == "__main__":
    main()
