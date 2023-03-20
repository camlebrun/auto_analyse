import pandas as pd
import time
import math
from main import read_data, data_preprocessing, plot
import streamlit as st
from io import StringIO
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

st.set_page_config(layout="wide")
st.title("Analyse de données")

# Définir dataframe comme une variable globale
dataframe = None

# Onglets
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Upload & data exploration", "Traitements des données", "Scatter plot", "Paireplot", "Corrélation", "Prédictions"])

# Page d'analyse
with tab1:
    uploaded_file = st.file_uploader("Choisissez un fichier CSV ou Excel")
    gauche, droite = st.columns(2)
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            with st.spinner("Analyse en cours..."):
                with gauche:
                    sep = st.selectbox(
                        "Sélectionner un séparateur :", [
                            ",", ";", "tab"])
                with droite:
                    encoding = st.selectbox(
                        "Sélectionner un formatage :", [
                            "UTF-8", "ISO-8859-1", "ASCII", "UTF_8_SIG", "UTF_16", "CP437"])
                dataframe = read_data.get_data_csv(
                    uploaded_file, sep, encoding)
                st.dataframe(dataframe)
        if uploaded_file.name.endswith(".xlsx"):
            with st.spinner("Analyse en cours..."):
                dataframe = read_data.get_data_excel(uploaded_file)
                st.dataframe(dataframe)
        if dataframe is not None:
            with gauche:
                st.write(
                    "Nombre de lignes et colonnes",
                    data_preprocessing.shape(dataframe))
                st.write("Statistiques descriptives :")
                st.dataframe(data_preprocessing.describe(dataframe))
            with droite:
                st.write("Type de données :")
                st.dataframe(data_preprocessing.d_type(dataframe))
                st.write("Nombre de valeurs manquantes :")
                st.dataframe(data_preprocessing.null_values(dataframe))
                col_list = dataframe.columns
            x_counter = 1
            y_counter = 1
            x = st.selectbox(
                "Enter the x-axis column name",
                col_list,
                key=f"x_{x_counter}")
            y = st.selectbox(
                "Enter the y-axis column name",
                col_list,
                key=f"y_{y_counter}")
            plot.scatter_plot(dataframe, x, y)
            x_counter += 1
            y_counter += 1


with tab2:
    if dataframe is not None:
        from main import CleanData
        dataframe_copy = dataframe.copy()
        col1, col2 = st.columns(2)
        # Récupération des noms des colonnes
        col_names = list(dataframe_copy.columns)
        x = st.number_input(
            "Choisir le nombre de décimales",
            min_value=1,
            max_value=4,
            value=1)

        # Affichage des options de prétraitement pour chaque colonne dans les
        # deux colonnes
        with col1:
            for col in col_names[:len(col_names) // 2]:
                option = st.radio(
                    f"Choisir une option de prétraitement pour la colonne {col}",
                    [
                        "Pas de modification",
                        "Supprimer NaN",
                        "Fill par 0",
                        "Fill Mean",
                        "Fill Median",
                        "Encoding",
                        "Arrondir"])
                dataframe_copy[col] = CleanData.preprocess_column(
                    dataframe_copy[col], option, x)
        with col2:
            for col in col_names[len(col_names) // 2:]:
                option = st.radio(
                    f"Choisir une option de prétraitement pour la colonne {col}",
                    [
                        "Pas de modification",
                        "Supprimer NaN",
                        "Fill par 0",
                        "Fill Mean",
                        "Fill Median",
                        "Encoding",
                        "Arrondir"])
                dataframe_copy[col] = CleanData.preprocess_column(
                    dataframe_copy[col], option, x)
        st.write(dataframe_copy)

with tab3:
    col_list = dataframe_copy.columns
    x = st.selectbox(
        "Enter the x-axis column name",
        col_list,
        key=f"x_{x_counter}")
    y = st.selectbox(
        "Enter the y-axis column name",
        col_list,
        key=f"y_{y_counter}")
    plot.scatter_plot(dataframe_copy, x, y)
    x_counter += 1
    y_counter += 1


with tab4:
    plot.pair_plot(dataframe_copy)
