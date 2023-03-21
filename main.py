import pandas as pd
import time
import math
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
st.markdown(
    """
    ### Cette application permet d'analyser des données de manière automatique.
    """)

# Définir dataframe comme une variable globale
dataframe = None

# Onglets
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Upload & data exploration", "Traitements des données", "Scatter plot", "Pairplot", "Corrélation", "Prédictions"])

# Page d'analyse
with tab1:
    uploaded_file = st.file_uploader("Choisissez un fichier CSV")
    gauche, droite = st.columns(2)
    if uploaded_file is not None:
        if uploaded_file.type == 'text/csv':
            with st.spinner("Analyse en cours..."):
                with gauche:
                    sep = st.selectbox(
                        "Sélectionner un séparateur :", [
                            ",", ";", "tab"])
                with droite:
                    encoding = st.selectbox(
                        "Sélectionner un formatage :", [
                            "UTF-8", "ISO-8859-1", "ASCII", "UTF_8_SIG", "UTF_16", "CP437"])
                if sep == sep:
                    dataframe = pd.read_csv(
                        StringIO(
                            uploaded_file.getvalue().decode(encoding)),
                        sep=sep)
                    st.dataframe(dataframe)

                else:
                    dataframe = pd.read_csv(
                        StringIO(
                            uploaded_file.getvalue().decode(encoding)),
                        sep="/t")
                    st.dataframe(dataframe)
                with droite:
                    float_cols = dataframe.select_dtypes(
                        include=['float64']).columns
                    dataframe[float_cols] = dataframe[float_cols].astype(
                        'float32')
                    st.write("Type de données :")
                    st.dataframe(dataframe.dtypes)
                with gauche:
                    st.write("Nombre de valeurs manquantes :")
                    null = dataframe.isnull().sum()
                    st.dataframe(null)
                st.write("Nombre de lignes et colonnes", dataframe.shape)
                st.write("Statistiques descriptives :")
                st.dataframe(dataframe.describe())
            if dataframe is not None:
                st.title("Visualisation des données")
                col_list = list(dataframe.columns[:-1].unique())
                x_val = st.selectbox("Sélectionner la valeur en x", col_list)
                y_val = st.selectbox("Sélectionner la valeur en y", col_list)
            if x_val == y_val:
                st.info("X et Y doivent être différentes")
            elif dataframe is not None and x_val is not None and y_val is not None:
                with st.spinner('Wait for it...'):
                    time.sleep(5)
                    fig, ax = plt.subplots()
                    ax.scatter(dataframe[x_val], dataframe[y_val])
                    ax.set_xlabel(x_val)
                    ax.set_ylabel(y_val)
                    st.pyplot(fig)
                    plt.clf()
    else:
        st.error(
            "Le fichier n'a pas été chargé correctement. Veuillez vérifier le format du fichier et réessayer.")


if dataframe is not None:
    with st.spinner('Wait for it...'):
        time.sleep(5)
    st.title("Prétraitement")
    dict_null = dict(null)
    col = dataframe.columns
import numpy as np
from sklearn.preprocessing import LabelEncoder


def preprocess_column(column, option, x=None):
    null_counts = column.isnull().sum()
    if null_counts > 0:
        if option == "Pas de modification":
            pass
        elif option == "Supprimer NaN":
            column = column.fillna(value=np.nan).dropna()
        elif option == "Fill par 0":
            column = column.fillna(0)
        elif option == "Fill Mean":
            column = column.fillna(column.mean())
        elif option == "Fill Median":
            column = column.fillna(column.median())
        elif option == "Fill with value":
            column = column.fillna(x)
    return column
def preprocess_column_without_nan(column, option, x=None):
    null_counts = column.isnull().sum()
    if null_counts == 0:
        if option == "Pas de modification":
            pass
        elif option == "encoding":
            column = LabelEncoder().fit_transform(column)
        elif option == "Arrondir":
            column = column.round(x)
        return column
dataframe_0 = dataframe.copy()
col1, col2 = st.columns(2)
for col in dataframe_0.columns:
    null_counts = dataframe_0[col].isnull().sum()
    with col1:
        if null_counts > 0:
            option_col2 = st.radio(
                f"Choisir une option de prétraitement pour la colonne {col}",
                [
                    "Pas de modification",
                    'Supprimer NaN',
                    "Fill par 0",
                    "Fill Mean",
                    "Fill Median",
                    "Encoding",
                    "Arrondir"
                ],
                key=f"{col}_option_col1")
            processed_col = preprocess_column(dataframe_0[col].copy(), option_col2, x = None)
    with col2:
        if null_counts == 0:
            option_col2 = st.radio(
                f"Choisir une option de prétraitement pour la colonne {col}",
                [
                    "Pas de modification",
                    "Encoding",
                    "Arrondir"
                ],
                key=f"{col}_option_col1")
            processed_col = preprocess_column_without_nan(dataframe_0[col].copy(), option_col2, x = None)

