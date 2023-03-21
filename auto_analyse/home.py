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
st.title("Analyse fde données")

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
                st.dataframe(dataframe.isnull().sum())
            st.write("Nombre de lignes et colonnes", dataframe.shape)
            st.write("Statistiques descriptives :")
            st.dataframe(dataframe.describe())

    else: 
        st.warning("Veuillez choisir un fichier CSV")


with tab2:
    if dataframe is not None:
        st.title("Prétraitement")

        def preprocess_column(column, option, x=None):
            if option == "Pas de modification":
                pass
            if option == "Supprimer NaN":
                column = column.fillna(value=np.nan).dropna()
            elif option == "Fill par 0":
                column = column.fillna(0)
            elif option == "Fill Mean":
                column = column.fillna(column.mean())
            elif option == "Fill Median":
                column = column.fillna(column.median())
            elif option == "Encoding":
                encoder = LabelEncoder()
                column = encoder.fit_transform(column)
            elif option == "Arrondir":
                column = column.round(x)
            return column

        def split_list(lst):
            middle = math.ceil(len(lst) / 2)
            return lst[:middle], lst[middle:]

        col1, col2 = st.columns(2)
        col_names1, col_names2 = split_list(dataframe.columns)
        col1, col2 = st.columns(2)
        dataframe_0 = dataframe.copy()
        x = st.number_input(
            "Choisir le nombre de décimales",
            min_value=1,
            max_value=4,
            value=1)
        with col1:
            for col in col_names1:
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
                processed_col = preprocess_column(dataframe[col], option, x)
                dataframe_0[col] = processed_col
        with col2:
            for col in col_names2:
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
                processed_col = preprocess_column(dataframe[col], option, x)
                dataframe_0[col] = processed_col
            old_val = None
            new_val = None
            replace_val = st.text_input("Remplacer les valeurs", value=old_val)
            by_val = st.text_input("Remplacer par", value=new_val)
            if replace_val != old_val or by_val != new_val:
                dataframe_0_new = dataframe_0.replace(replace_val, by_val)
                dataframe_0 = dataframe_0_new
                old_val = replace_val
                new_val = by_val

        col3, col4 = st.columns(2)
        with col3:
            st.write("Données avant prétraitement")
            st.dataframe(dataframe)
            st.write(dataframe.shape)
        with col4:
            st.write("Données après prétraitement")
            st.dataframe(dataframe_0)
            st.write(dataframe_0.shape)

        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every
            # rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(dataframe_0)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='output.csv',
            mime='text/csv',
        )
with tab3:
    st.title("Visualisation")
    st.write("Les données affichées sont celles du fichier modifiable dans l'onglet traitement des données")
    if dataframe is not None:
        col_list_0 = list(dataframe_0.columns[:-1].unique())
        st.markdown(col_list_0)
        x_val_0 = st.selectbox(
            "Sélectionner la valeur en x",
            col_list_0,
            key='unique_key_1')
        y_val_0 = st.selectbox(
            "Sélectionner la valeur en y",
            col_list_0,
            key='unique_key_2')
        if x_val_0 == y_val_0:
            st.info("X et Y doivent être différentes")
        elif dataframe is not None and x_val_0 is not None and y_val_0 is not None:

            #time.sleep(5)
            fig, ax = plt.subplots()
            ax.scatter(dataframe_0[x_val_0], dataframe_0[y_val_0])
            ax.set_xlabel(x_val_0)
            ax.set_ylabel(y_val_0)
            st.pyplot(fig)
            plt.clf()

# Page de modélisation
with tab4:
    st.write("Modélisation")
    if dataframe is not None:
        dataframe_num = dataframe_0.select_dtypes(include=[np.number])

        fig_pairplot = sns.pairplot(
            dataframe_num, diag_kind='kde', corner=True)
        fig_pairplot.fig.set_size_inches(15, 10)
        axes = fig_pairplot.axes
        for i in range(len(axes)):
            for j in range(len(axes)):
                if i == len(axes) - 1:
                    axes[i][j].xaxis.label.set_rotation(90)
                    axes[i][j].xaxis.labelpad = 15
                if j == 0:
                    axes[i][j].yaxis.label.set_rotation(0)
                    axes[i][j].yaxis.label.set_ha('right')
                    axes[i][j].yaxis.labelpad = 15
        if fig_pairplot is not None:
            st.pyplot(fig_pairplot)
            plt.clf()


with tab5:
    st.write("Correlation")
    if dataframe is not None:
        dataframe_num = dataframe_0.select_dtypes(include=[np.number])

        corr_matrix = round(dataframe_num.corr(), 2)
        headmap_cor = sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='Reds',
            linewidths=0.2)
        headmap_cor = headmap_cor.get_figure()
        headmap_cor.set_size_inches(8, 6)
        if headmap_cor is not None:
            st.pyplot(headmap_cor)


with tab6:
    st.write("Prédiction")
    d, g = st.columns(2)
    if dataframe is not None:

        col_list = list(dataframe_0.columns[:-1].unique())
        selected_columns_exp = st.multiselect(
            "Sélectionner la ou les valeur(s) explicatives",
            dataframe_0.columns)
        unselected_columns = list(set(col_list) - set(selected_columns_exp))
        selected_columns_pred = st.multiselect(
            "Sélectionner la ou les valeur(s) à prédire",
            dataframe_0.columns)

        X = dataframe_0[selected_columns_exp]
        y = dataframe_0[selected_columns_pred]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        normalisation = st.selectbox(
            "Choisir une méthode de normalisation", [
                "Aucune", "MinMax Scaler", "Standard Scaler"])
        model = st.selectbox(
            "Choisir un modèle", [
                "Régression linéaire", "Régression logistique"])
        if len(selected_columns_exp) > 0 and len(selected_columns_pred) > 0:

        # Split des données en train et test

            if normalisation == "Aucune":
                X_train_std = X_train
                X_test_std = X_test
            elif normalisation == "MinMax Scaler":
                scaler = MinMaxScaler()
                X_train_std = scaler.fit_transform(X_train)
                X_test_std = scaler.transform(X_test)
            else:
                scaler = StandardScaler()
                X_train_std = scaler.fit_transform(X_train)
                X_test_std = scaler.transform(X_test)

            if model == "Régression linéaire":

                # Régression linéaire sur les données normalisées
                reg_lin = LinearRegression()
                reg_lin.fit(X_train_std, y_train)

                # Evaluation du modèle sur les données de test
                y_pred = reg_lin.predict(X_test_std)
                r2 = r2_score(y_test, y_pred)
                st.write("R2 score : ", r2.round(decimals=2))
                mse = mean_squared_error(y_test, y_pred)
                st.write("MSE : ", mse.round(decimals=2))
                mape = mean_absolute_percentage_error(y_test, y_pred)
                st.write("MAPE : ", mape.round(decimals=2))
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.set_xlabel("Valeurs réelles")
                ax.set_ylabel("Valeurs prédites")
                st.balloons()
                st.pyplot(fig)
                plt.clf()
            if model == "Régression logistique":
                # Régression logistique sur les données normalisées
                reg_log = LogisticRegression()
                reg_log.fit(X_train_std, y_train)

                # Evaluation du modèle sur les données de test
                y_pred = reg_log.predict(X_test_std)
                r2 = r2_score(y_test, y_pred)
                st.write("R2 score : ", r2.round(decimals=2))
                mse = mean_squared_error(y_test, y_pred)
                st.write("MSE : ", mse.round(decimals=2))
                mape = mean_absolute_percentage_error(y_test, y_pred)
                st.write("MAPE : ", mape.round(decimals=2))
                # fig, ax = plt.subplots()
                # ax.scatter(y_test, y_pred)
                # ax.set_xlabel("Valeurs réelles")
                # ax.set_ylabel("Valeurs prédites")
                # st.pyplot(fig)
                # plt.clf()
