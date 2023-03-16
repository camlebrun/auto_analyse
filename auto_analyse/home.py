import pandas as pd
import streamlit as st
from io import StringIO
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
st.set_page_config(layout="wide")
st.title("Analyse de données")

# Définir dataframe comme une variable globale
dataframe = None

# Onglets
tab1, tab2, tab3, tab4= st.tabs(["Analyse", "Modélisation", "Correlation","Prédiction"])

# Page d'analyse
with tab1:
    uploaded_file = st.file_uploader("Choisissez un fichier CSV")
    gauche, droite = st.columns(2)
    if uploaded_file is not None:
                if uploaded_file.type == 'text/csv':
                    with st.spinner("Analyse en cours..."):
                        with gauche:
                            sep = st.selectbox("Sélectionner un séparateur :", [",", ";", "\t"])
                        with droite:
                            encoding = st.selectbox("Sélectionner un formatage :", ["UTF-8", "ISO-8859-1"])
                        if sep == "\t":
                            dataframe = pd.read_csv(StringIO(uploaded_file.getvalue().decode(encoding)), sep="\t")
                            st.dataframe(dataframe)
                        else:
                            dataframe = pd.read_csv(StringIO(uploaded_file.getvalue().decode(encoding)), sep=sep)
                            st.dataframe(dataframe)
                        with droite:
                            st.write("Type de données :")
                            st.dataframe(dataframe.dtypes)
                        with gauche:
                            st.write("Nombre de valeurs manquantes :")
                            st.dataframe(dataframe.isnull().sum())
                        st.write("Nombre de lignes et colonnes",dataframe.shape)
                        st.write("Statistiques descriptives :")
                        st.dataframe(dataframe.describe())
    else:
                    st.error("Le fichier n'a pas été chargé correctement. Veuillez vérifier le format du fichier et réessayer.")


# Page de modélisation
with tab2:
    st.write("Modélisation")
    st.write("Le code est bon, mais et long a charger")
    if dataframe is not None:
        fig_pairplot = sns.pairplot(dataframe, diag_kind='kde', corner=True)
        fig_pairplot.fig.set_size_inches(15, 10)
        axes = fig_pairplot.axes
        for i in range(len(axes)):
            for j in range(len(axes)):
                if i == len(axes)-1:
                    axes[i][j].xaxis.label.set_rotation(90)
                    axes[i][j].xaxis.labelpad = 15
                if j == 0:
                    axes[i][j].yaxis.label.set_rotation(0)
                    axes[i][j].yaxis.label.set_ha('right')
                    axes[i][j].yaxis.labelpad = 15
        st.pyplot(fig_pairplot)
        plt.clf()


with tab3:
    st.write("Correlation")
    if dataframe is not None:
        corr_matrix = round (dataframe.corr(),2)
        headmap_cor = sns.heatmap(corr_matrix, annot=True, cmap='Reds', linewidths=0.2)
        headmap_cor = headmap_cor.get_figure()
        headmap_cor.set_size_inches(8, 6)
        st.pyplot(headmap_cor)


with tab4:
    st.write("Prédiction")
    d,g = st.columns(2)
    if dataframe is not None:

        col_list = list(dataframe.columns[:-1].unique())
        selected_columns_exp = st.multiselect("Sélectionner la ou les valeur(s) explicatives", dataframe.columns, default=col_list)
        unselected_columns = list(set(col_list) - set(selected_columns_exp))
        selected_columns_pred = st.multiselect("Sélectionner la ou les valeur(s) à prédire", dataframe.columns, default=[dataframe.columns[-1]])

        X = dataframe[selected_columns_exp]
        y = dataframe[selected_columns_pred]

        # Split des données en train et test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        normalisation = st.selectbox("Choisir une méthode de normalisation", ["Aucune", "MinMax Scaler", "Standard Scaler"])
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
        
        model = st.selectbox("Choisir un modèle", ["Régression linéaire simple", "Régression logistique"])
        if model == "Régression linéaire simple":
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
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Valeurs réelles")
            ax.set_ylabel("Valeurs prédites")
            st.pyplot(fig)
            plt.clf()