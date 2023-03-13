import pandas as pd
import streamlit as st
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt

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
    try:
        if uploaded_file is not None:
            if uploaded_file.type == 'text/csv':
                with st.spinner("Analyse en cours..."):
                    with gauche:
                        sep = st.selectbox("Sélectionner un séparateur :", [",", ";", "\t"])
                    with droite:
                        encoding = st.selectbox("Sélectionner un formatage :", ["UTF-8", "ISO-8859-1"])
                    # Lire le fichier CSV avec le séparateur et le formatage choisis
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


    except:
                st.error("Le fichier n'a pas été chargé correctement. Veuillez vérifier le format du fichier et réessayer.")


# Page de modélisation
with tab2:
    st.write("Modélisation")
    st.write("Le code est bon, mais et long a charger")
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
    corr_matrix = dataframe.corr()
    headmap_cor = sns.heatmap(corr_matrix, annot=True, cmap='Reds', linewidths=0.2)
    headmap_cor = headmap_cor.get_figure()
    headmap_cor.set_size_inches(8, 6)
    st.pyplot(headmap_cor)


with tab4:
    st.write("Prédiction")
    d,g = st.columns(2)
    with d:
            columns = dataframe.columns
            selected_columns = st.multiselect("Sélectionner les colonnes X", columns)
            if selected_columns:
                X_val_df = dataframe[selected_columns]
                st.dataframe(X_val_df)
    with g:
            columns = dataframe.columns
            selected_columns = st.multiselect("Sélectionner les colonnes Y", columns)
            if selected_columns:
                Y_val_df = dataframe[selected_columns]
                st.dataframe(Y_val_df)
    