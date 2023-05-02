import pandas as pd
import time
import math
import streamlit as st
from io import StringIO
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")
st.title("Exploratory Data Analysis")

# Define dataframe as a global variable
dataframe = None

# Tabs
eda, pairplot, plot = st.tabs(
    ["Load", "Pairplot","Scatterplot"])

# Data analysis page
with eda:
    uploaded_file = st.file_uploader("Import your CSV file")
    left, right = st.columns(2)
    if uploaded_file is not None:
        if uploaded_file.type == 'text/csv':
            

            with left:
                sep = st.selectbox(
                    "Choose your separator:", [
                        ",", ";", "tab"])
            with right:
                encoding = st.selectbox(
                    "Select your encoding:", [
                        "UTF-8", "ISO-8859-1", "ASCII", "UTF_8_SIG", "UTF_16", "CP437"])
            if sep == "tab":
                dataframe = pd.read_csv(
                    StringIO(
                        uploaded_file.getvalue().decode(encoding)),
                    sep="\t")
                st.write("Number of rows and columns:", dataframe.shape)
            else:
                dataframe = pd.read_csv(
                    StringIO(
                        uploaded_file.getvalue().decode(encoding)),
                    sep=sep)
                st.write("Number of rows and columns:", dataframe.shape)
            with right:
                float_cols = dataframe.select_dtypes(
                    include=['float64']).columns
                dataframe[float_cols] = dataframe[float_cols].astype(
                    'float32')
                st.write("Data type:")
                st.dataframe(dataframe.dtypes, use_container_width=True)
            with left:
                st.write("Number of NaN values:")
                st.dataframe(dataframe.isnull().sum(), use_container_width=True)
        st.write("Descriptive statistics:")
        st.dataframe(dataframe.describe(),  use_container_width=True)
        st.write("Dataframe:")
        st.dataframe(dataframe,use_container_width=True)

    else: 
        st.warning("Please choose a CSV file")

#with nlp:
    
with plot:
    st.title("Visualization")
    if dataframe is not None:
        col_list_0 = list(dataframe.columns[:-1].unique())
        st.markdown(col_list_0)
        x_val_0 = st.selectbox(
            "Select x-axis value:",
            col_list_0,
            key='unique_key_1')
        y_val_0 = st.selectbox(
            "Select y-axis value:",
            col_list_0,
            key='unique_key_2')
        if x_val_0 == y_val_0:
            st.info("X and Y values must be different")
        elif dataframe is not None and x_val_0 is not None and y_val_0 is not None:

            fig, ax = plt.subplots()
            sns.barplot(x=x_val_0, y=y_val_0, data=dataframe)
            ax.set_xlabel(x_val_0)
            ax.set_ylabel(y_val_0)
            st.pyplot(fig)
            plt.clf()

# Page de modélisation
with pairplot:
    st.write("Modelling")
    if dataframe is not None:
        dataframe_num = dataframe.select_dtypes(include=[np.number])

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
