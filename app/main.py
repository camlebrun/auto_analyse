import pandas as pd
import numpy as np
import math
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


class read_data:
    def get_data_csv(file_name, sep, encoding):
        data = pd.read_csv(file_name, sep=sep, encoding=encoding)
        return data

    def get_data_excel(file_name, sheet_name):
        sheet_name = st.text_input("Enter the sheet name", sheet_name)
        data = pd.read_excel(file_name, sheet_name=sheet_name)
        return data


class data_preprocessing:
    def null_values(data):
        null_values = data.isnull().sum()
        return null_values

    def d_type(data):
        d_type = data.dtypes
        return d_type

    def shape(data):
        shape = data.shape
        return shape

    def describe(data):
        describe = data.describe()
        return describe

    def head(data):
        head = data.head()
        return head


class plot:
    def scatter_plot(data, x, y):
        col_list = data.columns
        if x == y:
            st.warning("Please select different columns for x and y axis")
        else:
            fig = plt.figure()
            plt.scatter(data[x], data[y], s=10, alpha=0.5)
            plt.xlabel(x)
            plt.ylabel(y)
            st.pyplot(fig)

    def pair_plot(data):
        if data is not None:
            dataframe_num = data.select_dtypes(include=[np.number])
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


class CleanData:
    @staticmethod
    def preprocess_column(column, option, x=3):

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
        elif option == "Encoding":
            encoder = LabelEncoder()
            column = encoder.fit_transform(column)
        elif option == "Arrondir":
            column = column.round(x)
        return column

    @staticmethod
    def split_list(lst):
        middle = math.ceil(len(lst) / 2)
        return lst[:middle], lst[middle:]
