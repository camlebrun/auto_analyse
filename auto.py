import pandas as pd
import seaborn as sns
from prettytable import PrettyTable
import matplotlib.pyplot as plt
path = input("Enter the path of the file: ")
sep = input('Enter the separator: ')

dataframe = pd.read_csv(path, sep=sep)
table = PrettyTable()   
table.field_names = dataframe.columns


# add data to the table
for row in dataframe.head().itertuples(index=False):
    table.add_row(row)
print(table)
print(f'Number of col and rows: {dataframe.shape}')
table_missing_values = PrettyTable()
table_missing_values.field_names = ["Attribute", "Missing Values"]

# add missing values to the table
for column in dataframe.columns:
    table_missing_values.add_row([column, dataframe[column].isnull().sum()])

# print the table for missing values
print("Missing Values:")
print(table_missing_values)

# create a PrettyTable object for data types
table_data_types = PrettyTable()
table_data_types.field_names = ["Attribute", "Data Type"]

# add data types to the table
for column in dataframe.columns:
    table_data_types.add_row([column, dataframe[column].dtype])

# print the table for data types
print("Data Types:")
print(table_data_types)
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
if fig_pairplot is not None:
    plt.show()

