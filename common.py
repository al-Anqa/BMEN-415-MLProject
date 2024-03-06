import pandas as pd

def regression_data():
    dataset = pd.read_csv('datasets/ProcessedObesityDataSet_Regression.csv')
    print(dataset.head())

    x = dataset.drop(['Weight'], axis=1)
    y = dataset.Weight.values
    return x, y

def classification_data():
    dataset = pd.readcsv