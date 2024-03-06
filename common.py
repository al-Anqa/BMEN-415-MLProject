import pandas as pd

def regression_data():
    dataset = pd.read_csv('datasets/ProcessedObesityDataSet_Regression.csv')
    print(dataset.head())

    x = dataset.drop(['Weight'], axis=1)
    y = dataset.Weight.values
    return x, y

def classification_data():
    dataset = pd.read_csv('datasets/ProcessedDiabetesDataset_Classification.csv')
    dataset = dataset.drop(['Unnamed: 0', 'Id'], axis=1)
    # print(dataset.head())

    x = dataset.drop(['Outcome'], axis=1)
    y = dataset.Outcome.values
    return x, y

print(classification_data())