import pandas as pd

'''
This is designed to serve as a hub for the filtered and split datasets to simplify the actual files.
'''

def regression_data():
    '''
    Function that returns the processed obesity dataset.
    Outputs:
    x -- pandas array of the processed input sans the weight
    y -- array of weights
    '''
    dataset = pd.read_csv('datasets/ProcessedObesityDataSet_Regression.csv')
    print(dataset.head())

    x = dataset.drop(['Weight'], axis=1)
    y = dataset.Weight.values
    return x, y

def classification_data():
    '''
    Function that returns the processed diabeters dataset.
    Outputs:
    x -- pandas array of the processed input sans the outcome
    y -- array of outcomes
    '''
    dataset = pd.read_csv('datasets/ProcessedDiabetesDataset_Classification.csv')
    dataset = dataset.drop(['Unnamed: 0', 'Id'], axis=1)
    # print(dataset.head())

    x = dataset.drop(['Outcome'], axis=1)
    y = dataset.Outcome.values
    return x, y
