import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 


dataset = pd.read_csv('ProcessedObesityDataSet_Regression.csv')
print(dataset.head())

x = dataset.drop(['Weight'], axis=1)
y = dataset.Weight.values

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
