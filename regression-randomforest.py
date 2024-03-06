import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error


dataset = pd.read_csv('ProcessedObesityDataSet_Regression.csv')
dataset = dataset.drop(index=0, axis=1)
print(dataset.head())

x = dataset.drop(['Weight'], axis=1)
y = dataset.Weight.values

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

regr = RandomForestRegressor(max_depth=10, random_state=0)
regr.fit(x_train, y_train)

y_train_pred = regr.predict(x_train)
y_test_pred = regr.predict(x_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'The training accuracy for the model is {r2_train}')
print(f'The testing accuracy for the model is {r2_test}')

rmse_train = (mean_squared_error(y_train, y_train_pred))**(1/2)
rmse_test = (mean_squared_error(y_test, y_test_pred))**(1/2)

print(f'The training RMSE for the model is {rmse_train}')
print(f'The testing RMSE for the model is {rmse_test}')