import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error
from common import regression_data

# Get regression data from common.py regression function
x, y = regression_data()

print(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

regr1 = RandomForestRegressor(max_depth=10, random_state=0)
regr1.fit(x_train, y_train)

regr2 = SVR(kernel='linear')
regr2.fit(x_train, y_train)

vot_regr = VotingRegressor(estimators=[('rfr', regr1), ('svr', regr2)])
vot_regr.fit(x_train, y_train)

y_train_pred = vot_regr.predict(x_train)
y_test_pred = vot_regr.predict(x_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'The training accuracy for the model is {r2_train}')
print(f'The testing accuracy for the model is {r2_test}')

rmse_train = (mean_squared_error(y_train, y_train_pred))**(1/2)
rmse_test = (mean_squared_error(y_test, y_test_pred))**(1/2)

print(f'The training RMSE for the model is {rmse_train}')
print(f'The testing RMSE for the model is {rmse_test}')