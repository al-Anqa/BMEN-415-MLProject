import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix
from common import classification_data

x, y = classification_data()

print(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

gnb_classification = CategoricalNB()
gnb_classification.fit(x_train, y_train)

y_train_pred = gnb_classification.predict(x_train)
y_test_pred = gnb_classification.predict(x_test)

train_score = gnb_classification.score(x_train, y_train, sample_weight=None)
test_score = gnb_classification.score(x_test, y_test, sample_weight=None)

print(f'The training accuracy for the model is {train_score}')
print(f'The testing accuracy for the model is {test_score}')

fig1, ax1 = plt.subplots(1,2,figsize=(8, 5))

y_test_1 = np.where(y_test == 1)
y_test_0 = np.where(y_test == 0)

ax1[0].scatter(x_test['Glucose'].iloc[y_test_1], x_test['BMI'].iloc[y_test_1])
ax1[0].scatter(x_test['Glucose'].iloc[y_test_0], x_test['BMI'].iloc[y_test_0])
plt.show()
# ax1[0].plot(x_test, y_test)

cm = confusion_matrix(y_test, y_test_pred)
print (cm)

tp, tn, fp, fn = cm[1,1], cm[0,0], cm[0,1], cm[1,0]

accuracy = (tp+tn)/(tp+tn+fn+fp)
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)

print(f'Accuracy = {accuracy}')
print(f'Sensitivity = {sensitivity}')
print(f'Specificity = {specificity}')