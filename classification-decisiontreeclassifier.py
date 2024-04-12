import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from common import classification_data
import seaborn as sns

x, y = classification_data()

print(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

dtc_classification = DecisionTreeClassifier()
dtc_classification.fit(x_train, y_train)

y_train_pred = dtc_classification.predict(x_train)
y_test_pred = dtc_classification.predict(x_test)

train_score = dtc_classification.score(x_train, y_train, sample_weight=None)
test_score = dtc_classification.score(x_test, y_test, sample_weight=None)

print(f'The training accuracy for the model is {train_score}')
print(f'The testing accuracy for the model is {test_score}')

cm = confusion_matrix(y_test, y_test_pred)
print (cm)

plt.figure(1)
sns.heatmap(cm, annot=True, fmt = 'd', cmap='Purples')
plt.xlabel('Non Event Observed         Event Observed')
plt.ylabel('Event Predicted          Non Event Predicted')
plt.title('Decision Tree Classifier Confusion Matrix')
plt.show()

tp, tn, fp, fn = cm[1,1], cm[0,0], cm[0,1], cm[1,0]

accuracy = (tp+tn)/(tp+tn+fn+fp)
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)

print(f'Accuracy = {accuracy}')
print(f'Sensitivity = {sensitivity}')
print(f'Specificity = {specificity}')

score = classification_report(y_test, y_test_pred)
print(score)