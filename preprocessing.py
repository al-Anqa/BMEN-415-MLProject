import pandas as pd
input = pd.read_csv('ObesityDataSet_Regression.csv')
# print(input)
# print(input.describe())

# print(input['family_history_with_overweight'])
output = input.copy()

output['family_history_with_overweight'] = [1 if x == 'yes' else 0 for x in output['family_history_with_overweight']]
output['Gender'] = [1 if x == 'Female' else 0 for x in output['Gender']]
print(output['family_history_with_overweight'])
print(output['Gender'])