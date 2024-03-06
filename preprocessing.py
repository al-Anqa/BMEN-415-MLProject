import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
input = pd.read_csv('datasets\ObesityDataSet_Regression.csv')

output = input.copy()

# Changing the binary data to 0s and 1s
# Takes list of columns and the values we want to be 1.
ones_col = ['family_history_with_overweight', 'FAVC','Gender', 'SMOKE', 'SCC']
ones = ['yes', 'yes', 'Female', 'yes', 'yes']

i=0
for col in ones_col:
    output[col] = [1 if x == ones[i] else 0 for x in output[col]]
    i += 1

# print(output['family_history_with_overweight'])
# print(output['Gender'])
    
# Convert all non-numeric categorical data to integers.
# Cleaning CAEC
# CALC and CAEC use the same values and thus same keys

caec_calc_dict = {'no': 0, 'Sometimes':1, 'Frequently': 2, 'Always': 3}
output['CAEC'] = output['CAEC'].replace(caec_calc_dict)
output['CALC'] = output['CALC'].replace(caec_calc_dict)

# print(output['CAEC'])
# print(output['CALC'])

# print(output['MTRANS'].unique())
mtrans_dict = {'Walking': 0, 'Bike': 1, 'Public_Transportation': 2, 'Motorbike': 3, 'Automobile': 4}
output['MTRANS'] = output['MTRANS'].replace(mtrans_dict)

# print(output['NObeyesdad'].unique())
nobeyesdad_dict = {'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2, 'Overweight_Level_II': 3, 'Obesity_Type_I': 4, 'Obesity_Type_II': 5, 'Obesity_Type_III': 6}
output['NObeyesdad'] = output['NObeyesdad'].replace(nobeyesdad_dict)

# Now, we round the height and weight to 2 decimals and age to 0 decimals
output['Age'] = output['Age'].round(0)
output['Height'] = output['Height'].round(2)
output['Weight'] = output['Weight'].round(0)

# Lastly we round the synthetic data to whole integers where necessary
output['FCVC'] = output['FCVC'].round(0)
output['NCP'] = output['NCP'].round(0)
output['CH2O'] = output['CH2O'].round(0)
output['FAF'] = output['FAF'].round(0)
output['TUE'] = output['TUE'].round(0)

print(output.describe())

fig, axes = plt.subplots(figsize=(8, 8)) 
sns.heatmap(data=output.corr(), annot=True, linewidths=.5, ax=axes) 
plt.show()

output.to_csv('datasets\ProcessedObesityDataSet_Regression.csv', index=False)