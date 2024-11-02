import pandas as pd

data = pd.read_csv("heart.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1 = data.copy(deep = True)

df1['Sex'] = le.fit_transform(df1['Sex'])
print(df1['Sex'])

df1['ChestPainType'] = le.fit_transform(df1['ChestPainType'])
print(df1['ChestPainType'])
df1['RestingECG'] = le.fit_transform(df1['RestingECG'])
print(df1['RestingECG'])
df1['ExerciseAngina'] = le.fit_transform(df1['ExerciseAngina'])
print(df1['ExerciseAngina'])
df1['ST_Slope'] = le.fit_transform(df1['ST_Slope'])
print(df1['ST_Slope'])
    
# Modifications in the original dataset will not be highlighted in this deep copy.
# Hence, we use this deep copy of dataset that has all the features converted into numerical values for visualization & modeling purposes.
