import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

data=pd.read_csv("Downloads/Decision_Tree.csv")
print("Original Dataset:")
print(data)
temp_encoder = LabelEncoder()
vib_encoder = LabelEncoder()
fail_encoder = LabelEncoder()

data["Temperature"] = temp_encoder.fit_transform(data["Temperature"])
data["Vibration"] = vib_encoder.fit_transform(data["Vibration"])
data["Failure"] = fail_encoder.fit_transform(data["Failure"])
print("Encoded Dataset:")
print(data)

x = data[["Temperature", "Vibration"]]
y = data["Failure"]

print("Independent Data:")
print(x)
print("Dependent Data:")
print(y)

model = DecisionTreeClassifier(criterion="entropy")
model.fit(x, y)

sample = [[
    temp_encoder.transform(["High"])[0],
    vib_encoder.transform(["Medium"])[0]
]]


prediction = model.predict(sample)
print("Encoded Prediction=",prediction)

if prediction==1:
    print("The mechine will fail")
elif prediction==0:
    print("The mechine will not fail")

plt.figure(figsize=(8,6))
tree.plot_tree(model, filled=True, feature_names=["Temperature", "Vibration"], class_names=["No Failure", "Failure"])
plt.show()