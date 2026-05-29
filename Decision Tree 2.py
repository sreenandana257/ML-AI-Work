import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt


data=pd.read_csv("Downloads/data.csv")
print("Original Dataset:")
print(data)

x = data[['tempMode','AQ','USS','CS','VOC','RP','IP','Temperature']]
y = data["fail"]

print("Independent Data:")
print(x)
print("Dependent Data:")
print(y)

model = DecisionTreeClassifier(criterion="entropy")
model.fit(x, y)

sample = [[4,5,3,6,1,45,5,1]]


prediction = model.predict(sample)
print("Prediction=",prediction)

if prediction==1:
    print("The mechine will fail")
elif prediction==0:
    print("The mechine will not fail")

plt.figure(figsize=(12,8))

tree.plot_tree(
    model,
    filled=True,
    feature_names=['tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature'],
    class_names=['No Fail', 'Fail']
)

plt.show()
