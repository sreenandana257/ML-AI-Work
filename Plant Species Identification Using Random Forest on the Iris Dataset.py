import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data= pd.read_csv("Downloads/Iris.csv")
print(data)

label_encoder=LabelEncoder()
data["Species"] =label_encoder.fit_transform(data["Species"])

x=data.drop(['Id',"Species"],axis=1)
print("Independent variables=",x)
y = data[["Species"]]
print("Dependant variable =",y)


x_train,x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(x_train,y_train)
y_predict=model.predict(x_test)
accuracy= accuracy_score(y_test,y_predict)
print('Accuracy=',accuracy)


sepal_length = float(input("Enter Sepal Length: "))
sepal_width = float(input("Enter Sepal Width: "))
petal_length = float(input("Enter Petal Length: "))
petal_width = float(input("Enter Petal Width: "))

user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
print("User input =", user_input)

prediction1 = model.predict(user_input)
print(prediction1)

output= label_encoder.inverse_transform(prediction1)

print("Predicted Flower Species:",output[0])

sepal_length = float(input("Enter Sepal Length: "))
sepal_width = float(input("Enter Sepal Width: "))
petal_length = float(input("Enter Petal Length: "))
petal_width = float(input("Enter Petal Width: "))

user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
print("User input =", user_input)

prediction2 = model.predict(user_input)
print(prediction2)

output= label_encoder.inverse_transform(prediction2)

print("Predicted Flower Species:",output[0])

sepal_length = float(input("Enter Sepal Length: "))
sepal_width = float(input("Enter Sepal Width: "))
petal_length = float(input("Enter Petal Length: "))
petal_width = float(input("Enter Petal Width: "))

user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
print("User input =", user_input)

prediction3 = model.predict(user_input)
print(prediction3)

output= label_encoder.inverse_transform(prediction3)

print("Predicted Flower Species:",output[0])