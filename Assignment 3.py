import pandas as pd 
from sklearn.neighbors import KNeighborsRegressor
data=pd.read_csv("Desktop/mlai/KNN_Dataset.csv")
x=data[["Temperature"]]
y=data[["Fuel_Consumption"]]

k=3
model=KNeighborsRegressor(n_neighbors=k)
model.fit(x,y)
print('When Temperature = 58, Fuel consumption = ', model.predict([[58]]))
