import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn. linear_model import LinearRegression

data= pd.read_csv("Desktop/data.csv")
x= data[["Load"]]
y= data["Extension"]
model=LinearRegression()
model.fit(x,y)

new_Load=55
new_Extension= model.predict([[new_Load]])
print("The predicted extension for the given load =",new_Extension)
plt.scatter(x,y)
plt.show()
