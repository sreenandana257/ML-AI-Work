import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn. linear_model import LinearRegression
import joblib
data= pd.read_csv("Desktop/power_data.csv")
x= data[["Wind_Speed","Blade_Angle","Rotor_Speed"]]
y= data["Power_Output"]
model=LinearRegression()
model.fit(x,y)
print("Coeff:",model.coef_)
print("Intercept:",model.intercept_)

joblib.dump(model,"power.pkl")