import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib


data = pd.read_csv("Downloads/insurance.csv")
print("DATA=",data)
sex_encoder = LabelEncoder()
smoker_encoder = LabelEncoder()
region_encoder = LabelEncoder()

data["sex"] = sex_encoder.fit_transform(data["sex"])
data["smoker"] = smoker_encoder.fit_transform(data["smoker"])
data["region"] = region_encoder.fit_transform(data["region"])

x = data.drop("charges", axis=1)
y = data["charges"]

print('Independent variable=',x)
print('Dependant variable=',y)

x_scaler = StandardScaler()
x_scaled = x_scaler.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

error= mean_squared_error(y_test, y_pred)
rms= np.sqrt(error)

print("Error=",error)
print("RMS=",rms)


joblib.dump(model, "insurance_model.pkl")
joblib.dump(x_scaler, "x_scaler.pkl")
joblib.dump(sex_encoder, "sex_encoder.pkl")
joblib.dump(smoker_encoder, "smoker_encoder.pkl")
joblib.dump(region_encoder, "region_encoder.pkl")

print("Model, scaler and encoders saved successfully!")