import joblib
import pandas as pd


model = joblib.load("insurance_model.pkl")
x_scaler = joblib.load("x_scaler.pkl")
sex_encoder = joblib.load("sex_encoder.pkl")
smoker_encoder = joblib.load("smoker_encoder.pkl")
region_encoder = joblib.load("region_encoder.pkl")


age = int(input("Enter Age: "))
sex = input("Enter Sex (male/female): ").lower()
bmi = float(input("Enter BMI: "))
children = int(input("Enter Number of Children: "))
smoker = input("Smoker? (yes/no): ").lower()
region = input("Enter Region (southwest/southeast/northwest/northeast): ").lower()

sex_encoded = sex_encoder.transform([sex])[0]
smoker_encoded = smoker_encoder.transform([smoker])[0]
region_encoded = region_encoder.transform([region])[0]


sex_encoded = sex_encoder.transform([sex])[0]
smoker_encoded = smoker_encoder.transform([smoker])[0]
region_encoded = region_encoder.transform([region])[0]

new_data = pd.DataFrame([[age,sex_encoded,bmi,children,smoker_encoded,region_encoded]], columns=["age","sex","bmi","children","smoker","region"])

new_data_scaled = x_scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

print("Predicted Insurance Cost =", prediction[0])