import joblib
model=joblib.load("power.pkl")
inp_wind=int(input("Enter a Wind_Speed"))
inp_angle=int(input("Enter the Blade_Angle"))
inp_speed=int(input("Enter the Rotor speed"))
new_power=model.predict([[inp_wind,inp_angle,inp_speed]])
print(new_power)