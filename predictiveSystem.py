import pickle
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES=['Pregnancies','Glucose' , 'BloodPressure' , 'SkinThickness','Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

cols= ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

with open (Path(__file__).parent / 'model_pipeline.pkl', 'rb') as file :
    model = pickle.load(file)


def ask(question, maximum, minimum=0, integer=False):
     while True :
        try:
            value=int(input(question)) if integer else float(input(question))
        except ValueError :
            print("Error: enter a number" + ("int." if integer else "."))
            continue
        if value < minimum or value > maximum:
            print (f"Error : illogical value ,enter number between {minimum} and {maximum} .")
            continue
        return value



pregnancies = ask("Enter number of Pregnancies: ", maximum=20 , integer=True)
glucose = ask("Enter Glucose: ", maximum=300)
blood_pressure = ask("Enter Blood Pressure: ", maximum=150)
skin_thickness = ask("Enter Skin Thickness: ", maximum=100)
insulin = ask("Enter Insulin: ", maximum=900)
bmi = ask("Enter BMI: ", maximum=70)
diabetes_pedigree = ask("Enter Diabetes Pedigree Function: ", maximum=2.5)
age = ask("Enter Age: ", minimum=21, maximum=100, integer=True)

input_data = [
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
]

row=pd.DataFrame([input_data], columns=FEATURES)
row[cols]= row[cols].replace(0 , np.nan)

prediction =model.predict(row)[0]

if prediction == 0:
    print("the person isn't diabetic")
else:
    print("the person is diabetic")









