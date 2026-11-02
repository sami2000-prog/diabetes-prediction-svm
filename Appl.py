import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import uuid



st.set_page_config(
    page_title="Hospital - Diabetes System",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "patients.csv"
MODEL_PATH= BASE_DIR / "model_pipeline.pkl"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

Columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]
MAX_UNKNOWN = 2
 

##ML model:

@st.cache_resource
def load_model():
    with open (MODEL_PATH, "rb") as file :
        return pickle.load(file)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load model file: {error}")
    st.stop()

## Patient DB storage :

PATIENT_COLUMNS = [
    "PatientID",
    "RegisteredAt",
    *FEATURES
]


def load_patients():
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=PATIENT_COLUMNS)
    return pd.read_csv(CSV_PATH, dtype={"PatientID": str})


def save_patient(patient_data):
    patients = load_patients()

    new_patient = pd.DataFrame([patient_data])
    patients = pd.concat(
        [patients, new_patient],
        ignore_index=True
    )

    patients.to_csv(CSV_PATH, index=False)

## Delete a patient:
def delete_patient(patient_id):
    patients=load_patients()
    patients=patients[patients["PatientID"] != patient_id]
    patients.to_csv(CSV_PATH , index= False)

def delete_selected_patient():
    patient_id = st.session_state["delete_select"]
    delete_patient(patient_id)
    st.session_state["flash"] = f"Patient {patient_id} deleted"
    st.session_state["delete_confirm"] = False  
    st.session_state.pop("delete_select", None)


## Predicition :

def predict_patient(patient):
    row =pd .DataFrame(
        [[float(patient[column]) for column in FEATURES ]],
        columns=FEATURES
    )

    row[Columns] = row[Columns].replace(0, np.nan)
    return int (model.predict(row)[0])

#  Controle :

def count_unk(patient):
    return sum(float(patient[c]) == 0 for c in Columns)

def check_measurements(patient):
    if float(patient["Glucose"]) == 0:
        return "Glucose is required"
    
    unk = count_unk(patient)
    if unk > MAX_UNKNOWN :
        return f"{unk} measurements are unknown ,mamximum :{MAX_UNKNOWN}"
    return None



## APP Interface :

st.title(" Hospital Patient Management")

tab_register, tab_patients, tab_prediction = st.tabs(
    ["➕ Register Patient", "📋 Patient List", "🧠 Prediction"]
)


## 1: Register a patient 

with tab_register:
    st.subheader("New Patient Registration")

    with st.form("patient_form"):
       
            pregnancies = st.number_input(
                "Pregnancies", min_value=0, max_value=20, step=1
            )
            glucose = st.number_input(
                "Glucose", min_value=0.0 , max_value=300.0
            )
            blood_pressure = st.number_input(
                "Blood Pressure", min_value=0.0 , max_value=150.0
            )
            skin_thickness = st.number_input(
                "Skin Thickness", min_value=0.0 , max_value=100.0
            )
            insulin = st.number_input(
                "Insulin", min_value=0.0 , max_value=900.0
            )
            bmi = st.number_input(
                "BMI", min_value=0.0 , max_value= 70.0
            )
            diabetes_pedigree = st.number_input(
                "Diabetes Pedigree Function", min_value=0.0, max_value=2.5 ,step=0.01
            )
            age = st.number_input(
                "Age", min_value=21,max_value=100 ,value=30, step=1
            )
        
            submitted = st.form_submit_button("Save")

    if submitted:
        patient_id = str(uuid.uuid4())[:8].upper()

        patient_data = {
            "PatientID": patient_id,
            "RegisteredAt": datetime.now().isoformat(
                timespec="seconds"
            ),
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree,
            "Age": age
        }

        problem = check_measurements(patient_data)
        if problem:
            st.error(f"Patient Not registred :{problem}")
        else:
            save_patient(patient_data)

            st.success(
                f"Patient registered successfully. ID: {patient_id}"
            )
            st.info("No prediction was performed.")


## 2: Patient List 

with tab_patients:
    st.subheader("Registered Patients")

    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))


    patients = load_patients()

    st.metric("Total registered patients", len(patients))

    if patients.empty:
        st.info("No patients registered yet")
    else:
        st.dataframe(
            patients,
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.subheader("Delete patient")
        labels={
            row["PatientID"]:
            f"{row['PatientID']} — Age {row['Age']} — {row['RegisteredAt']}"
            for _, row in patients.iterrows()
        }

        st.selectbox(
            "Select the patient to delete",
            list(labels.keys()),
            format_func=lambda patient_id: labels[patient_id],
            key="delete_select"
        )
        confirm = st.checkbox(
            "I confirm that I want to delete this patient permanently",
            key="delete_confirm"
        )
 
        st.button(
            "Delete patient",
            disabled=not confirm,
            on_click=delete_selected_patient
        )


## 3: Predicition 

with tab_prediction:
    st.subheader("Run a Prediction")
    patients = load_patients()

    if patients.empty:
        st.info("Register a patient first")
    else:
        patient_options = {
            f"{row['PatientID']} — Age {row['Age']}":
            row["PatientID"]
            for _, row in patients.iterrows()
        }

        selected_label = st.selectbox(
            "Select a registered patient",
            list(patient_options.keys())
        )

        selected_id = patient_options[selected_label]

        patient_row = patients[
            patients["PatientID"] == selected_id
        ].iloc[0]

        st.write("Patient measurements:")
        st.dataframe(
            patient_row[FEATURES].to_frame("Value"),
            use_container_width=True
        )

        unk=count_unk(patient_row)

        if st.button("Run Prediction", type="primary"):

            problem=check_measurements(patient_row)

            if problem :
                st.error(f"prediction blocked {problem}")

            else:
                try:
                    result = predict_patient(patient_row)

                    if result == 0:
                        st.success(
                            "Not diabetic"
                        )
                    else:
                        st.warning(
                            "Diabetic"
                        )

        
                    if unk:
                        st.info(
                            f"{unk} unknown measurement(s) replaced by the median :"
                            "the prediction is less reliable ,especially if glucose is missing"
                            )

                except Exception as error:
                    st.error(f"Prediction failed: {error}")
