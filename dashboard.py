import streamlit as st
import pandas as pd
import joblib

# Load trained Random Forest model
model = joblib.load('optimized_rf_model.pkl')

st.title("Student Outcome Prediction Dashboard")
st.markdown("Fill in the student's information to predict the outcome (Dropout, Enrolled, Graduate)")

# --- Categorical Dropdowns ---
# Tuition fees up to date
tuition_up_to_date = st.selectbox("Tuition Fees Up to Date", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")

# Scholarship holder
scholarship_holder = st.selectbox("Scholarship Holder", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")

# Application mode dropdown
application_mode_dict = {
    1: "1st phase - general", 2: "Ordinance No. 612/93", 5: "1st phase - special (Azores)",
    7: "Holders of other higher courses", 10: "Ordinance No. 854-B/99", 15: "International student (bachelor)",
    16: "1st phase - special (Madeira)", 17: "2nd phase - general", 18: "3rd phase - general",
    26: "Ordinance No. 533-A/99, b2)", 27: "Ordinance No. 533-A/99, b3)", 39: "Over 23 years old",
    42: "Transfer", 43: "Change of course", 44: "Technological specialization diploma holders",
    51: "Change of institution/course", 53: "Short cycle diploma holders", 57: "Change of institution/course (International)"
}
application_mode = st.selectbox("Application Mode", list(application_mode_dict.keys()), format_func=lambda x: application_mode_dict[x])

# Course dropdown
course_dict = {
    33:"Biofuel Production Technologies", 171:"Animation and Multimedia Design", 8014:"Social Service (evening)",
    9003:"Agronomy", 9070:"Communication Design", 9085:"Veterinary Nursing", 9119:"Informatics Engineering",
    9130:"Equinculture", 9147:"Management", 9238:"Social Service", 9254:"Tourism", 9500:"Nursing",
    9556:"Oral Hygiene", 9670:"Advertising and Marketing Management", 9773:"Journalism and Communication",
    9853:"Basic Education", 9991:"Management (evening)"
}
course = st.selectbox("Course", list(course_dict.keys()), format_func=lambda x: course_dict[x])

# Father's occupation dropdown
father_occupation_dict = {
    0:"Student", 1:"Director/Manager", 2:"Specialist Intellectual/Scientific",
    3:"Intermediate Technician", 4:"Administrative Staff", 5:"Personal Services/Security",
    6:"Farmer/Skilled Worker Agriculture", 7:"Skilled Industry Worker", 8:"Machine Operator",
    9:"Unskilled Worker", 10:"Armed Forces", 90:"Other Situation", 99:"Blank"
}
father_occupation = st.selectbox("Father's Occupation", list(father_occupation_dict.keys()), format_func=lambda x: father_occupation_dict[x])

# --- Numeric Inputs ---
curr_2nd_approved = st.number_input("Curricular Units 2nd Sem Approved", min_value=0)
curr_2nd_grade = st.number_input("Curricular Units 2nd Sem Grade", min_value=0.0, max_value=20.0, step=0.1)
curr_1st_approved = st.number_input("Curricular Units 1st Sem Approved", min_value=0)
curr_1st_grade = st.number_input("Curricular Units 1st Sem Grade", min_value=0.0, max_value=20.0, step=0.1)
age_at_enrollment = st.number_input("Age at Enrolment", min_value=15, max_value=100)
curr_2nd_evaluations = st.number_input("Curricular Units 2nd Sem Evaluations", min_value=0)
admission_grade = st.number_input("Admission Grade", min_value=0, max_value=200)
prev_qualification_grade = st.number_input("Previous Qualification Grade", min_value=0, max_value=200)
curr_1st_evaluations = st.number_input("Curricular Units 1st Sem Evaluations", min_value=0)
curr_2nd_enrolled = st.number_input("Curricular Units 2nd Sem Enrolled", min_value=0)

# --- Prediction Button ---
if st.button("Predict Outcome"):
    input_data = pd.DataFrame({
        'Curricular units 2nd sem (approved)': [curr_2nd_approved],
        'Curricular units 2nd sem (grade)': [curr_2nd_grade],
        'Curricular units 1st sem (approved)': [curr_1st_approved],
        'Curricular units 1st sem (grade)': [curr_1st_grade],
        'Tuition fees up to date': [tuition_up_to_date],
        'Age at enrollment': [age_at_enrollment],
        'Curricular units 2nd sem (evaluations)': [curr_2nd_evaluations],
        'Admission grade': [admission_grade],
        'Previous qualification (grade)': [prev_qualification_grade],
        'Scholarship holder': [scholarship_holder],
        'Curricular units 1st sem (evaluations)': [curr_1st_evaluations],
        'Application mode': [application_mode],
        'Course': [course],
        'Curricular units 2nd sem (enrolled)': [curr_2nd_enrolled],
        "Father's occupation": [father_occupation]
    })
     # --- Clean feature names
    input_data.columns = [c.strip().replace("(", "_").replace(")", "_").replace(" ","_").replace(" ","_") for c in input_data.columns]

    classes = model.classes_  
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Outcome: {prediction}") 