import streamlit as st
import numpy as np
import joblib

# ======================
# LOAD MODEL
# ======================
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Career Recommendation System", layout="centered")

# ======================
# TITLE
# ======================
st.title("🎓 ML Based Career Recommendation System")
st.markdown("Answer the questions below to find the best career for you 🚀")

# ======================
# QUESTIONS (ORDER MUST MATCH TRAINING DATA)
# ======================
features = [
    'Math_Logic', 'Coding', 'Physics', 'Data_Stats', 'Strategy',
    'Debate', 'Philosophy', 'Fin_Markets', 'Sketching', 'Photography',
    'Writing', 'Performing_Arts', 'Fashion', 'Interior', 'Audio_Eng',
    'UI_UX', 'Hardware_Fix', 'Automotive', 'Model_Building',
    'Civil_Const', 'Tools_Usage', 'Robotics', 'Aviation', 'Lab_Work',
    'Leadership', 'Public_Speaking', 'Psychology', 'Sales', 'Teaching',
    'Organizing', 'HR_Mgmt', 'Social_Service', 'Outdoors', 'Biology',
    'Agriculture', 'Wildlife', 'Medicine', 'Tourism', 'Fitness',
    'Chemistry'
]

questions = {
    'Math_Logic': "Do you enjoy solving math and logic problems?",
    'Coding': "Do you like coding or programming?",
    'Physics': "Are you interested in physics?",
    'Data_Stats': "Do you enjoy working with data and statistics?",
    'Strategy': "Do you like strategic thinking?",
    'Debate': "Do you enjoy debates?",
    'Philosophy': "Are you interested in philosophy?",
    'Fin_Markets': "Do you like finance or stock markets?",
    'Sketching': "Do you enjoy sketching or drawing?",
    'Photography': "Do you like photography?",
    'Writing': "Do you enjoy writing?",
    'Performing_Arts': "Do you like acting or performing arts?",
    'Fashion': "Are you interested in fashion?",
    'Interior': "Do you like interior design?",
    'Audio_Eng': "Are you interested in audio/sound engineering?",
    'UI_UX': "Do you like designing apps/websites?",
    'Hardware_Fix': "Do you enjoy fixing electronics?",
    'Automotive': "Are you interested in automobiles?",
    'Model_Building': "Do you enjoy building models?",
    'Civil_Const': "Are you interested in construction?",
    'Tools_Usage': "Do you like using tools/machines?",
    'Robotics': "Are you interested in robotics?",
    'Aviation': "Do you like aviation?",
    'Lab_Work': "Do you enjoy lab work?",
    'Leadership': "Do you see yourself as a leader?",
    'Public_Speaking': "Do you like public speaking?",
    'Psychology': "Are you interested in psychology?",
    'Sales': "Do you enjoy sales?",
    'Teaching': "Do you like teaching?",
    'Organizing': "Are you good at organizing?",
    'HR_Mgmt': "Are you interested in HR/management?",
    'Social_Service': "Do you like helping people?",
    'Outdoors': "Do you enjoy outdoor activities?",
    'Biology': "Are you interested in biology?",
    'Agriculture': "Do you like agriculture?",
    'Wildlife': "Are you interested in wildlife?",
    'Medicine': "Do you want a medical career?",
    'Tourism': "Do you like traveling?",
    'Fitness': "Are you into fitness?",
    'Chemistry': "Do you enjoy chemistry?"
}

# ======================
# INPUT SECTION
# ======================
inputs = []

st.subheader("📝 Answer All Questions")

for feature in features:
    ans = st.radio(questions[feature], ["No", "Yes"], key=feature)
    inputs.append(1 if ans == "Yes" else 0)

# ======================
# PREDICTION
# ======================
if st.button("🚀 Predict Career"):

    # Safety check
    if len(inputs) != 40:
        st.error("❌ Feature mismatch! Check inputs.")
    else:
        input_array = np.array([inputs])

        prediction = model.predict(input_array)[0]
        probabilities = model.predict_proba(input_array)[0]

        career = le.inverse_transform([prediction])[0]

        st.success(f"🎯 Recommended Career: {career}")

        # ======================
        # TOP 3 CAREERS
        # ======================
        st.subheader("🔥 Top 3 Career Matches")

        top3_idx = np.argsort(probabilities)[-3:][::-1]

        for i in top3_idx:
            st.write(f"{le.inverse_transform([i])[0]} → {round(probabilities[i]*100,2)}%")
