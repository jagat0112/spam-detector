import streamlit as st
import joblib

model = joblib.load("scripts/spam_model.pkl")
vectorizer = joblib.load("scripts/tfidf_vectorizer.pkl")

st.title("📧 Spam Message Classifier")

user_input = st.text_area("Enter your SMS message:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]
        prob = model.predict_proba(transformed)[0][1]

        if prediction == 1:
            st.error(f"🚨 Spam! (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ Not Spam (Confidence: {1 - prob:.2f})")
