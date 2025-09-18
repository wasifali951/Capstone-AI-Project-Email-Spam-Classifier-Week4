import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page config
st.set_page_config(page_title="📧 Email Spam Classifier", page_icon="📩", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stTextArea textarea {
        font-size: 16px !important;
        border: 2px solid #4CAF50 !important;
        border-radius: 10px !important;
    }
    .result-box {
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
    }
    .spam {
        background-color: #ffcccc;
        color: #a10000;
    }
    .ham {
        background-color: #ccffcc;
        color: #006600;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("📩 Email Spam Classifier")
st.write("An AI-powered app to classify emails as **Spam** or **Not Spam** using Machine Learning.")

# Input area
email_input = st.text_area("✉️ Enter your email text below:", height=150)

if st.button("🔍 Check Email"):
    if email_input.strip() == "":
        st.warning("⚠️ Please enter an email text to analyze.")
    else:
        # Prediction
        X = vectorizer.transform([email_input])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        # Confidence
        confidence = round(max(proba) * 100, 2)

        # Display result with styling
        if prediction == "spam":
            st.markdown(
                f"<div class='result-box spam'>🚨 This email appears to be **SPAM** <br> Confidence: {confidence}%</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box ham'>✅ This email appears to be **Not Spam** <br> Confidence: {confidence}%</div>",
                unsafe_allow_html=True
            )

# Footer
st.markdown("---")
st.caption("Developed by Wasif Ali | Week 4 – AI Internship @ Code Saviours")
