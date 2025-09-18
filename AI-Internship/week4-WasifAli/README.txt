===========================================
   Week 4 – AI Internship @ Code Saviours
        Capstone Project: Spam Classifier
===========================================

This folder contains the final capstone project of my AI Internship 
with Code Saviours SMC Pvt. Ltd. – an **Email Spam Classifier** 
built using Natural Language Processing (NLP) and deployed with Streamlit.

-------------------------------------------
1. Data Preprocessing
-------------------------------------------
File: train_model.py

Description:
- Handles text preprocessing (lowercasing, punctuation removal).
- Uses TF-IDF Vectorization to convert text into numerical features.
- Splits dataset into training and testing sets.

How to Run:
python train_model.py
→ Generates two files:
   - spam_classifier_model.pkl
   - vectorizer.pkl

-------------------------------------------
2. Model Training
-------------------------------------------
File: train_model.py

Description:
- Trains a machine learning model to classify emails as **Spam** or **Not Spam**.
- Uses Naive Bayes for text classification.
- Saves trained model and vectorizer for deployment.

-------------------------------------------
3. Deployment with Streamlit
-------------------------------------------
File: app.py

Description:
- Loads the trained model (`spam_classifier_model.pkl`) 
  and vectorizer (`vectorizer.pkl`).
- Provides a simple web interface to enter email text.
- Predicts whether the email is **Spam** or **Not Spam** with confidence score.

How to Run:
streamlit run app.py
→ Opens the web app in your browser.

-------------------------------------------
4. Screenshots
-------------------------------------------
Screenshots folder includes:
- Model training output
- Streamlit interface
- Example predictions (Spam & Not Spam)

-------------------------------------------
Requirements
-------------------------------------------
- Python 3.x
- Libraries:
    pandas
    scikit-learn
    joblib
    streamlit

Install requirements:
pip install pandas scikit-learn joblib streamlit

-------------------------------------------
Credits
-------------------------------------------
Developed by Wasif Ali
Week 4 – AI Internship @ Code Saviours
