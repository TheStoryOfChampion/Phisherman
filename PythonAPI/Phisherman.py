import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open('CEAS_detector.pkl', 'rb') as f:
    model = pickle.load(f)

with open('CEAS_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="Phisherman", page_icon="📧")
st.title("Phisherman")
st.write("Detect if an email is **Phishing** or **Legitimate** using AI. Paste the email text or upload a .txt file.")

option = st.radio("Choose an option:", ("Paste email text", "Upload .txt file"))
if option == "Paste email text":
    email_text = st.text_area("Paste the email text here:")
    if st.button("Check"):
        if email_text:
            # Preprocess the input text
            email_text = [email_text]
            email_text_vectorized = vectorizer.transform(email_text)
            prediction = model.predict(email_text_vectorized)
            if prediction[0] == 1:
                st.success("This email is **Phishing**.")
            else:
                st.success("This email is **Legitimate**.")
        else:
            st.warning("Please paste the email text.")
elif option == "Upload .txt file":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        email_text = uploaded_file.read().decode("utf-8")
        st.text_area("Email text:", email_text, height=300)
        if st.button("Check"):
            # Preprocess the input text
            email_text = [email_text]
            email_text_vectorized = vectorizer.transform(email_text)
            prediction = model.predict(email_text_vectorized)
            if prediction[0] == 1:
                st.success("This email is **Phishing**.")
            else:
                st.success("This email is **Legitimate**.")

if st.button("🔍 Detect Phishing"):
    if email_text.strip() == "":
        st.error("❌ Please provide email text!")
    else:
        # Vectorize input
        X_vec = vectorizer.transform([email_text])

        # Predict
        pred = model.predict(X_vec)[0]
        proba = model.predict_proba(X_vec)[0]

        label = "🚨 **Phishing**" if pred == 1 else "✅ **Legitimate**"
        st.subheader(f"Prediction: {label}")

        # Confidence scores
        st.write(f"**Confidence (Phishing): {proba[1]*100:.2f}%**")
        st.write(f"**Confidence (Legit): {proba[0]*100:.2f}%**")

        # ✅ Optional: Show top words contributing to phishing
        st.write("---")
        st.write("### 🔎 Top words contributing to prediction:")

        # Get feature importance
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]

        # Get words present in this email
        email_vec_array = X_vec.toarray()[0]
        present_indices = np.where(email_vec_array > 0)[0]
        present_words = [(feature_names[i], coefficients[i]) for i in present_indices]

        # Sort by weight (importance)
        present_words_sorted = sorted(present_words, key=lambda x: abs(x[1]), reverse=True)[:10]

        df_words = pd.DataFrame(present_words_sorted, columns=["Word", "Importance"])
        st.table(df_words)

st.write("---")
st.write("Made with ❤️ for Cyber Security Projects")