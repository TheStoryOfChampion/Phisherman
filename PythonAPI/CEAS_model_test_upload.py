import pickle
# Load them back to verify they are fitted
with open("CEAS_vectorizer.pkl", "rb") as f:
    loaded_vectorizer = pickle.load(f)

with open("CEAS_detector.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Try a test email
test_email = "Your account has been locked. Click here to verify."
X_input = loaded_vectorizer.transform([test_email])
pred = loaded_model.predict(X_input)
proba = loaded_model.predict_proba(X_input)

print("Prediction:", "Phishing" if pred[0] == 1 else "Legitimate")
print("Confidence:", round(max(proba[0]) * 100, 2), "%")
