import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("CEAS_08.csv", encoding='latin1') 

print("Columns:", df.columns)
df.columns = [col.strip().lower() for col in df.columns]

if 'text' not in df.columns or 'label' not in df.columns:
    for col in df.columns:
        if 'body' in col or 'text' in col:
            df.rename(columns={col: 'text'}, inplace=True)
        elif 'label' in col or 'class' in col:
            df.rename(columns={col: 'label'}, inplace=True)

df.dropna(subset=['text', 'label'], inplace=True)

if df['label'].dtype == 'object':
    df['label'] = df['label'].str.lower().map({'phishing': 1, 'legitimate': 0, 'spam': 1})
    df.dropna(subset=['label'], inplace=True)

print("Label distribution:\n", df['label'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

with open("CEAS_detector.pkl", "wb") as f:
    pickle.dump(model, f)

with open("CEAS_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

