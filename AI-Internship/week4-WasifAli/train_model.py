import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Sample dataset (extend with more real examples for better accuracy)
data = {
    "text": [
        "Congratulations! You've won a $1000 gift card. Click here to claim.",
        "Reminder: Your assignment is due tomorrow.",
        "Lowest price guaranteed!!! Buy now.",
        "Your bank account has been suspended. Verify immediately.",
        "Hello friend, are we meeting later?",
        "Urgent! Update your account info to avoid deactivation.",
        "You have been selected for a lottery prize. Claim fast!",
        "Lunch at 2 PM today?",
        "Exclusive offer just for you! Limited time only.",
        "Can you send me the report by tonight?"
    ],
    "label": [
        "spam", "ham", "spam", "spam", "ham",
        "spam", "spam", "ham", "spam", "ham"
    ]
}

df = pd.DataFrame(data)

# Features and labels
X = df["text"]
y = df["label"]

# Vectorization
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model & vectorizer
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
