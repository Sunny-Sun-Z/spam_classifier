import pickle

#1. load Model

with open("model/spam_classifier_tfidf.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

#2. Predict on New Emails
test_emails = [
    "Free meeting today!",  # Likely spam
    "Buy your statement",   # Likely ham
    "Hello, how are you?"   # New words
]

X_test = vectorizer.transform(test_emails)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Print Results

for email, pred, prob in zip(test_emails, predictions, probabilities):
    print(f"Email: '{email}")
    print(f" Prediction: {'Spam' if pred==1 else 'Ham'}")
    print(f" Confidence: Ham={prob[0]:.2f}, Spam={prob[1]:.2f}\n")

'''
predict file run result: 

Email: 'Free meeting today!
 Prediction: Spam
 Confidence: Ham=0.44, Spam=0.56

Email: 'Buy your statement
 Prediction: Ham
 Confidence: Ham=0.54, Spam=0.46

Email: 'Hello, how are you?
 Prediction: Ham
 Confidence: Ham=0.83, Spam=0.17
'''