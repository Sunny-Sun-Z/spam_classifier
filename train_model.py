import os 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB 
import pickle

#1. Trianing Data 
train_emails = [
      "Win a free prize! Click now!",  # Spam (1)
    "Meeting at 3pm tomorrow",       # Ham (0)
    "Your account statement",        # Ham (0)
    "Limited offer! Buy today!"      # Spam (1)
]
train_labels = [1,0,0,1]

#2. Preprocess and Train
vectorizer = CountVectorizer(lowercase=True, stop_words = "english")
X_train = vectorizer.fit_transform(train_emails)

model = MultinomialNB(alpha=1)
model.fit(X_train, train_labels)

#3. Save Model and Vecotroizor
os.makedirs("model",exist_ok=True)
with open("model/spam_classifier.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("Model trained and saved to /model/spam_classifier.pkl!")