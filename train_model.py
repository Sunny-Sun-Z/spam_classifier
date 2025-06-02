import os 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB 
import pickle
import tarfile
import urllib.request

# Download and extract SpanAssassin dataset
def download_spam_data():
    url = "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2"
    urllib.request.urlretrieve(url, "easy_ham.tar.bz2")
    with tarfile.open("easy_ham.tar.bz2","r:bz2") as tar:
        tar.extractall("data")
        
    url = "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2"
    urllib.request.urlretrieve(url, "spam.tar.bz2")
    with tarfile.open("spam.tar.bz2", "r:bz2") as tar:
        tar.extractall("data")
    
# load emails from directory      
def load_emails(path, label):
    emails = []
    labels = []
    for fileName in os.listdir(path):
        with open(os.path.join(path, fileName), "r", encoding="latin-1") as f:
            emails.append(f.read())
            labels.append(label)
    return emails, labels
    
# Main training code
if not os.path.exists("data"):
    download_spam_data()
    
ham_emails, ham_labels = load_emails ("data/easy_ham", 0)
spam_emails, spam_labels= load_emails("data/spam", 1)

train_emails = ham_emails + spam_emails
train_labels = ham_labels + spam_labels

print("train_emails: ", train_emails)
print("train_labels ", train_labels)
# Rest of your training code remains the same...
vectorizer = CountVectorizer(lowercase=True, stop_words="english")
X_train = vectorizer.fit_transform(train_emails)

model = MultinomialNB(alpha=1)
model.fit(X_train, train_labels)   

# Saver model and Vecotroizor
os.makedirs("model",exist_ok=True)
with open("model/spam_classifier.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)
    
print(f"Model trained on {len(train_emails)} emails and saved!")
