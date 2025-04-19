import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

# Define the output directory
# output_dir = '/TEMP/'

# Load the model
with open('model_new_nb.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as vec_file:
    loaded_vectorizer = pickle.load(vec_file)

# # Load the label encoder
# with open('label_encoder.pkl', 'rb') as lbl_file:
#     loaded_label_encoder = pickle.load(lbl_file)


# if loaded_label_encoder is not None:
#     print("Label encoder loaded successfully.")

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)                 # Remove hyperlinks
    text = re.sub(r'[^\w\s]', '', text)                 # Remove punctuation
    text = text.lower()                                 # Lowercase
    text = re.sub(r'\s+', ' ', text).strip()            # Remove extra spaces
    return text

# Prediction function using loaded components
def predict_text(input_text):
    processed = preprocess_text(input_text)
    vectorized = loaded_vectorizer.transform([processed])
    prediction = loaded_model.predict(vectorized)
    return "spam" if prediction[0] == 0 else "not spam"

# --- Sample Testing Below ---

# List of sample inputs
sample_emails = [
    "Thanks for subscribing to our monthly newsletter! You'll now receive updates on our latest features, blog posts, and offers.",
    "Please find attached your pending invoice. Kindly clear the payment at the earliest to avoid penalties. [Click here to pay]",
    "Congratulations! You've won a free iPhone. Click here to claim now.",
    "Reminder: Project deadline is next Monday. Let me know if you need help."
]

# Test predictions
for i, email in enumerate(sample_emails, 1):
    result = predict_text(email)
    print(f"Email {i}: {result}\n→ {email}\n")
