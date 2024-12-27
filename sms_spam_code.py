import pickle
import string
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load pre-trained models
ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Custom CSS for modern UI with visible background image
st.markdown("""
    <style>
        /* Set the background image on the main container */
        [data-testid="stAppViewContainer"] {
            background: url('https://unsplash.com/photos/a-lone-tree-in-a-field-with-mountains-in-the-background-u8kwr3pWVA4') no-repeat center center fixed;
            background-size: cover;
        }

        /* Add a black overlay to enhance contrast */
        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7); /* Black overlay */
            z-index: -1;
        }

        /* Style the main title */
        .main-title {
            font-size: 36px;
            color: #ffffff;
            text-align: center;
            font-weight: bold;
        }

        /* Style the subtitle */
        .sub-title {
            font-size: 18px;
            text-align: center;
            color: #dcdcdc;
        }

        /* Style for the results */
        .result-header {
            font-size: 24px;
            text-align: center;
            margin-top: 20px;
        }

        .spam-result {
            color: red;
            font-weight: bold;
        }

        .not-spam-result {
            color: lightgreen;
            font-weight: bold;
        }

        /* Style buttons */
        .stButton > button {
            background-color: #ffffff;
            color: #4CAF50;
            font-size: 16px;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease, color 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #45a049;
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<h1 class="main-title">Spam Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Detect whether a message is spam or not in seconds!</p>', unsafe_allow_html=True)

# Input field
input_sms = st.text_area("Enter the message", placeholder="Type your message here...")

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Predict button
if st.button('Predict'):
    # Preprocessing
    transformed_text = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transformed_text])
    # Predict
    result = model.predict(vector_input)[0]
    # Display result
    if result == 1:
        st.markdown('<h2 class="result-header spam-result">🚨 It\'s a Spam Message!</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 class="result-header not-spam-result">✅ It\'s Not a Spam Message!</h2>', unsafe_allow_html=True)






