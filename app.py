import nltk
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Input text
text = "This is a sample sentence. It contains some stop words."
st.info(text)
# Tokenize the text into words
words = word_tokenize(text)

# Filter out stop words
stop_words = set(stopwords.words("english"))
filtered_words = [word for word in words if word.casefold() not in stop_words]

# Print the original text, tokenized words, and filtered words
print("Original Text: ", text)
st.write(text)
print("Tokenized Words: ", words)
st.write(words)
print("Filtered Words: ", filtered_words)
st.write(filtered_words)
