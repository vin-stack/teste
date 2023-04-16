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

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLTK's stop words
stop_words = set(stopwords.words('english'))

# Function to preprocess text data
def preprocess_text(text):
    # Tokenize into sentences
    sentences = nltk.sent_tokenize(text)
    # Tokenize sentences into words
    words = [nltk.word_tokenize(sent) for sent in sentences]
    # Convert words to lowercase
    words = [[word.lower() for word in sent if word.isalpha()] for sent in words]
    # Remove stopwords
    words = [[word for word in sent if word not in stop_words] for sent in words]
    return words

# Function to compute sentence similarity using cosine similarity
def sentence_similarity(sent1, sent2, vectorizer):
    tfidf_matrix = vectorizer.transform([sent1, sent2])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity_score

# Function to compute sentence similarity matrix
def similarity_matrix(sentences, vectorizer):
    similarity_matrix = []
    for i in range(len(sentences)):
        similarity_row = []
        for j in range(len(sentences)):
            if i != j:
                similarity_score = sentence_similarity(' '.join(sentences[i]), ' '.join(sentences[j]), vectorizer)
                similarity_row.append(similarity_score)
        similarity_matrix.append(similarity_row)
    return similarity_matrix

# Function to calculate semantic similarity using fuzzywuzzy
def semantic_similarity(sent1, sent2):
    from fuzzywuzzy import fuzz
    similarity_score = fuzz.ratio(sent1, sent2) / 100
    return similarity_score

# Function to perform extractive summarization
def extractive_summarization(text, num_sentences=3):
    sentences = preprocess_text(text)
    sentences = [' '.join(sent) for sent in sentences]
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    sentence_similarity_matrix = similarity_matrix(sentences, vectorizer)
    semantic_similarity_scores = [[semantic_similarity(sentences[i], sentences[j]) for j in range(len(sentences))] for i in range(len(sentences))]
    sentence_similarity_scores = [[sentence_similarity_matrix[i][j] * semantic_similarity_scores[i][j] if i!=j else 0 for j in range(len(sentences))] for i in range(len(sentences))]
    sentence_similarity_scores = [sum(row) for row in sentence_similarity_scores]
    sorted_indices = sorted(range(len(sentence_similarity_scores)), key=lambda k: sentence_similarity_scores[k], reverse=True)
    selected_indices = sorted_indices[:num_sentences]
    selected_sentences = [sentences[i] for i in selected_indices]
    summary = ' '.join(selected_sentences)
    return summary

# Example usage
text = "Centurion University of Technology and Management is a multi-sector, private state university from Odisha, India. With its main campus earlier at Parlakhemundi in the Gajapati and another constituent campus located at Jatni, on the fringes of Bhubaneswar,which is now as main campus & it was accorded the status of a university in the year 2010"
summary = extractive_summarization(text)
print("Original Text:")
print(text)
print("\nSummary:")
print(summary)

st.write(text)

st.write(summary)

