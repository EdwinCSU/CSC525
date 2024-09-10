import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from textblob import TextBlob

# Use a smaller spaCy model
nlp = spacy.load("en_core_web_sm")

# Download NLTK data
nltk.download('wordnet')
nltk.download('punkt')

# Use Streamlit's caching to load models only once
@st.cache(allow_output_mutation=True)
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

    return tokenizer, model
# Load spaCy model for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Download NLTK data
nltk.download('wordnet')
nltk.download('punkt')

def preprocess_text(text):
  # Tokenize, lemmatize, and remove stop words
  doc = nlp(text)
  tokens = [token.lemma_ for token in doc if not token.is_stop]
  return ' '.join(tokens)

def get_response(user_input, chat_history):
  # Preprocess input
  user_input_processed = preprocess_text(user_input)

  # Update chat history
  chat_history.append(user_input_processed)

  # Use seq2seq model for generation
  inputs = tokenizer(chat_history[-2:], return_tensors="pt", padding=True, truncation=True)
  outputs = model.generate(**inputs)
  generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

  # Use TextBlob for sentiment analysis and adjust response accordingly
  sentiment = TextBlob(generated_text).sentiment.polarity
  if sentiment > 0.2:
    # Positive sentiment, adjust response accordingly
    generated_text = "That sounds great! " + generated_text
  elif sentiment < -0.2:
    # Negative sentiment, adjust response accordingly
    generated_text = "I understand. " + generated_text

  return generated_text

def main():
  st.title("Chatbot")

  chat_history = []

  while True:
    user_input = st.text_input("You:")
    if user_input:
      response = get_response(user_input, chat_history)
      chat_history.append(response)
      st.text_area("Chatbot:", value=response, height=200)

if __name__ == "__main__":
    tokenizer, model = load_models()
    main()
