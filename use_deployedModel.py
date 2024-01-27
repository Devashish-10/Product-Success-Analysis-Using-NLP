import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from joblib import load as load_model


nltk.download('stopwords')
nltk.download('punkt')

# Loading the deployed models and vectorizer
sentiment_model = load_model('sentiment_model.pkl')

regression_model = load_model('revenueEstimate_model.pkl')

vectorizer = load_model('vectorizer.pkl')

# Preprocessing the text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = word_tokenize(text)
    words = [ps.stem(word) for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

# Predicting the Sentiment
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    sentiment = sentiment_model.predict(vectorized_text)
    return sentiment[0]

# Predicting Success Score
def predict_success(latitude, longitude, sentiment):
    features = pd.DataFrame({'latitude': [latitude], 'longitude': [longitude], 'sentiment': [sentiment]})
    success_score = regression_model.predict(features)
    return success_score[0]

# Asking User for reiew data
csv_file_path = input("Enter the path of the CSV file: ")

# Reading Data
user_data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

# Analyzing each row in the user-provided dataset
for index, row in user_data.iterrows():
    review_text = row['review']
    latitude = row['latitude']
    longitude = row['longitude']

    # Making predictions
    predicted_sentiment = predict_sentiment(review_text)
    predicted_success_score = predict_success(latitude, longitude, predicted_sentiment)

    # Displaingy predictions for each row
    print(f'\nReview: {review_text}')
    print(f'Predicted Sentiment: {predicted_sentiment}')
    print(f'Predicted Success Score: {predicted_success_score}')
