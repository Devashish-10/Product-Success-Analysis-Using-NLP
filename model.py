import pandas as pd
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')

# Load the CSV file
data = pd.read_csv('TeePublic_review.csv', encoding='ISO-8859-1')

# Feature engineering
# Assuming 'review' column contains the text data for sentiment analysis
reviews = data['review'].astype(str)
labels = data['review-label']

# Text preprocessing
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [ps.stem(word) for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

reviews = reviews.apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Save the vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

# Train a sentiment analysis model (Naive Bayes)
sentiment_model = MultinomialNB()
sentiment_model.fit(X_train_vectorized, y_train)

# Save the sentiment analysis model
joblib.dump(sentiment_model, 'sentiment_model.pkl')

# Evaluate the sentiment analysis model
accuracy = accuracy_score(y_test, sentiment_model.predict(X_test_vectorized))
print(f'Sentiment Analysis Model Accuracy: {accuracy * 100:.2f}%')

# Extract relevant features for Linear Regression
features = data[['latitude', 'longitude']]

# Combine sentiment analysis result with features
features['sentiment'] = sentiment_model.predict(vectorizer.transform(reviews))

# Split the dataset for Linear Regression
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train Linear Regression model
regression_model = LinearRegression()
regression_model.fit(X_train_lr, y_train_lr)

# Save the revenue estimation model
joblib.dump(regression_model, 'revenueEstimate_model.pkl')

# Evaluate the Linear Regression model
regression_accuracy = regression_model.score(X_test_lr, y_test_lr)
print(f'Linear Regression Model Accuracy: {regression_accuracy * 100:.2f}%')

# Print coefficients and intercept for insights
print('Coefficients:', regression_model.coef_)
print('Intercept:', regression_model.intercept_)

# Analyze feature importance
feature_importance = pd.Series(regression_model.coef_, index=features.columns)
sorted_feature_importance = feature_importance.abs().sort_values(ascending=False)
print('\nFeature Importance:')
print(sorted_feature_importance)

# Calculate correlation between sentiment and success
sentiment_correlation = features['sentiment'].corr(labels)
print('\nCorrelation between Sentiment and Success:', sentiment_correlation)

# Visualize the correlation between sentiment and success
plt.figure(figsize=(8, 6))
sns.scatterplot(x=features['sentiment'], y=labels)
plt.title('Sentiment vs Success Score')
plt.xlabel('Sentiment Score')
plt.ylabel('Success Score')
plt.show()