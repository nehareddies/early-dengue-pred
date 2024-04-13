from flask import Flask, render_template, request
import pickle
import pandas as pd
from model import *
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
model = pickle.load(open('dengue.pkl', 'rb'))

stop_words = set(stopwords.words('english'))
tfidf_vectorizer = TfidfVectorizer(max_features=1500)
X = []
y = []

# Assuming symptoms is a list of strings containing symptoms
for symptom in symptoms:
    words = word_tokenize(symptom.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    preprocessed_symptom = ' '.join(words)
    X.append(preprocessed_symptom)

# Assuming labels is a list of labels corresponding to symptoms
y = labels

# Fit the vectorizer on your data
X = tfidf_vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit your model
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    symptoms = str(request.form['text'])
    words = word_tokenize(symptoms.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    preprocessed_symptoms = ' '.join(words)
    
    # Transform the input using the same vectorizer
    symptom_tfidf = tfidf_vectorizer.transform([preprocessed_symptoms])
    
    # Make predictions
    predictions = knn_classifier.predict(symptom_tfidf)
    
    if predictions[0] == "Dengue":
        result = "You are more likely to be suffering from dengue!"
    else:
        result = "You have less chances that you are suffering from dengue but there are chances that you are suffering from " + predictions[0]
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
