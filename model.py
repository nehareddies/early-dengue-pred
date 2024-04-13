import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('Symptom2Disease.csv', encoding='latin-1') 

labels = df['label']  
symptoms = df['text']

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

preprocessed_symptoms = symptoms.apply(preprocess_text)
tfidf_vectorizer = TfidfVectorizer(max_features=1500) 
tfidf_features = tfidf_vectorizer.fit_transform(preprocessed_symptoms).toarray()
X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels, test_size=0.2, random_state=42)
knn_classifier = KNeighborsClassifier(n_neighbors=5)  
knn_classifier.fit(X_train, y_train)
predictions = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, predictions))

print("Number of neighbors:", knn_classifier.n_neighbors)
print("Classes:", knn_classifier.classes_)
print("Training data shape:", knn_classifier._fit_X.shape)
print("Training target shape:", knn_classifier._y.shape)


