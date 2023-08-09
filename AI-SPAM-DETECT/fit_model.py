import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from keras.layers import Dense, Dropout

# Loading data
with open('spam.txt', 'r', encoding='utf-8') as file:
    train_data = file.readlines()

train_labels = np.ones(len(train_data))

vectorizer = TfidfVectorizer()

# Преобразование текстовых данных в TF-IDF векторы
train_vectors = vectorizer.fit_transform(train_data)
train_vectors_dense = train_vectors.toarray()

# Creating model
model = tf.keras.Sequential([
    Dense(64, activation='relu', input_dim=train_vectors_dense.shape[1]),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_vectors_dense, train_labels, epochs=100, batch_size=32)

# Test data
with open('test_spam.txt', 'r', encoding='utf-8') as file:
    test_data = file.readlines()

test_vectors = vectorizer.transform(test_data)
test_vectors_dense = test_vectors.toarray()

test_predictions = model.predict(test_vectors_dense)
test_predictions_classes = [1 if pred > 0.5 else 0 for pred in test_predictions]

print("Predictions for test data:", test_predictions_classes)

model.save("SpamDetector.tf")