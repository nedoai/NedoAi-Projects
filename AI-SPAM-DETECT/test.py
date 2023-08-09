import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

loaded_model = tf.keras.models.load_model("SpamDetector.tf")
with open("spam.txt", "r", encoding="utf-8") as file:
    data = file.readlines()

custom_message = input("> ")

# Обучение векторизатора на обучающем наборе данных, а потом трансформация на тестовый набор данных
# Training the vectorizer of the subsequent training dataset, transformation of the test dataset
vectorizer = TfidfVectorizer()
vectorizer.fit(data) 
test_vectors = vectorizer.transform([custom_message]).toarray()
expected_num_features = loaded_model.input_shape[1]
#Если кол-во признаков, ожидаемое моделью превышает ожидание
if test_vectors.shape[1] != expected_num_features:
    print(f"Ошибка: Ожидается {expected_num_features} признаков, но получено {test_vectors.shape[1]}. Изменяю...")
    if test_vectors.shape[1] > expected_num_features:
        test_vectors = test_vectors[:, :expected_num_features]
    else:
        print("Тестовые данные имеют меньше признаков, чем ожидалось.")

prediction = loaded_model.predict(test_vectors)

threshold_spam = 0.99
threshold_non_spam = 0.97

decimal_part = int(prediction[0][0] * 100) #Float to int

print(decimal_part, prediction)
if decimal_part >= int(threshold_spam * 100):
    print("Сообщение является спамом! Сообщение - ", custom_message)
elif decimal_part >= 98:
    print("Сообщение похоже на спам, но модель не уверена на 100%! Сообщение - ", custom_message)
elif decimal_part <= 97:
    print("Сообщение не похоже на спам! Сообщение - ", custom_message)
