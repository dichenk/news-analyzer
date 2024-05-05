import os
from joblib import load
import numpy as np
from razdel import sentenize, tokenize
from gensim.models import FastText
from classes import fasttext_lowercase_train_config, TextPreprocessor
import fnmatch
from collections import Counter


# Функция загрузки моделей по шаблону
def load_models(model_name, test_size):
    pattern = f"{model_name}_*_{test_size}.joblib"
    models = []
    for file in os.listdir('models'):
        if fnmatch.fnmatch(file, pattern):
            model_path = os.path.join('models', file)
            model = load(model_path)
            models.append(model)
    return models


# Функция классификации текста
def classify_text(text, model, clf, preprocessor):
    processed_text = preprocessor.text_cleaning(text)
    sentences = sentenize(processed_text)
    text_vectors = []
    for sentence in sentences:
        words = [token.text for token in tokenize(sentence.text)]
        word_weights = Counter(words)  # Подсчет количества каждого слова
        word_vectors = [model.wv[word] * word_weights[word] for word in words if word in model.wv]
        if word_vectors:
            sentence_vector = np.sum(word_vectors, axis=0) / len(words)  # Используем среднее или можно использовать другой метод агрегации
        else:
            sentence_vector = np.zeros(model.wv.vector_size)
        text_vectors.append(sentence_vector)

    if text_vectors:
        overall_vector = np.mean(text_vectors, axis=0).reshape(1, -1)
    else:
        overall_vector = np.zeros((1, model.wv.vector_size))

    prediction = clf.predict_proba(overall_vector)[:, 1]  # Предсказываем вероятность для класса 1
    return prediction[0]


# Загрузка модели FastText и инициализация препроцессора
fasttext_model = FastText.load("models/fasttext_model1.bin")
preprocessor = TextPreprocessor(fasttext_lowercase_train_config)


def analize(texts, model_name="log_reg_model", test_size=0.2, threshold=0.9):

    # Загрузка соответствующих моделей
    classifiers = load_models(model_name, test_size)

    # Прогоняем тексты через каждую загруженную модель и собираем результаты
    results = {f"Model {i}": [] for i in range(len(classifiers))}
    for text in texts:
        for i, clf in enumerate(classifiers):
            probability = classify_text(text, fasttext_model, clf, preprocessor)
            results[f"Model {i}"].append(probability)

    ## Создаем структуру для хранения результатов транспонированно
    transposed_results = {f"Text {i+1}": [] for i in range(len(texts))}

    # Заполняем транспонированные результаты
    for model, probabilities in results.items():
        for i, prob in enumerate(probabilities):
            if prob > threshold:
                transposed_results[f"Text {i+1}"].append((model, prob))

    # Выводим результаты для каждого текста
    for text, model_probs in transposed_results.items():
        if model_probs:  # Проверяем, не пустой ли список
            print(f"{text} exceeded threshold with models:")
            for model, prob in model_probs:
                print(f"  {model}: {prob:.4f}")
        else:
            print(f"{text} did not exceed threshold with any model.")

    return results
