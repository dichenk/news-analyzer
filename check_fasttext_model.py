from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models import FastText
import numpy as np

model_path = 'models/fasttext_model1.bin'
model = FastText.load(model_path)

# Получаем вектора и слова
words = list(model.wv.key_to_index.keys())
vectors = np.array([model.wv[word] for word in words])

# Применяем PCA
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Визуализация
plt.figure(figsize=(12, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.5)

# Подписываем некоторые слова
words_to_plot = ["производство", "оборудование", "технология", "компания", "продукт", "план"]
for word in words_to_plot:
    index = words.index(word)
    plt.annotate(word, (vectors_2d[index, 0], vectors_2d[index, 1]))

plt.xlabel("PCA dimension 1")
plt.ylabel("PCA dimension 2")
plt.title("PCA визуализация векторов слов")
plt.show()
