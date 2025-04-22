import numpy as np
import pandas as pd


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")



# считываем данные с csv файлов и создаем датафрейм

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
print("Первые пять строк набора данных:")

print(ratings.head())
print(len(ratings))
print(movies.head())