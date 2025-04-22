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

'''
print("Первые пять строк набора данных:")
print(ratings.head())
print(movies.head())
'''
#Создание матрицы  предпочтений "элемент-пользователь"
item_user_matrix = ratings.pivot(index="movieId",
                                 columns="userId",
                                 values= "rating").fillna(0)

item_similarity = cosine_similarity(item_user_matrix)

#Рекомендации на основе сходства элементов

def reccomend_movie_item_based(movie_title, movies, item_user_matrix, item_similarity, n_recommendation = 10):
    #получаем индекс фильма по названию
    movie_idx = movies[movies["title"] == movie_title].index[0]
    
    sim_scores = item_similarity[movie_idx]
    #получаем индексы похожих фильмов
    similar_movies = np.argsort(sim_scores)[::-1][1:n_recommendation+1]
    
    #возврат самых похожих фильмов
    return movies['title'].iloc[similar_movies].tolist()

def get_movie_input():
    '''получает входные данные от пользователя и обрабатывает ошибки'''
    while True:
        try:
            input_movie = input("\n\nВведите название фильма ('Выход', 'Exit', чтобы выйти):")
            if input_movie.lower() == 'выход' or input_movie.lower() == 'exit':
                return None #сигнал к выходу
            return input_movie
        except Exception as e:
            print(f"Ошибка {e}")

while True: 
    input_movie = get_movie_input()
    
    if input_movie is None:
        break
    
    try: 
        reccomend_movie_item_based = reccomend_movie_item_based(input_movie, movies, item_user_matrix, item_similarity)
        
        if reccomend_movie_item_based:
            print(f"10 рекомендуемых фильмов на основе - {input_movie}")
            print("\n".join(reccomend_movie_item_based))
        else:
            print(f"Не найдено рекомендаций для {input_movie}. Возможно название фильма не найдено")
        
    except Exception as e:
        print(f"Во время рекомендации произошла ошибка {e}")
            