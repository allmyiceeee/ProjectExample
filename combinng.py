import numpy as np
import pandas as pd


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")



rating = pd.read_csv('rating.csv')
movies = pd.read_csv('movies.csv')

#выравниваем данные
common_movie_ids = set(rating['movieId']).intersection(set(movies['movieId']))
aligned_ratings_df = rating[rating['movieId'].isin(common_movie_ids)]
aligned_movies_df = movies[movies['movieId'].isin(common_movie_ids)]

#создаем матрицу предпочтений
item_user_matrix_v2 = aligned_ratings_df.pivot(index='moveId',
                                               columns='iserId',
                                               values='rating').fillna(0)

item_similarity_ratings = cosine_similarity(item_user_matrix_v2)

tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
tfidf_matrix = tfidf.fit_transform(aligned_movies_df['genres'])

item_similarity_genres = cosine_similarity(tfidf_matrix)

combined_similarity = (item_similarity_ratings + item_similarity_genres) / 2

def get_movie_input():
	"""получает входные данные от пользователя и обрабатывает ошибки"""
	while True:
		try:
			input_movie = input(
				"\n\nВведите название фильма ('Выход', 'Exit', чтобы выйти):"
			)
			if input_movie.lower() == "выход" or input_movie.lower() == "exit":
				return None  # сигнал к выходу
			return input_movie
		except Exception as e:
			print(f"Ошибка {e}")


#функция рекомендованных фильмов 
def recommend_movies_combined(movie_title, movies, combined_similarity, n_recomendations = 10):
    movie_idx = movies[movies['title'] == movie_title].index[0]
    
    sim_scores = combined_similarity[movie_idx]
    
    similar_movies = np.argsort(sim_scores)[::-1][1:n_recomendations+1]
    
    return movies['title'].iloc[similar_movies].tolist()


while True:
    input_movie = get_movie_input()
    
    if input_movie is None:
        break
    
    try:
        recommend_movies_combined = recommend_movies_combined(input_movie, aligned_movies_df, combined_similarity)
        
        if recommend_movies_combined:
            print(f'Рекомендуемые 10 фильмов на основе:{input_movie}')
            print('\n'.join(recommend_movies_combined))
        else:
            print(f'Не найдено рекомендаций для {input_movie}')
    except Exception as e:
        print(f'Ошибка {e}')
    