import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Загрузка данных
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Подготовка данных (общая часть)
common_movie_ids = set(ratings['movieId']).intersection(set(movies['movieId']))
aligned_ratings_df = ratings[ratings['movieId'].isin(common_movie_ids)]
aligned_movies_df = movies[movies['movieId'].isin(common_movie_ids)]

item_user_matrix = aligned_ratings_df.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# ---------- Item-Based Recommender ----------
def item_based_recommender(movie_title, movies_df, ratings_matrix, n=10):
    try:
        item_similarity = cosine_similarity(ratings_matrix)
        movie_idx = movies_df[movies_df['title'] == movie_title].index[0]
        sim_scores = item_similarity[movie_idx]
        similar_movies = np.argsort(sim_scores)[::-1][1:n+1]
        return movies_df['title'].iloc[similar_movies].tolist()
    except:
        return []

# ---------- Hybrid Recommender ----------
def hybrid_recommender(movie_title, movies_df, ratings_matrix, n=10):
    try:
        item_similarity_ratings = cosine_similarity(ratings_matrix)
        tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
        tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
        item_similarity_genres = cosine_similarity(tfidf_matrix)
        combined_similarity = (item_similarity_ratings + item_similarity_genres) / 2

        movie_idx = movies_df[movies_df['title'] == movie_title].index[0]
        sim_scores = combined_similarity[movie_idx]
        similar_movies = np.argsort(sim_scores)[::-1][1:n+1]
        return movies_df['title'].iloc[similar_movies].tolist()
    except:
        return []

# ---------- Тестовая функция ----------
def run_tests(test_title="Toy Story (1995)"):
    print(f"\nСравнение рекомендаций для фильма: {test_title}\n")

    start = time.time()
    item_result = item_based_recommender(test_title, aligned_movies_df, item_user_matrix)
    time_item = time.time() - start

    start = time.time()
    hybrid_result = hybrid_recommender(test_title, aligned_movies_df, item_user_matrix)
    time_hybrid = time.time() - start

    print("Item-based рекомендации:")
    print("\n".join(item_result))
    print(f"Время: {time_item:.4f} сек")

    print("\nHybrid рекомендации:")
    print("\n".join(hybrid_result))
    print(f"Время: {time_hybrid:.4f} сек")

    # Coverage test (сколько фильмов каждая модель может рекомендовать)
    sample_movies = aligned_movies_df['title'].sample(50, random_state=42)
    item_coverage = sum([1 for title in sample_movies if item_based_recommender(title, aligned_movies_df, item_user_matrix)])
    hybrid_coverage = sum([1 for title in sample_movies if hybrid_recommender(title, aligned_movies_df, item_user_matrix)])

    print("\nCoverage (на 50 случайных фильмах):")
    print(f"Item-based: {item_coverage}/50 фильмов")
    print(f"Hybrid:     {hybrid_coverage}/50 фильмов")


if __name__ == "__main__":
    run_tests("Othello (1995)")
