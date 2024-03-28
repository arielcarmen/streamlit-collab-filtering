import streamlit as st 
import numpy as np

st.title("my app")

ratings = np.array([
    [1, 0, 3, 0, 0, 5, 0, 0, 5, 0, 4, 0],
    [0, 0, 5, 4, 0, 0, 4, 0, 0, 2, 1, 3],
    [2, 4, 0, 1, 2, 0, 3, 0, 4, 3, 5, 0],
    [0, 2, 4, 0, 5, 0, 0, 4, 0, 0, 2, 0],
    [0, 0, 4, 3, 4, 2, 4, 0, 0, 0, 2, 5],
    [1, 0, 3, 0, 3, 0, 0, 2, 0, 0, 4, 0],
])

N = 2

def movie_mean(movie):
    return np.sum(ratings[movie])/np.count_nonzero(ratings[movie])


def similarity(movie1, movie2):
    
    ratings_movie1 = ratings[movie1]
    ratings_movie2 = ratings[movie2]

    mean_movie_1 = movie_mean(movie1)
    mean_movie_2 = movie_mean(movie2)

    users_means = []

    for i in range(12):
        if (ratings_movie1[i] != 0 and ratings_movie2[i] != 0):
            users_means.append((ratings_movie1[i]-ratings_movie1) * (ratings_movie2[i]-ratings_movie2))
            

    
    similarity = 1 
    print(similarity)
    return similarity


def top_n_similarity(movie, n):
    # similarites 
    similarities = [(i, similarity_mean(movie, i)) for i in range(len(ratings)) if i != movie]
    # Trier les similarités par ordre décroissant
    similarities.sort(key=lambda x: x[1], reverse=True)
    # Sélectionner les top N similarités
    top_n = similarities[:n]
    return top_n

# top_n_similarities = top_n_similarity(4, N)


# film5_ratings = ratings[1]


# user1_ratings = [film5_ratings[movie] for movie, _ in top_n_similarities]


# predicted_rating = np.mean(user1_ratings)
# print(f"La note prédite que l'utilisateur 1 aurait donnée au film 5 est : {predicted_rating:.2f}")

similarity_mean(0,1)