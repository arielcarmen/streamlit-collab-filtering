import numpy as np
import pandas as pd
import math
import streamlit as st

def movie_mean(ratings, movie):
    return np.sum(ratings[movie])/np.count_nonzero(ratings[movie])


def similarity(ratings, movie1, movie2, users_number):
    ratings_movie1 = ratings[movie1]
    ratings_movie2 = ratings[movie2]

    mean_movie_1 = movie_mean(ratings, movie1)
    mean_movie_2 = movie_mean(ratings, movie2)

    users_means = []
    indexes = []
    temp1 = temp2 = 0

    for i in range(users_number):
        if ratings_movie1[i] != 0 and ratings_movie2[i] != 0:
            indexes.append(i)
            users_means.append((ratings_movie1[i]-mean_movie_1) * (ratings_movie2[i]-mean_movie_2))
    
    for i in indexes:
        temp1 += (ratings_movie1[i] - mean_movie_1) ** 2
        temp2 += (ratings_movie2[i] - mean_movie_2) ** 2
        
    std_movie1 = np.sqrt(temp1)
    std_movie2 = np.sqrt(temp2)
        
    if ((std_movie1*std_movie2) not in [0, np.NaN]) and ((std_movie1*std_movie2) not in [0, np.NaN]):
        similarity_value = sum(users_means)/(std_movie1*std_movie2)
    else:
        similarity_value = 0

    return round(similarity_value, 2)


def top_n_similarity(ratings, movie, n, users_number):
    similarities = []
    
    for i in range(len(ratings)):
        if i != movie:
            sim = similarity(ratings, movie, i, users_number)
            similarities.append((i, sim))
   
    similarities.sort(key=lambda x: x[1], reverse=True)

    top_n = similarities[:n]
    return top_n

def predict_rating(ratings, n, users_number, selected_movie, selected_user):
    top_n_similarities = top_n_similarity(ratings, selected_movie, n, users_number)
    rates_by_means = 0
    sims= []
    for item in top_n_similarities:
        user_rating = ratings[item[0]][selected_user]
        if user_rating != 0:
            sim = similarity(ratings, selected_movie, item[0], users_number)
            rates_by_means += sim * user_rating
            sims.append(sim)


    if sum(sims) not in [0, np.NaN]:
        return rates_by_means / sum(sims)
    else:
        return 0


def generate_table(scores_grid, movies_number, users_number):
    movies_labels = [f"Film {i}" for i in range (1, movies_number + 1)]
    users_labels = [f"Utilisateur {i}" for i in range (1, users_number + 1)]
    # scores_grid = scores_grid.replace(0, np.nan)
    scores_grid = scores_grid.set_index(pd.Index(movies_labels))
    scores_grid = scores_grid.rename(columns = {i: users_labels[i] for i in range(users_number)})
    
    return scores_grid

def highlight_changes(df_before, df_after, color='lightgreen'):
    
    df_highlighted = df_after.copy()

    # Identifier les cellules qui passent de NaN à une valeur numérique dans le second DataFrame
    mask = (~df_before.isna()) & (df_after.notna())

    # Mettre en couleur les cellules modifiées dans la copie du DataFrame
    for col in df_before.columns:
        for idx in df_before.index:
            if mask.loc[idx, col]:
                df_highlighted.loc[idx, col] = f'background-color: {color}'

    return df_highlighted
