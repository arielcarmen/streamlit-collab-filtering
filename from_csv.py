import streamlit as st
import numpy as np
import pandas as pd

st.header(':blue[Prediction de notes inserées manuellement]')

dataframe_container = st.empty()

csv_loader = st.container()

scores_field = st.container()

n_choice = st.container()

user_choice = st.container()

predicted_result = st.container()

if 'dataframe' not in st.session_state:
    st.session_state['dataframe'] = pd.DataFrame()

users_number = 0
movies_number = 0

if 'show_csv_field' not in st.session_state:
    st.session_state['show_csv_field'] = True

if 'choose_n' not in st.session_state:
    st.session_state['choose_n'] = False

if 'show_user_choice' not in st.session_state:
    st.session_state['show_user_choice'] = False

if 'show_result' not in st.session_state:
    st.session_state['show_result'] = False

if 'n' not in st.session_state:
    st.session_state['n'] = 2

scores_grid = st.session_state['dataframe']

# Calculer les notes maquantes
def validate_scores_datas():
    st.session_state['n'] = N
    st.session_state['choose_n'] = not st.session_state['choose_n']
    st.session_state['show_user_choice'] = not st.session_state['show_user_choice']

    grid = st.session_state['dataframe']
    print(grid)

    for i in range(st.session_state['movies_number']):
        for j in range(st.session_state['users_number']):
            if np.isnan(grid.iloc[i,j]):
                predicted_value = predict_user_rating(movie_index=i, user_index= j, top_n= st.session_state['n'], df= grid)
                scores_grid.iloc[i,j] = round(predicted_value)
                st.session_state['dataframe'] = scores_grid
    dataframe_container.dataframe(scores_grid)

# Definir la valeur de N
def define_n():
    st.session_state['show_csv_field'] = not st.session_state['show_csv_field']
    st.session_state['choose_n'] = not st.session_state['choose_n']
    movies_number = len(scores_grid)
    users_number = len(scores_grid.columns)

    st.session_state.users_number = users_number
    st.session_state.movies_number = movies_number

def predict_user_rating(df, movie_index, user_index, top_n):
    # Remplacer les valeurs NaN par des zéros pour le calcul de la similarité
    
    ratings_matrix = df.fillna(0).values
    
    # Calculer la matrice de similarité cosinus entre les films
    similarity_matrix = np.dot(ratings_matrix, ratings_matrix.T)
    norms = np.linalg.norm(ratings_matrix, axis=1)
    similarity_matrix = similarity_matrix / norms[:, None]
    similarity_matrix = similarity_matrix / norms[None, :]
    
    # Extraire les notes de l'utilisateur et vérifier si la note est manquante
    user_ratings = df.iloc[:, user_index].values
    if np.isnan(user_ratings[movie_index]):
        # Trouver les films les plus similaires qui ont été notés par l'utilisateur
        similar_films = np.argsort(-similarity_matrix[movie_index, :])
        similar_films = similar_films[~np.isnan(user_ratings[similar_films])]
        if len(similar_films) > top_n:
            similar_films = similar_films[:top_n]
        
        # Calculer la note prédite
        numerator = similarity_matrix[movie_index, similar_films].dot(user_ratings[similar_films])
        denominator = similarity_matrix[movie_index, similar_films].sum()
        predicted_rating = numerator / denominator if denominator != 0 else 0
    else:
        predicted_rating = user_ratings[movie_index]
    
    return predicted_rating

def check_movie_score(user, movie):
    actual_dataframe = st.session_state['dataframe']
    original_dataframe = st.session_state['original_dataframe']
    st.session_state['show_result'] = True
    predicted_rating = actual_dataframe.iloc[movie, user]

    if st.session_state['show_result'] == True:
        if actual_dataframe.iloc[movie, user] != original_dataframe.iloc[movie, user]:
            predicted_result.write("Cette valeur a été calculée par le programme")
            if predicted_rating == np.nan:
                predicted_result.write("Cette valeur n'a pas pu être prédite")
            elif predicted_rating < 3:
                predicted_result.write(f"Cet utilisateur {user +1} n'aimerait pas le film {movie +1}, avec une note possible de {actual_dataframe.iloc[movie, user]}")
            elif predicted_rating >= 3:
                predicted_result.write(f"L'utilisateur {user +1} pourrait apprécier pas le film {movie +1}, avec une note possible de {actual_dataframe.iloc[movie, user]}")
        else:
            predicted_result.write(f"Cette valeur a n'a pas été calculée, la note existante est de {actual_dataframe.iloc[movie, user]}")

# CODE......
if st.session_state['show_csv_field']:
    uploaded_file = csv_loader.file_uploader("Charger un csv:", type=["csv"])
    if uploaded_file is not None:
        scores_grid = pd.read_csv(uploaded_file)
        st.session_state['dataframe'] = scores_grid
        st.session_state['original_dataframe'] = scores_grid.copy()

    csv_loader.button("Valider", on_click= define_n)

if st.session_state['choose_n'] == True:
    dataframe_container.dataframe(scores_grid)
    N = n_choice.number_input('Valeur du top n:', 2)
    st.session_state['n'] = N
    n_choice.button("Valider N", on_click= validate_scores_datas)

if st.session_state['show_user_choice'] == True:
    dataframe_container.dataframe(st.session_state['dataframe'])
    grid = st.session_state['dataframe']
    user_choice.write("Choisir une donnée à predire")
    user_to_predict = user_choice.number_input('Utilisateur:', 1, max_value= len(grid.columns))
    movie_to_predict = user_choice.number_input('Films:', 1, max_value= len(grid))

    user_choice.button("Evaluer", on_click= check_movie_score(user= user_to_predict-1, movie= movie_to_predict-1))
