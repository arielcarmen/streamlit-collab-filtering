import streamlit as st
import numpy as np
import pandas as pd

st.header(':blue[Prediction de notes à partir d\'un csv]')

# Création des différents containers
dataframe_container_ = st.empty()

csv_loader = st.container()

scores_field_ = st.container()

n_choice_ = st.container()

user_choice_ = st.container()

predicted_result_ = st.container()

if 'dataframe_' not in st.session_state:
    st.session_state['dataframe_'] = pd.DataFrame()

users_number_ = 0
movies_number_ = 0

if 'show_csv_field_' not in st.session_state:
    st.session_state['show_csv_field_'] = True

if 'choose_n_' not in st.session_state:
    st.session_state['choose_n_'] = False

if 'show_user_choice_' not in st.session_state:
    st.session_state['show_user_choice_'] = False

if 'show_result_' not in st.session_state:
    st.session_state['show_result_'] = False

if 'n_' not in st.session_state:
    st.session_state['n_'] = 2

scores_grid_ = st.session_state['dataframe_']

# Calculer les notes maquantes
def validate_scores_datas():
    st.session_state['n_'] = N
    st.session_state['choose_n_'] = not st.session_state['choose_n_']
    st.session_state['show_user_choice_'] = not st.session_state['show_user_choice_']

    grid = st.session_state['dataframe_']

    for i in range(st.session_state['movies_number_']):
        for j in range(st.session_state['users_number_']):
            if np.isnan(grid.iloc[i,j]):
                predicted_value = predict_user_rating(movie_index=i, user_index= j, top_n= st.session_state['n_'], df= grid)
                scores_grid_.iloc[i,j] = round(predicted_value)
                st.session_state['dataframe_'] = scores_grid_
    dataframe_container_.dataframe(scores_grid_)

# Definir la valeur de N
def define_n():
    st.session_state['show_csv_field_'] = not st.session_state['show_csv_field_']
    st.session_state['choose_n_'] = not st.session_state['choose_n_']
    movies_number_ = len(scores_grid_)
    users_number_ = len(scores_grid_.columns)

    st.session_state.users_number_ = users_number_
    st.session_state.movies_number_ = movies_number_

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

# Rechercher le score dans le tableau
def check_movie_score(user, movie):
    actual_dataframe = st.session_state['dataframe_']
    original_dataframe = st.session_state['original_dataframe_']
    st.session_state['show_result_'] = True
    predicted_rating = actual_dataframe.iloc[movie, user]

    if st.session_state['show_result_'] == True:
        if actual_dataframe.iloc[movie, user] != original_dataframe.iloc[movie, user]:
            predicted_result_.write("Cette valeur a été calculée par le programme")
            if predicted_rating == np.nan:
                predicted_result_.write("Cette valeur n'a pas pu être prédite")
            elif predicted_rating < 3:
                predicted_result_.write(f"Cet utilisateur {user +1} n'aimerait pas le film {movie +1}, avec une note possible de {actual_dataframe.iloc[movie, user]}")
            elif predicted_rating >= 3:
                predicted_result_.write(f"L'utilisateur {user +1} pourrait apprécier le film {movie +1}, avec une note possible de {actual_dataframe.iloc[movie, user]}")
        else:
            predicted_result_.write(f"Cette valeur a n'a pas été calculée, la note existante est de {actual_dataframe.iloc[movie, user]}")

# Déroulement du programme
if st.session_state['show_csv_field_']:
    uploaded_file = csv_loader.file_uploader("Charger un csv:", type=["csv"])
    if uploaded_file is not None:
        scores_grid_ = pd.read_csv(uploaded_file)
        st.session_state['dataframe_'] = scores_grid_
        st.session_state['original_dataframe_'] = scores_grid_.copy()
        csv_loader.button("Valider", on_click= define_n)
    

if st.session_state['choose_n_'] == True:
    dataframe_container_.dataframe(scores_grid_)
    N = n_choice_.number_input('Valeur du top n:', 2)
    st.session_state['n_'] = N
    n_choice_.button("Valider N", on_click= validate_scores_datas)

if st.session_state['show_user_choice_'] == True:
    dataframe_container_.dataframe(st.session_state['dataframe_'])
    grid = st.session_state['dataframe_']
    user_choice_.write("Choisir une donnée à predire")
    user_to_predict = user_choice_.number_input('Utilisateur:', 1, max_value= len(grid.columns))
    movie_to_predict = user_choice_.number_input('Films:', 1, max_value= len(grid))

    user_choice_.button("Evaluer", on_click= check_movie_score(user= user_to_predict-1, movie= movie_to_predict-1))
