import streamlit as st
import numpy as np
import pandas as pd

st.header(':blue[Prediction de note]')

dataframe_container = st.empty()

dataframe_fields = st.container()

scores_field = st.container()

n_choice = st.container()

user_choice = st.container()

predicted_result = st.container()

if 'dataframe' not in st.session_state:
    st.session_state['dataframe'] = pd.DataFrame()

users_number = 4
movies_number = 4
N = 2

if 'show_dataframe_fields' not in st.session_state:
    st.session_state['show_dataframe_fields'] = True
    
if 'show_scores_fields' not in st.session_state:
    st.session_state['show_scores_fields'] = False

if 'choose_n' not in st.session_state:
    st.session_state['choose_n'] = False

if 'show_user_choice' not in st.session_state:
    st.session_state['show_user_choice'] = False

if 'show_result' not in st.session_state:
    st.session_state['show_result'] = False

if 'n' not in st.session_state:
    st.session_state['n'] = 2

scores_grid = st.session_state['dataframe']

def validate_dataframe_size():
    movies_labels = [f"Film {i}" for i in range (1, movies_number + 1)]
    users_labels = [f"Utilisateur {i}" for i in range (1, users_number + 1)]
    
    dataframe_container.dataframe(scores_grid)
    
    st.session_state.users_number = users_number
    st.session_state.movies_number = movies_number

    st.session_state['show_dataframe_fields'] = not st.session_state['show_dataframe_fields']
    st.session_state['show_scores_fields'] = not st.session_state['show_scores_fields']

def validate_scores_datas():
    st.session_state['n'] = N
    st.session_state['choose_n'] = not st.session_state['choose_n']
    st.session_state['show_user_choice'] = not st.session_state['show_user_choice']

    grid = st.session_state['dataframe']
    grid.replace(0.0, np.nan, inplace=True)

    for i in range(st.session_state['movies_number']):
        for j in range(st.session_state['users_number']):
            if np.isnan(grid.loc[i,j]):
                predicted_value = predict_user_rating(movie_index=i, user_index= j, top_n= st.session_state['n'], df= grid)
                scores_grid.loc[i,j] = round(predicted_value)
                st.session_state['dataframe'] = scores_grid
    dataframe_container.dataframe(scores_grid)

def define_n():
    st.session_state['show_scores_fields'] = not st.session_state['show_scores_fields']
    st.session_state['choose_n'] = not st.session_state['choose_n']

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
    predicted_rating = actual_dataframe.loc[movie, user]

    if st.session_state['show_result'] == True:
        if actual_dataframe.loc[movie, user] != original_dataframe.loc[movie, user]:
            predicted_result.write("Cette valeur a été calculée par le programme")
            if predicted_rating == np.nan:
                predicted_result.write("Cette valeur n'a pas pu être prédite")
            elif predicted_rating < 3:
                predicted_result.write(f"Cet utilisateur {user +1} n'aimerait pas le film {movie +1}")
            elif predicted_rating >= 3:
                predicted_result.write(f"L'utilisateur {user +1} pourrait apprécier pas le film {movie +1}")
        else:
            predicted_result.write("Cette valeur a n'a pas été calculée")


# CODE......
if st.session_state['show_dataframe_fields']:
    column1, column2 = dataframe_fields.columns(2)
    with column1:
        movies_number = dataframe_fields.number_input('Nombre de films:', 4)
    with column2:
        users_number = dataframe_fields.number_input('Nombre d\'utilisateurs:', 4)

    dataframe_fields.button("Valider", on_click= validate_dataframe_size)

if st.session_state['show_scores_fields'] == True:
    scores_field.write('Entrez les notes: (si aucune laisser a 0)')
    for i in range(st.session_state['movies_number']):
        for j in range(st.session_state['users_number']):
            scores_grid.loc[i,j] = scores_field.number_input(f'Note du Film {i+1} par l\'utilisateur {j+1}: ', 0, max_value=5)
            st.session_state['dataframe'] = scores_grid
            st.session_state['original_dataframe'] = scores_grid.copy()
            dataframe_container.dataframe(scores_grid)

    scores_field.button("Valider les notes", on_click= define_n)
        
if st.session_state['choose_n'] == True:
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


###### Pour utilisation avec csv

# initial_dataframe = pd.read_csv("class_table.csv")

# if 'dataframe' not in st.session_state:
#     main_dataframe = pd.read_csv("class_table.csv")
#     st.session_state.dataframe = main_dataframe

# scores_grid = st.session_state.dataframe
# scores_grid = scores_grid.apply(pd.to_numeric, axis= 1)
# container.dataframe(scores_grid)
# st.session_state.can_predict = True

# if 'can_predict' in st.session_state and st.session_state.can_predict == True:
#     column3, column4, column5 = st.columns(3)
#     with column3:
#         top_number = st.number_input('N:', 2)
#     with column4:
#         user_to_predict = st.number_input('Utilisateur:', 1)
#     with column5:
#         movie_to_predict = st.number_input('Film à noter:', 1)

#     if st.button(label= "Predire"):

#         if  movie_to_predict < 1 or movie_to_predict > scores_grid.count(axis=1).size:
#             st.error(f"Le film {movie_to_predict} n'existe pas !")
#         elif user_to_predict < 1 or user_to_predict > scores_grid.count(axis=0).size :
#             st.error(f"L'utilisateur {user_to_predict} n'existe pas !")
#         elif top_number > scores_grid.count(axis=1).size or top_number < 1:
#             st.error(f"Le top {top_number} n'est pas en accord avec le nombre de films !")
#         else:
#             existing_score = scores_grid.iloc[movie_to_predict -1, user_to_predict -1]
#             if not pd.isna(existing_score):
#                 st.write(f"L'utilisateur {user_to_predict} a déja accordé la note de {existing_score} au film {movie_to_predict}")
#             else:
#                 scores_array = scores_grid.values
#                 scores_array = np.nan_to_num(scores_array)
#                 scores_array = scores_array.astype(int)
#                 predicted_rating = predict_rating(
#                     ratings= scores_array,
#                     n= top_number,
#                     users_number= scores_grid.count(axis=1).size,
#                     selected_movie= movie_to_predict - 1,
#                     selected_user= user_to_predict - 1
#                 )
#                 predicted_rating = round(predicted_rating)

#                 if predicted_rating == 0 or 0 < predicted_rating > 5 :
#                     st.write("Impossible de predire cette note à partir des données")
#                 else: 
#                     scores_grid.iloc[movie_to_predict -1, user_to_predict -1] = predicted_rating
#                     # st.session_state.dataframe = highlight_changes(df_before= initial_dataframe, df_after=scores_grid)
#                     st.session_state.dataframe = scores_grid
#                     st.write(f"L'utilisateur {user_to_predict} pourrait donner au film {movie_to_predict}, un score de: {round(predicted_rating, 2)}")
#                     container.dataframe(scores_grid)




        