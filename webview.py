import streamlit as st
import numpy as np
import pandas as pd
from functions import predict_rating, array_to_dataframe

st.header(':blue[Prediction de note]')

class_ratings = np.array([
    [1, 0, 3, 0, 0, 5, 0, 0, 5, 0, 4, 0],
    [0, 0, 5, 4, 0, 0, 4, 0, 0, 2, 1, 3],
    [2, 4, 0, 1, 2, 0, 3, 0, 4, 3, 5, 0],
    [0, 2, 4, 0, 5, 0, 0, 4, 0, 0, 2, 0],
    [0, 0, 4, 3, 4, 2, 4, 0, 0, 0, 2, 5],
    [1, 0, 3, 0, 3, 0, 0, 2, 0, 0, 4, 0],
])

@st.cache_resource
def generate_table():
    return 
column1, column2 = st.columns(2)

# movies_number = st.number_input('Nombre de films:', 1)
# users_number = st.number_input('Nombre d\'utilisateurs:', 1)
scores_grid = pd.read_csv("movies.csv")
scores_grid = scores_grid.apply(pd.to_numeric, axis= 1)
table = None

table = st.write(scores_grid)
top_number = st.number_input('N:', 2)
user_to_predict = st.number_input('Utilisateur dont on veut prédire la note:', 1)
movie_to_predict = st.number_input('Film dont on veut prédire la note:', 1)

if st.button(label= "Predire"):

    if  movie_to_predict < 1 or movie_to_predict > scores_grid.count(axis=0).size:
        st.error(f"Le film {movie_to_predict} n'existe pas !")
    elif user_to_predict < 1 or user_to_predict > scores_grid.count(axis=1).size :
        st.error(f"L'utilisateur {user_to_predict} n'existe pas !")
    elif top_number > scores_grid.count(axis=1).size or top_number < scores_grid.count(axis=1).size:
        st.error(f"Le top {top_number} n'est pas en accord avec le nombre de films !")
    else:
        existing_score = scores_grid.iloc[movie_to_predict -1, user_to_predict -1]
        if existing_score:
            st.write(f"L'utilisateur {user_to_predict} a déja accordé la note de {existing_score} au film {movie_to_predict}")
        else:
            scores_array = scores_grid.values
            scores_array = np.nan_to_num(scores_array)
            scores_array = scores_array.astype(int)
            predicted_rating = predict_rating(
                ratings= scores_array,
                n= top_number,
                users_number= scores_grid.count(axis=1).size,
                selected_movie= movie_to_predict - 1,
                selected_user= user_to_predict - 1
            )

            if predicted_rating == 0 or 0 < predicted_rating > 5 :
                st.write("Impossible de predire cette note")
            else: 
                st.write(f"L'utilisateur {user_to_predict} pourrait donner au film {movie_to_predict}, un score de: {round(predicted_rating, 2)}")

def generate_csv():
    movies_labels = [f"Film {i}" for i in range (1, movies_number + 1)]
    users_labels = [f"Utilisateur {i}" for i in range (1, users_number + 1)]
    scores_grid = pd.DataFrame(np.random.choice([0, 1, 2, 3, 4, 5], size=(movies_number, users_number), p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))
    scores_grid = scores_grid.replace(0, '')
    scores_grid = scores_grid.set_index(pd.Index(movies_labels))
    scores_grid = scores_grid.rename(columns = {i: users_labels[i] for i in range(users_number)})
    
    st.table(scores_grid)
    scores_grid.to_csv('movies_.csv')
        