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
top_number = st.number_input('N:', 2)
scores_grid = pd.read_csv("movies.csv")

if st.button(label= "Generer le tableau"):
    
    st.write(scores_grid)


user_to_predict = st.number_input('Utilisateur dont on veut prédire la note:', 1)
movie_to_predict = st.number_input('Film dont on veut prédire la note:', 1)
if st.button(label= "Predire"):
    print(scores_grid.count(axis=0).size)
    predicted_rating = predict_rating(
        n= top_number,
        users_number= scores_grid.count(axis=1).size,
        selected_movie= movie_to_predict - 1,
        selected_user= user_to_predict - 1
    )
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
        