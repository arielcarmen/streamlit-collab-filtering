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

column1, column2 = st.columns(2)

movies_number = st.number_input('Nombre de films:', 1)
users_number = st.number_input('Nombre d\'utilisateurs:', 1)
top_number = st.number_input('N:', 2)

if st.button(label= "Generer le tableau"):
    scores_grid = arr = np.random.choice([0, 1, 2, 3, 4, 5], size=(movies_number, users_number), p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    st.dataframe(array_to_dataframe(scores_grid))
        

user_to_predict = st.number_input('Utilisateur dont on veut prédire la note:', 1)
movie_to_predict = st.number_input('Film dont on veut prédire la note:', 1)
if st.button(label= "Predire"):
    predicted_rating = predict_rating(
        n= top_number,
        users_number= users_number,
        selected_movie= movie_to_predict - 1,
        selected_user= user_to_predict - 1
    )
    st.write(f"L'utilisateur {user_to_predict} pourrait donner au film {movie_to_predict}, un score de: {round(predicted_rating, 2)}")


