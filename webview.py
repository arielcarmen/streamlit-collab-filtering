import streamlit as st
import pandas as pd
from functions import movie_mean, predict_rating, similarity

st.header(':blue[Prediction de note]')

column1, column2 = st.columns(2)

with column1:
    movies_number = st.number_input('Nombre de films:', 1)
    users_number = st.number_input('Nombre d\'utilisateurs:', 1)
    top_number = st.number_input('N:', 2)

with column2:
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

