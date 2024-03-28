import streamlit as st
import pandas as pd
from main import movie_mean, predict_rating, similarity

# st.title("Tutoriel :red[Streamlit]")
st.header(':blue[Prediction de note]')

movies_number = st.number_input('Nombre de films:', 1)
users_number = st.number_input('Nombre d\'utilisateurs:', 1)
top_number = st.number_input('N:', 2)


user_to_predict = st.number_input('N:', 1)
movie_to_predict = st.number_input('N:', 1)

if st.button(label= "Predire"):
    predict_rating = predict_rating(n=top_number, users_number= users_number)
    st.write(f"L'utilisateur {user_to_predict} pourrait donner au film {movie_to_predict}, un score de: {predicted_rating}")


# data = pd.DataFrame({'Nom': ['Alice', 'Bob', 'Charlie'], 'Âge': [25, 30, 22]})
# st.write(data)