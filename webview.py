import streamlit as st
import pandas as pd
# from main import movie_mean, predict_rating, similarity

# st.title("Tutoriel :red[Streamlit]")
st.header(':blue[Prediction de note]')

movie_number = st.number_input('Nombre de films:', 1)
user_number = st.number_input('Nombre d\'utilisateurs:', 1)
top_number = st.number_input('N:', 2)


user_to_predict = st.number_input('N:', 2)
st.button(label= "Predire")


# data = pd.DataFrame({'Nom': ['Alice', 'Bob', 'Charlie'], 'Âge': [25, 30, 22]})
# st.write(data)