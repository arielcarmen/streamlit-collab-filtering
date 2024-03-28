import streamlit as st
import pandas as pd
from main import movie_mean, predict_rating, similarity

st.title("Tutoriel :red[Streamlit]")
st.header(':blue[Introduction aux bases de données]')
st.subheader("👨🏾‍💻 Applications web")
st.text("Ma première application web avec Streamlit ! ")

data = pd.DataFrame({'Nom': ['Alice', 'Bob', 'Charlie'], 'Âge': [25, 30, 22]})
st.write(data)