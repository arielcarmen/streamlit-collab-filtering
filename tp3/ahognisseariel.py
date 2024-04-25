import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# FONCTIONS

st.header(':blue[Apriori à partir d\'un csv]')

# Création des différents containers
dataframe_container_ = st.empty()

csv_loader = st.container()

k_choice_ = st.container()



if 'dataframe_' not in st.session_state:
    st.session_state['dataframe_'] = pd.DataFrame()

if 'show_csv_field_' not in st.session_state:
    st.session_state['show_csv_field_'] = True

if 'choose_k_' not in st.session_state:
    st.session_state['choose_k_'] = False

if 'point_predict_container_' not in st.session_state:
    st.session_state['point_predict_container_'] = False


if 'k_' not in st.session_state:
    st.session_state['k_'] = 2


K = st.session_state['k_']

# Definir la valeur de k
def define_k():
    st.session_state['show_csv_field_'] = not st.session_state['show_csv_field_']
    st.session_state['choose_k_'] = not st.session_state['choose_k_']
    st.session_state['dataset_size_'] = len(datas_)
    scatter_colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(st.session_state['dataset_size_'])]
    st.session_state['scatter_colors_'] = scatter_colors

    value_max = list_[0]
    index_max = 0
    for i in range(1, len(list_)):
        if list_[i] > value_max:
            value_max = list_[i]
            index_max = i
    
    return index_max

# Déroulement du programme
if st.session_state['show_csv_field_']:
    uploaded_file = csv_loader.file_uploader("Charger un csv:", type=["csv"])
    if uploaded_file is not None:
        datas_ = pd.read_csv(uploaded_file)
        st.session_state['dataframe_'] = datas_
        
        csv_loader.button("Valider", on_click= define_k)
    

if st.session_state['choose_k_'] == True:
    dataframe_container_.dataframe(datas_)
    # b) Choisir K
    K = k_choice_.number_input('Valeur du K:', 2, max_value= 10)
    st.session_state['k_'] = K

    if  k_choice_.button("Lancer Kmeans"):
        aa = 0

        