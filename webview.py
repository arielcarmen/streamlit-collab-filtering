import streamlit as st
import numpy as np
import pandas as pd
from functions import predict_rating, generate_random_table, highlight_changes

st.header(':blue[Prediction de note]')


container = st.empty()
scores_grid = pd.DataFrame()

# if 'table_generated' not in st.session_state and 'dataframe' not in st.session_state:
#     column1, column2 = st.columns(2)
#     with column1:
#         movies_number = st.number_input('Nombre de films:', 5)
#     with column2:
#         users_number = st.number_input('Nombre d\'utilisateurs:', 5)

#     if st.button("Générer le tableau"):
#         scores_grid = generate_random_table(users_number= users_number, movies_number= movies_number)
#         st.session_state.dataframe = scores_grid
#         container.dataframe(scores_grid)
#         st.session_state.table_generated = True
#         st.session_state.can_predict = True


###### Pour utilisation avec csv

initial_dataframe = pd.read_csv("class_table.csv")

if 'dataframe' not in st.session_state:
    main_dataframe = pd.read_csv("class_table.csv")
    st.session_state.dataframe = main_dataframe

scores_grid = st.session_state.dataframe
scores_grid = scores_grid.apply(pd.to_numeric, axis= 1)
container.dataframe(scores_grid)
st.session_state.can_predict = True

if 'can_predict' in st.session_state and st.session_state.can_predict == True:
    column3, column4, column5 = st.columns(3)
    with column3:
        top_number = st.number_input('N:', 2)
    with column4:
        user_to_predict = st.number_input('Utilisateur:', 1)
    with column5:
        movie_to_predict = st.number_input('Film à noter:', 1)

    if st.button(label= "Predire"):

        if  movie_to_predict < 1 or movie_to_predict > scores_grid.count(axis=1).size:
            st.error(f"Le film {movie_to_predict} n'existe pas !")
        elif user_to_predict < 1 or user_to_predict > scores_grid.count(axis=0).size :
            st.error(f"L'utilisateur {user_to_predict} n'existe pas !")
        elif top_number > scores_grid.count(axis=1).size or top_number < 1:
            st.error(f"Le top {top_number} n'est pas en accord avec le nombre de films !")
        else:
            existing_score = scores_grid.iloc[movie_to_predict -1, user_to_predict -1]
            if not pd.isna(existing_score):
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
                predicted_rating = round(predicted_rating)

                if predicted_rating == 0 or 0 < predicted_rating > 5 :
                    st.write("Impossible de predire cette note à partir des données")
                else: 
                    scores_grid.iloc[movie_to_predict -1, user_to_predict -1] = predicted_rating
                    # st.session_state.dataframe = highlight_changes(df_before= initial_dataframe, df_after=scores_grid)
                    st.session_state.dataframe = scores_grid
                    st.write(f"L'utilisateur {user_to_predict} pourrait donner au film {movie_to_predict}, un score de: {round(predicted_rating, 2)}")
                    container.dataframe(scores_grid)




        