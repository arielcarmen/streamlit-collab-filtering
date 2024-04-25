import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


# FONCTIONS

st.header(':blue[Apriori à partir d\'un csv]')

# Création des différents containers
dataframe_container_ = st.empty()

csv_loader = st.container()

support_choice_ = st.container()

if 'dataframe_' not in st.session_state:
    st.session_state['dataframe_'] = pd.DataFrame()

if 'dataframe_' not in st.session_state:
    st.session_state['dataframe_'] = pd.DataFrame()

if 'show_csv_field_' not in st.session_state:
    st.session_state['show_csv_field_'] = True

if 'choose_support_' not in st.session_state:
    st.session_state['choose_support_'] = False

if 'point_predict_container_' not in st.session_state:
    st.session_state['point_predict_container_'] = False

if 'support_' not in st.session_state:
    st.session_state['support_'] = 2


support = st.session_state['support_']
datas_ = st.session_state['dataframe_']

# Definir la valeur de support
def define_support():
    st.session_state['show_csv_field_'] = not st.session_state['show_csv_field_']
    st.session_state['choose_support_'] = not st.session_state['choose_support_']
    st.session_state['dataset_size_'] = len(datas_)
    return 0

def apriori(transactions, min_support):
    """
    Implémentation de l'algorithme Apriori pour trouver les itemsets fréquents.
    """
    n = len(transactions)
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1

    L = [set([item]) for item, count in item_counts.items() if count >= min_support * n]

    k = 2
    while len(L[k - 2]) > 0:
        C_k = apriori_gen(L[k - 2], k)
        L_k = []
        item_counts = defaultdict(int)
        for transaction in transactions:
            for c in C_k:
                if set(c).issubset(transaction):
                    item_counts[frozenset(c)] += 1
        for itemset, count in item_counts.items():
            if count >= min_support * n:
                L_k.append(itemset)
        L.append(L_k)
        k += 1

    return L

def apriori_gen(L_k_minus_1, k):
    """
    Fonction pour générer les candidats pour l'étape suivante de l'algorithme Apriori.
    """
    C_k = []
    for i in range(len(L_k_minus_1)):
        for j in range(i + 1, len(L_k_minus_1)):
            l1 = list(L_k_minus_1[i])
            l2 = list(L_k_minus_1[j])
            l1.sort()
            l2.sort()
            if l1[:k - 2] == l2[:k - 2]:
                c = l1[:k - 2] + [max(l1[k - 2], l2[k - 2])] + [min(l1[k - 2], l2[k - 2])]
                if not any(set(c).issubset(frozenset(x)) for x in L_k_minus_1):
                    C_k.append(c)
    return C_k


# Déroulement du programme
if st.session_state['show_csv_field_']:
    uploaded_file = csv_loader.file_uploader("Charger un csv:", type=["csv"])
    if uploaded_file is not None:
        datas_ = pd.read_csv(uploaded_file)
        st.session_state['dataframe_'] = datas_
        
        csv_loader.button("Valider", on_click= define_support)
    

if st.session_state['choose_support_'] == True:
    dataframe_container_.dataframe(datas_)
    # b) Choisir support
    support = support_choice_.number_input('Valeur du support:', 2, max_value= 10)
    st.session_state['support_'] = support

    if  support_choice_.button("Lancer"):
        aa = 0

        