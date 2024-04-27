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

if 'support_' not in st.session_state:
    st.session_state['support_'] = 2

if 'itemset_' not in st.session_state:
    st.session_state['itemset_'] = {}

support = st.session_state['support_']
datas_ = st.session_state['dataframe_']
itemset_ = st.session_state['itemset_']

# Definir la valeur de support
def define_support():
    st.session_state['show_csv_field_'] = not st.session_state['show_csv_field_']
    st.session_state['choose_support_'] = not st.session_state['choose_support_']
    
    return 0

def generate_transactions(dataset):
    transactions = []
    for index, row in dataset.iterrows():
        transactions.append(set(row.dropna().astype(str).tolist()))
    return transactions

def generate_itemset(transactions):
    itemsets = set()
    for transaction in transactions:
        for item in transaction:
            itemsets.add(item)
    return itemsets

# Fonction pour calculer le support des itemsets
def calculate_support(transactions, itemsets, min_support):
    support_data = {}
    for itemset in itemsets:
        for transaction in transactions:
            if itemset.issubset(transaction):
                support_data[itemset] = support_data.get(itemset, 0) + 1
    num_transactions = len(transactions)
    return {item: support / num_transactions for item, support in support_data.items() if support / num_transactions >= min_support}

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
        transactions = generate_transactions(datas_)
        itemsets = generate_itemset(transactions)
        print(itemsets)
        

        