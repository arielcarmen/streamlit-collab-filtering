import streamlit as st
import pandas as pd
from collections import defaultdict
import time


# FONCTIONS

st.header(':blue[Apriori à partir d\'un csv]')

# Création des différents containers
dataframe_container_ = st.empty()

csv_loader = st.container()

support_choice_ = st.container()

results_ = st.container()

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

if 'max_support_' not in st.session_state:
    st.session_state['max_support_'] = 2

if 'itemset_' not in st.session_state:
    st.session_state['itemset_'] = {}

if 'transactions_' not in st.session_state:
    st.session_state['transactions_'] = {}

support = st.session_state['support_']
max_support = st.session_state['max_support_']
datas_ = st.session_state['dataframe_']
itemset_ = st.session_state['itemset_']
transactions_ = st.session_state['transactions_']

# Definir la valeur de support
def define_support():
    st.session_state['show_csv_field_'] = not st.session_state['show_csv_field_']
    st.session_state['choose_support_'] = not st.session_state['choose_support_']
    
    return 0

def generate_itemset():
    all_products = datas_.values.flatten().tolist()
    itemset = list(set(all_products))
    return itemset

def generate_transactions(dataset):
    transactions = []
    for index, row in dataset.iterrows():
        transactions.append(set(row.dropna().astype(str).tolist()))
    return transactions

def combinations(lst):
    if len(lst) == 0:
        return [[]]
    else:
        head = lst[0]
        tail = lst[1:]
        tail_combinations = combinations(tail)
        result = []
        for combination in tail_combinations:
            result.append(combination)
            result.append([head] + combination)
        return result

def calculate_support(combination):
    count = 0
    for order in transactions_:
        if set(combination).issubset(order):
            count += 1
    return count

def generate_frequent_items(lst, k):
    all_combinations = combinations(lst)
    combinations_dict = {}
    supports_dict = {}
    for i, combination in enumerate(all_combinations):
        combination_support = calculate_support(combination)
        if combination_support >= k:
            combinations_dict[f"Combinaison_{i+1}"] = combination
            supports_dict[f"Combinaison_{i+1}"] = combination_support
    return combinations_dict, supports_dict

def common_elements(main_list, target_list):
    if '' in main_list or '' in target_list:
        return True
    
    if all(elem in main_list for elem in target_list):
        return True
    else:
        for element in target_list:
            if element in main_list:
                return True
    return False

def generate_association_rules(items_dic, supports_dic):
    association_rules_dict = {}
    for key in items_dic:
        if not items_dic[key] == ['']:
            dic_copy = items_dic.copy()
            dic_copy.pop(key)
            subitems = []
            for other_key in dic_copy:
                if set(items_dic[other_key]).issubset(items_dic[key]):
                    subitems.append(other_key)
            
            for subitem in subitems:
                if not items_dic[subitem] == ['']:
                    _subitems = subitems.copy()
                    _subitems.remove(subitem)
                    for item in _subitems:
                        if not set(items_dic[item]).issubset(items_dic[subitem]) and not common_elements(items_dic[subitem],items_dic[item]):
                            association_rules_dict[f"{items_dic[subitem]} -> {items_dic[item]}"] = supports_dic[key]/supports_dic[subitem]

    top_10_list = sorted(association_rules_dict.items(), key=lambda item: item[1], reverse=True)

    top_10_dict = dict(top_10_list[:10])

    return top_10_dict

# Déroulement du programme
if st.session_state['show_csv_field_']:
    uploaded_file = csv_loader.file_uploader("Charger un csv:", type=["csv"])
    if uploaded_file is not None:
        datas_ = pd.read_csv(uploaded_file, header= None, on_bad_lines='skip', dtype=str, na_filter=False)
        transactions_ = generate_transactions(datas_)
        max_support = len(datas_)
        st.session_state['dataframe_'] = datas_
        st.session_state['transactions_'] = transactions_
        st.session_state['max_support_'] = max_support
        csv_loader.button("Valider", on_click= define_support)

if st.session_state['choose_support_'] == True:
    # b) Choisir support
    support = support_choice_.number_input('Valeur du support:', 2, max_value= max_support)
    st.session_state['support_'] = support

    if  support_choice_.button("Lancer"):
        itemset = generate_itemset()

        item_combinations, item_supports = generate_frequent_items(itemset, support) 
        valid_items = [['Sous-ensembles', 'Supports']]
        top_10 = [['Association', 'Confiance']]

        for icombination, isupport in zip(item_combinations, item_supports):
            valid_items.append([item_combinations[icombination], item_supports[isupport]])

        df = pd.DataFrame(valid_items[1:], columns=valid_items[0])

        # c) Affichage des items fréquents dans un tableau
        results_.subheader(f'Ensembles ayant un support minimal de {support}:')
        results_.table(df)

        item_combinations.pop('Combinaison_1')

        association_rules = generate_association_rules(item_combinations, item_supports)

        for rule in association_rules:
            top_10.append([rule, association_rules[rule]])

        # d) Affichage du top 10
        top_10_df = pd.DataFrame(top_10[1:], columns=top_10[0])
        results_.subheader(f'Top 10 des règles d\'association:')
        results_.table(top_10_df)

        
        

        