def create_apriori_dict(products):
    apriori_dict = {}
    
    # Ajouter les produits uniques
    for product in set(products):
        apriori_dict[frozenset([product])] = []
    
    # Ajouter les combinaisons de produits
    for length in range(2, len(set(products)) + 1):
        for combo in generate_combos(set(products), length):
            apriori_dict[frozenset(combo)] = []
    
    return apriori_dict

def generate_combos(items, length):
    combos = []
    for combo in combinations(items, length):
        combos.append(combo)
    return combos

def combinations(items, length):
    combos = []
    for combo in combine(items, length):
        combos.append(combo)
    return combos

def combine(items, length):
    if length == 0:
        yield []
    else:
        for i, item in enumerate(items):
            for combo in combine(items[i+1:], length-1):
                yield [item] + combo

# Exemple d'utilisation
product_list = ['A', 'B', 'C', 'D', 'E']
apriori_dict = create_apriori_dict(product_list)

print("Dictionnaire Apriori :")
for key, value in apriori_dict.items():
    print(f"{key}: {value}")