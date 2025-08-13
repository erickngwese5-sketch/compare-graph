import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler

# -------------------------

# 1. Données corrigées Labplas
# -------------------------
data_labplas = {
    'Categorie_Produit': [
        "Standard", "Standard", "Standard", "Standard", "Standard", "Standard", "Standard", "Standard",
        "Write-On", "Write-On", "Write-On", "Write-On", "Write-On", "Write-On", "Write-On",
        "Filtre", "Filtre", "Filtre", "Filtre", "Filtre", "Filtre", "Filtre", "Filtre", "Filtre", "Filtre", "Filtre", "Filtre",
        "Auto-Portant", "Auto-Portant",
        "Opaque", "Opaque", "Opaque",
        "Eponge Polyurethane", "Eponge Polyurethane", "Eponge Polyurethane", "Eponge Polyurethane", "Eponge Polyurethane", "Eponge Polyurethane", "Eponge Polyurethane", "Eponge Polyurethane", "Eponge Polyurethane", "Eponge Polyurethane", "Eponge Polyurethane", "Eponge Polyurethane",
        "Eponge Cellulose", "Eponge Cellulose", "Eponge Cellulose", "Eponge Cellulose", "Eponge Cellulose", "Eponge Cellulose", "Eponge Cellulose", "Eponge Cellulose", "Eponge Cellulose", "Eponge Cellulose", "Eponge Cellulose", "Eponge Cellulose",
        np.nan, np.nan
    ],
    'Marque': ['Labplas'] * 58,
    'SKU': [
        "EPL-3050", "EPL-3070", "EPL-3570", "EPL-4575", "EPL-4590", "EPL-5512", "EPL-5515", "EPL-7015",
        "EPR-3050", "EPR-3070", "EPR-3570", "EPR-4575", "EPR-4590", "EPR-5512", "EPR-7015",
        "EFT-3780A", "EFT-6090A", "SCT-6090A", "EFT-7012A", "SCT-7012A", "EFT-7015A", "EFT-1015A", "SCT-1015A", "EFT-1515A", "SCT-1515A", "EFT-1520A", "SCT-1520A",
        "EPS-4590N", "EPS-6090N",
        "EPR-4590", "EPR-4512", "EPR-4515",
        "KSS-61800", "KSS-61805", "KSS-67815-BPW", "KSS-67815-DE", "KSS-67815-LT", "KSS-67815-NE", "KSS-67910", "KSS-67915", "KSS-67915-BPW", "KSS-67915-DE", "KSS-67915-LT", "KSS-67915-NE",
        "KSS-61100", "KSS-61105", "KSS-67110-BPW", "KSS-67115-BPW", "KSS-67110-DE", "KSS-67115-DE", "KSS-67110-LT", "KSS-67115-LT", "KSS-67110-NE", "KSS-67115-NE", "KSS-67310", "KSS-67315",
        np.nan, np.nan
    ],
    'Description': [
        "Sac standard", "Sac standard", "Sac standard", "Sac standard", "Sac standard", "Sac standard", "Sac standard", "Sac standard",
        "Sac avec zone d'écriture", "Sac avec zone d'écriture", "Sac avec zone d'écriture", "Sac avec zone d'écriture", "Sac avec zone d'écriture", "Sac avec zone d'écriture", "Sac avec zone d'écriture",
        "Sac avec filtre", "Sac avec filtre", "Sac avec filtre", "Sac avec filtre", "Sac avec filtre", "Sac avec filtre", "Sac avec filtre", "Sac avec filtre", "Sac avec filtre", "Sac avec filtre", "Sac avec filtre", "Sac avec filtre",
        "Sac auto-portant", "Sac auto-portant",
        "Sac opaque", "Sac opaque", "Sac opaque",
        "Kit éponge poly, sèche, sans gants", "Kit éponge poly, sèche, avec gants", "Kit éponge poly, hydratée, gants", "Kit éponge poly, hydratée, gants", "Kit éponge poly, hydratée, gants", "Kit éponge poly, hydratée, gants", "Sani-Stick poly, sec, sans gants", "Sani-Stick poly, sec, avec gants", "Sani-Stick poly, hydraté, gants", "Sani-Stick poly, hydraté, gants", "Sani-Stick poly, hydraté, gants", "Sani-Stick poly, hydraté, gants",
        "Kit éponge cellulose, sèche, sans gants", "Kit éponge cellulose, sèche, avec gants", "Kit éponge cellulose, hydratée, sans gants", "Kit éponge cellulose, hydratée, avec gants", "Kit éponge cellulose, hydratée, sans gants", "Kit éponge cellulose, hydratée, avec gants", "Kit éponge cellulose, hydratée, sans gants", "Kit éponge cellulose, hydratée, avec gants", "Kit éponge cellulose, hydratée, sans gants", "Kit éponge cellulose, hydratée, avec gants", "Sani-Stick cellulose, sec, sans gants", "Sani-Stick cellulose, sec, avec gants",
        np.nan, np.nan
    ],
    'Dimension_Pouces': [
        "3 x 5", "3 x 7", "3.5 x 7", "4.5 x 7.5", "4.5 x 9", "5.5 x 12", "5.5 x 15", "7 x 15",
        "3 x 5", "3 x 7", "3.5 x 7", "4.5 x 7.5", "4.5 x 9", "5.5 x 12", "7 x 15",
        "3.75 x 8", "6 x 9", "6 x 9", "7 x 12", "7.5 x 12", "7.5 x 12", "10 x 15", "10 x 15", "15 x 15", "15 x 15", "15 x 20", "15 x 20",
        "4.5 x 9", "6 x 9",
        "4.5 x 9", "4.5 x 12", "4.5 x 15",
        "4.5 x 9", "4.5 x 9", "7.5 x 11.5", "7.5 x 11.5", "7.5 x 11.5", "7.5 x 11.5", "7.5 x 11.5", "7.5 x 11.5", "7.5 x 11.5", "7.5 x 11.5", "7.5 x 11.5", "7.5 x 11.5",
        "4.5 x 9", "4.5 x 9", "7.5 x 11.5", "7.5 x 11.5", "7.5 x 11.5", "7.5 x 11.5", "7.5 x 11.5", "7.5 x 11.5", "7.5 x 11.5", "7.5 x 11.5", "7.5 x 11.5", "7.5 x 11.5",
        np.nan, np.nan
    ],
    'Volume_ml': [
        160, 230, 310, 490, 650, 1400, 1900, 2900,
        120, 210, 280, 490, 650, 1400, 2900,
        270, 910, 910, 2100, 2100, 2100, 5100, 5100, 10500, 10500, 16000, 16000,
        500, 1000,
        650, 960, 1300,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan
    ],
    'Epaisseur_mils': [
        2.5, 2.5, 3, 2.5, 2.5, 3, 3, 3,
        2.5, 2.5, 3, 2.5, 2.5, 3, 3,
        4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4,
        3.5, 3.5,
        2.5, 2.5, 2.5,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan
    ],
    'Fermeture': [
        "2 fils ronds", "2 fils ronds", "2 fils ronds", "2 fils ronds", "2 fils ronds", "2 fils ronds", "2 fils ronds", "2 fils ronds",
        "2 fils ronds", "2 fils ronds", "2 fils ronds", "2 fils ronds", "2 fils ronds", "2 fils ronds", "2 fils ronds",
        "Détachable", "Détachable", "Perforée", "Détachable", "Perforée", "Détachable", "Détachable", "Perforée", "Détachable", "Perforée", "Détachable", "Perforée",
        "2 fils ronds", "2 fils ronds",
        "2 fils ronds", "2 fils ronds", "2 fils ronds",
        "Non applicable", "Non applicable", "Non applicable", "Non applicable", "Non applicable", "Non applicable", "Non applicable", "Non applicable", "Non applicable", "Non applicable", "Non applicable", "Non applicable",
        "Non applicable", "Non applicable", "Non applicable", "Non applicable", "Non applicable", "Non applicable", "Non applicable", "Non applicable", "Non applicable", "Non applicable", "Non applicable", "Non applicable",
        np.nan, np.nan
    ],
    'Barriere_Securite': ['Oui'] * 58,
    'Type_Sterilisation': ['Irradiation'] * 58,
    'Solution': [
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan,
        np.nan, np.nan, np.nan,
        "Sèche", "Sèche", "BPW", "DE", "LT", "NE", "Sèche", "Sèche", "BPW", "DE", "LT", "NE",
        "Sèche", "Sèche", "BPW", "BPW", "DE", "DE", "LT", "LT", "NE", "NE", "Sèche", "Sèche",
        np.nan, np.nan
    ],
    'Type_Eponge': [
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan,
        np.nan, np.nan, np.nan,
        "Polyuréthane", "Polyuréthane", "Polyuréthane", "Polyuréthane", "Polyuréthane", "Polyuréthane", "Polyuréthane", "Polyuréthane", "Polyuréthane", "Polyuréthane", "Polyuréthane", "Polyuréthane",
        "Cellulose", "Cellulose", "Cellulose", "Cellulose", "Cellulose", "Cellulose", "Cellulose", "Cellulose", "Cellulose", "Cellulose", "Cellulose", "Cellulose",
        np.nan, np.nan
    ]
}

# -------------------------
# 2. Données Whirlpak (CORRIGÉ - toutes les listes ont 24 éléments)
# -------------------------
data_whirlpak = {
    'Categorie_Produit': [
        "Standard", "Standard", "Standard", "Standard", "Standard",
        "Write-On", "Write-On", "Write-On",
        "Filtre", "Filtre", "Filtre",
        "Auto-Portant", "Auto-Portant",
        "Opaque", "Opaque",
        "Eponge Polyurethane", "Eponge Polyurethane", "Eponge Polyurethane",
        "Eponge Cellulose", "Eponge Cellulose", "Eponge Cellulose",
        np.nan, np.nan, np.nan  # 3 NaN pour atteindre 24
    ],
    'Marque': ['Whirlpak'] * 24,
    'SKU': [
        "WPL-3050", "WPL-3070", "WPL-4575", "WPL-5512", "WPL-7015",
        "WPR-3050", "WPR-4575", "WPR-5512",
        "WFT-3780A", "WFT-6090A", "WFT-7012A",
        "WPS-4590N", "WPS-6090N",
        "WPN-4590", "WPN-4512",
        "WKS-61800", "WKS-67815-BPW", "WKS-67815-DE",
        "WKS-61100", "WKS-67110-BPW", "WKS-67110-DE",
        np.nan, np.nan, np.nan  # 3 NaN pour atteindre 24
    ],
    'Description': [
        "Sac standard Whirlpak", "Sac standard Whirlpak", "Sac standard Whirlpak", "Sac standard Whirlpak", "Sac standard Whirlpak",
        "Sac avec zone d'écriture Whirlpak", "Sac avec zone d'écriture Whirlpak", "Sac avec zone d'écriture Whirlpak",
        "Sac avec filtre Whirlpak", "Sac avec filtre Whirlpak", "Sac avec filtre Whirlpak",
        "Sac auto-portant Whirlpak", "Sac auto-portant Whirlpak",
        "Sac opaque Whirlpak", "Sac opaque Whirlpak",
        "Kit éponge poly, sèche, sans gants Whirlpak", "Kit éponge poly, hydratée, gants Whirlpak", "Kit éponge poly, hydratée, gants Whirlpak",
        "Kit éponge cellulose, sèche, sans gants Whirlpak", "Kit éponge cellulose, hydratée, sans gants Whirlpak", "Kit éponge cellulose, hydratée, sans gants Whirlpak",
        np.nan, np.nan, np.nan  # 3 NaN pour atteindre 24
    ],
    'Dimension_Pouces': [
        "3 x 5", "3 x 7", "4.5 x 7.5", "5.5 x 12", "7 x 15",
        "3 x 5", "4.5 x 7.5", "5.5 x 12",
        "3.75 x 8", "6 x 9", "7 x 12",
        "4.5 x 9", "6 x 9",
        "4.5 x 9", "4.5 x 12",
        "4.5 x 9", "7.5 x 11.5", "7.5 x 11.5",
        "4.5 x 9", "7.5 x 11.5", "7.5 x 11.5",
        np.nan, np.nan, np.nan  # 3 NaN pour atteindre 24
    ],
    'Volume_ml': [
        160, 230, 490, 1400, 2900,
        120, 490, 1400,
        270, 910, 2100,
        500, 1000,
        650, 960,
        np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan  # 3 NaN pour atteindre 24
    ],
    'Epaisseur_mils': [
        2.5, 2.5, 2.5, 3, 3,
        2.5, 2.5, 3,
        4, 4, 3,
        3.5, 3.5,
        2.5, 2.5,
        np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan  # 3 NaN pour atteindre 24
    ],
    'Fermeture': [
        "2 fils ronds", "2 fils ronds", "2 fils ronds", "2 fils ronds", "2 fils ronds",
        "2 fils ronds", "2 fils ronds", "2 fils ronds",
        "Détachable", "Détachable", "Détachable",
        "2 fils ronds", "2 fils ronds",
        "2 fils ronds", "2 fils ronds",
        "Non applicable", "Non applicable", "Non applicable",
        "Non applicable", "Non applicable", "Non applicable",
        np.nan, np.nan, np.nan  # 3 NaN pour atteindre 24
    ],
    'Barriere_Securite': ['Oui'] * 24,
    'Type_Sterilisation': ['Irradiation'] * 24,
    'Solution': [
        np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan,
        np.nan, np.nan,
        np.nan, np.nan,
        "Sèche", "BPW", "DE",
        "Sèche", "BPW", "DE",
        np.nan, np.nan, np.nan  # 3 NaN pour atteindre 24
    ],
    'Type_Eponge': [
        np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan,
        np.nan, np.nan,
        np.nan, np.nan,
        "Polyuréthane", "Polyuréthane", "Polyuréthane",
        "Cellulose", "Cellulose", "Cellulose",
        np.nan, np.nan, np.nan  # 3 NaN pour atteindre 24
    ]
}

# -------------------------
# 3. DataFrame
# -------------------------
df_labplas = pd.DataFrame(data_labplas)
df_whirlpak = pd.DataFrame(data_whirlpak)

# -------------------------
# 4. Fonctions
# -------------------------
def parse_dimension(dim_str):
    if pd.isna(dim_str): return np.nan, np.nan
    parts = dim_str.replace(' ', '').split('x')
    return float(parts[0]), float(parts[1])

def calculate_area(dim_str):
    if pd.isna(dim_str): return np.nan
    w, h = parse_dimension(dim_str)
    return w * h

def map_fermeture_to_numeric(fermeture_str):
    if pd.isna(fermeture_str): return np.nan
    fermeture_str = fermeture_str.lower()
    if "2 fils ronds" in fermeture_str: return 1
    elif "1 rond, 1 plat" in fermeture_str: return 2
    elif "fils plats" in fermeture_str: return 3
    elif "détachable" in fermeture_str or "perforée" in fermeture_str: return 4
    elif "sans fermeture" in fermeture_str: return 5
    else: return 0

# -------------------------
# 5. Nettoyage & Conversion
# -------------------------
for df in [df_labplas, df_whirlpak]:
    df['Largeur_Pouces'] = df['Dimension_Pouces'].apply(lambda x: parse_dimension(x)[0])
    df['Hauteur_Pouces'] = df['Dimension_Pouces'].apply(lambda x: parse_dimension(x)[1])
    df['Surface_Pouces_Carres'] = df['Dimension_Pouces'].apply(calculate_area)
    df['Fermeture_Num'] = df['Fermeture'].apply(map_fermeture_to_numeric)

# CORRECTION CRUCIALE : Conversion en string et gestion des NaN
df_labplas['Categorie_Produit'] = df_labplas['Categorie_Produit'].astype(str)
df_whirlpak['Categorie_Produit'] = df_whirlpak['Categorie_Produit'].astype(str)

# Remplacer 'nan' (string) par NaN pour un traitement cohérent
df_labplas['Categorie_Produit'] = df_labplas['Categorie_Produit'].replace('nan', np.nan)
df_whirlpak['Categorie_Produit'] = df_whirlpak['Categorie_Produit'].replace('nan', np.nan)

# Filtrer les sacs (exclure les éponges)
df_labplas_sacs = df_labplas[~df_labplas['Categorie_Produit'].str.contains('Eponge', na=True)].copy()
df_whirlpak_sacs = df_whirlpak[~df_whirlpak['Categorie_Produit'].str.contains('Eponge', na=True)].copy()

comparison_features = ['Surface_Pouces_Carres', 'Volume_ml', 'Epaisseur_mils', 'Fermeture_Num']

df_labplas_sacs_clean = df_labplas_sacs.dropna(subset=comparison_features)
df_whirlpak_sacs_clean = df_whirlpak_sacs.dropna(subset=comparison_features)

# Vérifier qu'il y a des données à scaler
if not df_labplas_sacs_clean.empty and not df_whirlpak_sacs_clean.empty:
    scaler = MinMaxScaler()
    scaler.fit(pd.concat([df_labplas_sacs_clean[comparison_features], df_whirlpak_sacs_clean[comparison_features]]))

    df_labplas_sacs_scaled = df_labplas_sacs_clean.copy()
    df_whirlpak_sacs_scaled = df_whirlpak_sacs_clean.copy()

    df_labplas_sacs_scaled[comparison_features] = scaler.transform(df_labplas_sacs_clean[comparison_features])
    df_whirlpak_sacs_scaled[comparison_features] = scaler.transform(df_whirlpak_sacs_clean[comparison_features])
else:
    print("Aucune donnée valide pour le scaling. Vérifiez vos données.")
    exit()

# -------------------------
# 6. Logique de correspondance
# -------------------------
def find_best_match(labplas_row, df_competitor_scaled, df_competitor_original, features, threshold=0.5):
    labplas_features = labplas_row[features].values
    competitor_category_df_scaled = df_competitor_scaled[
        (df_competitor_scaled['Categorie_Produit'] == labplas_row['Categorie_Produit'])
    ]
    if competitor_category_df_scaled.empty:
        return None, "Aucune correspondance dans la même catégorie de produit."
    
    distances = []
    for _, competitor_row in competitor_category_df_scaled.iterrows():
        if not pd.isna(competitor_row['SKU']):
            distances.append((euclidean(labplas_features, competitor_row[features].values), competitor_row['SKU']))
    
    if not distances:
        return None, "Aucune correspondance trouvée."
    
    distances.sort(key=lambda x: x[0])
    best_dist, best_sku = distances[0]
    
    if best_dist > threshold:
        return None, "Aucune correspondance directe ou très proche."
    
    match_product = df_competitor_original[df_competitor_original['SKU'] == best_sku].iloc[0]
    return match_product, f"Correspondance trouvée (distance: {best_dist:.2f})."

# -------------------------
# 7. Comparaison
# -------------------------
results = []
for _, lp_row in df_labplas_sacs_scaled.iterrows():
    match_prod, observation = find_best_match(lp_row, df_whirlpak_sacs_scaled, df_whirlpak_sacs, comparison_features)
    
    result_row = {
        'Categorie_Produit': lp_row['Categorie_Produit'],
        'Labplas_SKU': lp_row['SKU'],
        'Labplas_Description': lp_row['Description'],
        'Labplas_Dimension_Pouces': lp_row['Dimension_Pouces'],
        'Labplas_Volume_ml': lp_row['Volume_ml'],
        'Labplas_Epaisseur_mils': lp_row['Epaisseur_mils'],
        'Labplas_Fermeture': lp_row['Fermeture'],
        'Whirlpak_SKU': match_prod['SKU'] if match_prod is not None else "N/A",
        'Whirlpak_Description': match_prod['Description'] if match_prod is not None else "-",
        'Whirlpak_Dimension_Pouces': match_prod['Dimension_Pouces'] if match_prod is not None else "-",
        'Whirlpak_Volume_ml': match_prod['Volume_ml'] if match_prod is not None else "-",
        'Whirlpak_Epaisseur_mils': match_prod['Epaisseur_mils'] if match_prod is not None else "-",
        'Whirlpak_Fermeture': match_prod['Fermeture'] if match_prod is not None else "-",
        'Observation': observation
    }
    results.append(result_row)

# Créer le DataFrame résultat
df_results = pd.DataFrame(results)

# Afficher les résultats
print(df_results)