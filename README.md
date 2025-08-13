import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import streamlit as st

# --- 1. Chargement des Données depuis les Fichiers CSV ---
try:
    df_labplas = pd.read_csv('labplas_data.csv')
    df_whirlpak = pd.read_csv('whirlpak_data.csv')
    df_labplas['Marque'] = 'Labplas'
    df_whirlpak['Marque'] = 'Whirl-Pak'
except FileNotFoundError:
    st.error("Erreur : Assurez-vous que les fichiers 'labplas_data.csv' et 'whirlpak_data.csv' sont dans le même dossier que le script.")
    st.stop()

# --- 2. Fonctions de Nettoyage et de Conversion ---
def parse_dimension(dim_str):
    if pd.isna(dim_str): return np.nan, np.nan
    parts = str(dim_str).replace(' ', '').split('x')
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except (ValueError, TypeError):
            return np.nan, np.nan
    return np.nan, np.nan

def calculate_area(dim_str):
    if pd.isna(dim_str): return np.nan
    w, h = parse_dimension(dim_str)
    if pd.notna(w) and pd.notna(h):
        return w * h
    return np.nan

def map_fermeture_to_numeric(fermeture_str):
    if pd.isna(fermeture_str): return 0
    fermeture_str = str(fermeture_str).lower()
    if "2 fils ronds" in fermeture_str: return 1
    elif "1 rond, 1 plat" in fermeture_str: return 2
    elif "fils plats" in fermeture_str: return 3
    else: return 0

for df in [df_labplas, df_whirlpak]:
    df['Surface_Pouces_Carres'] = df['Dimension_Pouces'].apply(calculate_area)
    df['Fermeture_Num'] = df['Fermeture'].apply(map_fermeture_to_numeric)

# --- 3. Logique de Correspondance et Interface Streamlit ---
st.set_page_config(layout="wide")
st.title("Analyse Concurrentielle Interactive : Labplas vs. Whirl-Pak")

comparison_features = ['Surface_Pouces_Carres', 'Volume_ml', 'Epaisseur_mils', 'Fermeture_Num']
df_labplas_clean = df_labplas.dropna(subset=comparison_features).copy()
df_whirlpak_clean = df_whirlpak.dropna(subset=comparison_features).copy()

if not df_labplas_clean.empty and not df_whirlpak_clean.empty:
    scaler = MinMaxScaler()
    df_combined = pd.concat([df_labplas_clean[comparison_features], df_whirlpak_clean[comparison_features]])
    scaler.fit(df_combined)
    
    df_labplas_scaled = df_labplas_clean.copy()
    df_whirlpak_scaled = df_whirlpak_clean.copy()

    df_labplas_scaled[comparison_features] = scaler.transform(df_labplas_clean[comparison_features])
    df_whirlpak_scaled[comparison_features] = scaler.transform(df_whirlpak_clean[comparison_features])

    def find_best_match(labplas_row_scaled, df_competitor_scaled, df_competitor_original, features, threshold=0.5):
        labplas_features = labplas_row_scaled[features].values
        competitor_category_df = df_competitor_scaled[df_competitor_scaled['Categorie_Produit'] == labplas_row_scaled['Categorie_Produit']]
        
        if competitor_category_df.empty:
            return None, "Aucun produit dans la catégorie"

        distances = [(euclidean(labplas_features, comp_row[features].values), comp_row['SKU']) for _, comp_row in competitor_category_df.iterrows()]
        distances.sort(key=lambda x: x[0])
        best_dist, best_sku = distances[0]

        if best_dist > threshold:
            return None, f"Aucun concurrent proche (dist: {best_dist:.2f})"
        
        match = df_competitor_original[df_competitor_original['SKU'] == best_sku].iloc[0]
        return match, f"Correspondance: {best_sku} (dist: {best_dist:.2f})"

    results = []
    for index, lp_row in df_labplas_scaled.iterrows():
        original_lp_row = df_labplas_clean.loc[index]
        match_prod, observation = find_best_match(lp_row, df_whirlpak_scaled, df_whirlpak_clean, comparison_features)
        
        row_data = {
            'Labplas_SKU': original_lp_row['SKU'],
            'Categorie_Produit': original_lp_row['Categorie_Produit'],
            'Concurrent_SKU': match_prod['SKU'] if match_prod is not None else "N/A",
            'Observation': observation
        }
        results.append(row_data)

    df_results = pd.DataFrame(results)
    st.header("Tableau des Correspondances")
    st.dataframe(df_results)

    st.header("Visualisation Comparative par SKU")
    selected_sku = st.selectbox("Sélectionnez un SKU Labplas pour comparer :", df_labplas['SKU'].unique())

    if selected_sku:
        labplas_product = df_labplas[df_labplas['SKU'] == selected_sku].iloc[0]
        concurrent_info = df_results[df_results['Labplas_SKU'] == selected_sku]
        
        if not concurrent_info.empty:
            concurrent_sku = concurrent_info.iloc[0]['Concurrent_SKU']
            if concurrent_sku != "N/A":
                whirlpak_product = df_whirlpak[df_whirlpak['SKU'] == concurrent_sku].iloc[0]
                
                radar_features_display = ['Surface (in²)', 'Volume (ml)', 'Épaisseur (mils)', 'Type Fermeture (Num)']
                labplas_values = labplas_product[['Surface_Pouces_Carres', 'Volume_ml', 'Epaisseur_mils', 'Fermeture_Num']].values
                whirlpak_values = whirlpak_product[['Surface_Pouces_Carres', 'Volume_ml', 'Epaisseur_mils', 'Fermeture_Num']].values

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=labplas_values, theta=radar_features_display, fill='toself', name=f"Labplas: {selected_sku}"))
                fig.add_trace(go.Scatterpolar(r=whirlpak_values, theta=radar_features_display, fill='toself', name=f"Whirl-Pak: {concurrent_sku}"))
                
                st.plotly_chart(fig)
            else:
                st.warning(f"Aucun concurrent direct trouvé pour {selected_sku} selon les critères.")
else:
    st.warning("Impossible de traiter les données. Vérifiez les fichiers CSV.")
