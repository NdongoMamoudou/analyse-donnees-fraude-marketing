import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Charger les mêmes données que dans le notebook
df = pd.read_csv("D:/ProjetPerso/analyse-donnees-fraude-marketing/data/marketing_campaign.csv", sep=";")

# Remplir les valeurs manquantes par la moyenne
df['Income'].fillna(df['Income'].mean(), inplace=True)

# Titre principal
st.title("Tableau de Bord Interactif : Analyse des Clients")
st.write("""
    Ce tableau de bord vous permet d'explorer les données des clients, leurs comportements d'achat et leur segmentation.
""")

# Section de sélection pour l'analyse par niveau d'éducation
st.subheader("Répartition des clients par niveau d'éducation")
education_counts = df['Education'].value_counts()

# Option interactive pour choisir de visualiser un graphique ou un tableau
show_education = st.radio("Sélectionner le type de visualisation", ('Graphique', 'Tableau'))

if show_education == 'Graphique':
    # Visualisation sous forme de diagramme circulaire
    fig, ax = plt.subplots()
    education_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=sns.color_palette("Set2", len(education_counts)))
    ax.set_ylabel("")
    st.pyplot(fig)
else:
    # Affichage du tableau
    st.write(education_counts)

# Section de la répartition des statuts matrimoniaux
st.subheader("Répartition des clients par statut marital")
matrimonial_count = df['Marital_Status'].value_counts()

# Affichage du graphique en barres
fig, ax = plt.subplots(figsize=(10, 6))
matrimonial_count.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
ax.set_title("Répartition des statuts matrimoniaux")
ax.set_xlabel("Statut matrimonial")
ax.set_ylabel("Nombre de clients")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)

# Affichage du tableau des statuts matrimoniaux
st.write(matrimonial_count)

# Section des montants dépensés dans différentes catégories
categories = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
total_spent = df[categories].sum()

st.subheader("Montants totaux dépensés dans différentes catégories")
fig, ax = plt.subplots(figsize=(10, 6))
total_spent.plot(kind='bar', color='coral', edgecolor='black', ax=ax)
ax.set_title("Montants totaux dépensés dans les différentes catégories")
ax.set_xlabel("Catégories")
ax.set_ylabel("Montant total dépensé")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)

# Section des fréquences d'achats par canal
channels = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
total_channels = df[channels].sum()

st.subheader("Fréquence d'achat par canal")
fig, ax = plt.subplots(figsize=(10, 6))
total_channels.plot(kind='bar', color='lightgreen', edgecolor='black', ax=ax)
ax.set_title("Fréquence d'achat par canal")
ax.set_xlabel("Canal d'achat")
ax.set_ylabel("Nombre total d'achats")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)

# Calcul des dépenses totales
df['TotalSpent'] = (df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
                    df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds'])

# Standardisation des données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Income', 'TotalSpent']])

# Application de K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
df['Segment'] = kmeans.fit_predict(scaled_data)

# Section pour la segmentation des clients
st.subheader("Segmentation des Clients : Revenu vs Dépenses Totales")
st.write("""
Cette section permet de visualiser la segmentation des clients basée sur leur revenu et leurs dépenses totales dans différentes catégories de produits.
""")

# Visualisation des clusters : Revenu vs Dépenses
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Income', y='TotalSpent', hue='Segment', data=df, palette='viridis', ax=ax)
ax.set_title("Segmentation des clients : Revenu vs Dépenses Totales")
ax.set_xlabel("Revenu")
ax.set_ylabel("Dépenses Totales")
st.pyplot(fig)

# Option pour afficher un aperçu des données sous forme de tableau
show_data_preview = st.checkbox("Afficher un aperçu des données")
if show_data_preview:
    st.dataframe(df.head())

# Ajouter une option pour télécharger les données filtrées
st.subheader("Télécharger les données filtrées")
st.write("""
Vous pouvez télécharger les données filtrées après avoir effectué des calculs et des transformations.
""")
csv = df.to_csv(index=False)
st.download_button("Télécharger les données", csv, "donnees_clients.csv", "text/csv")

# Option pour ajuster le nombre de clusters
st.sidebar.subheader("Paramètres de K-Means")
num_clusters = st.sidebar.slider("Nombre de clusters", min_value=2, max_value=10, value=4)

# Recalculer K-Means avec le nombre de clusters sélectionné
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Segment'] = kmeans.fit_predict(scaled_data)

# Visualisation avec le nouveau nombre de clusters
st.subheader(f"Segmentation avec {num_clusters} Clusters")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Income', y='TotalSpent', hue='Segment', data=df, palette='viridis', ax=ax)
ax.set_title(f"Segmentation des clients : Revenu vs Dépenses Totales ({num_clusters} Clusters)")
ax.set_xlabel("Revenu")
ax.set_ylabel("Dépenses Totales")
st.pyplot(fig)
