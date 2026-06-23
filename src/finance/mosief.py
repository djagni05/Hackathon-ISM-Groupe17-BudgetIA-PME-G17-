#Calculer les dépenses par catégorie

import pandas as pd

df = pd.read_csv("transactions_pme.csv")

print(df)
depenses = df.groupby("Categorie")["Montant"].sum()

print(depenses)

#Comparer budget et dépenses réelles

budget = pd.read_csv("budget.csv")

reel = df.groupby("Categorie")["Montant"].sum().reset_index()

analyse = budget.merge(reel,on="Categorie")

analyse["Ecart"] = analyse["Montant"] - analyse["Budget"]

print(analyse)

#lecture du fichier

transactions = pd.read_csv("transactions_pme.csv")
print(transactions)

#Convertion en vraies dates(panda lit souvent les dates comme du texte) 

transactions["Date"] = pd.to_datetime(transactions["Date"])

#Extraire le mois(nous voulons connaître les dépenses totales de chaque mois.)

transactions["Mois"] = transactions["Date"].dt.to_period("M").dt.to_timestamp()

#Calculer les dépenses totales par mois

depenses_mensuelles = transactions.groupby("Mois")["Montant"].sum()


#Transformer en tableau
depenses_mensuelles = depenses_mensuelles.reset_index()


#Renommer les colonnes(on crèe les colonnes demandées par prophet)

depenses_mensuelles.columns = ["ds", "y"]

print(depenses_mensuelles)


# MODÈLE PROPHET(donner ces données à Prophet)

from prophet import Prophet

model = Prophet()

model.fit(depenses_mensuelles)

#Prévoir le mois suivant

future = model.make_future_dataframe(periods=30)

forecast = model.predict(future)

print(forecast[["ds","yhat"]].tail())

#Faire des graphiques avec plotly

import plotly.express as px
fig = px.bar(
    analyse,
    x="Categorie",
    y="Montant",
    title="Dépenses réelles"
)

fig.show()

#Calculer la rentabilité

recettes = 3000000

depenses_totales = df["Montant"].sum()

benefice = recettes - depenses_totales

print(benefice)