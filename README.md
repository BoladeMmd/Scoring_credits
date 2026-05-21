# 💳 Système de Scoring de Crédit avec XGBoost, SHAP et Streamlit

## 📌 Présentation du projet

Ce projet consiste à développer un système intelligent de scoring de crédit permettant de prédire le risque de défaut de paiement d’un client à partir de différentes informations financières et socio-économiques.

L’objectif principal est d’aider les institutions financières à :
- détecter les clients à risque,
- réduire les pertes liées aux impayés,
- améliorer la prise de décision lors de l’octroi de crédit.

Le projet combine :
- Machine Learning avec XGBoost,
- Intelligence Artificielle explicable avec SHAP,
- Déploiement interactif avec Streamlit.

---

## 🎯 Objectifs

✅ Prédire le défaut de paiement d’un client  
✅ Générer un score de crédit  
✅ Expliquer les décisions du modèle grâce à SHAP  
✅ Déployer une application web interactive avec Streamlit  

---

## 📂 Structure du projet

```bash
Scoring_credits/
│
├── app.py
├── requirements.txt
├── xgboost_credit_scoring.pkl
├── features.pkl
├── scaler.pkl
├── README.md
│
├── dataset/
│   └── dataset_credi_senegal.csv
│
├── notebooks/
│   └── risk-credits.ipynb
│
└── images/
    ├── matrice confusion.png
    ├── classification repport.png
    ├── courbe apprentissage.png
    └── courbe AUC.png
```



## 🛠️ Technologies utilisées

-Python
-Pandas
-NumPy
-Scikit-learn
-XGBoost
-SHAP
-Streamlit
-Matplotlib
-Joblib


## 📊 Variables utilisées

Le modèle utilise plusieurs variables décrivant le profil du client du dataset  :

- âge,
- revenu mensuel,
- montant du crédit,
- durée du crédit,
- ancienneté emploi,
- historique de crédit,
- défaut antérieur,
- niveau d’éducation,
- garantie,
- nombre de personnes à charge,
- mobile money,
- zone urbaine.
etc.


## 🔍 Analyse exploratoire des données (EDA)

Une analyse exploratoire complète a été réalisée afin de :

- comprendre la distribution des variables,

- identifier les facteurs influençant le défaut de paiement,

- analyser les relations entre les variables et la cible.

Les visualisations utilisées incluent :

- distributions,

- countplots,

- heatmaps,

- courbes ROC,

- courbes d’apprentissage.

## 🤖 Modèles testés

Plusieurs modèles de classification ont été évalués :

- ## Régression Logistique,
- ## Random Forest,
- ## XGBoost.

Le modèle XGBoost a été retenu grâce à ses meilleures performances globales.

## ⚙️ Optimisation du modèle

Le modèle XGBoost a été optimisé afin de :

- ## réduire le surapprentissage,
  
- ## améliorer la généralisation,
  
- ## optimiser le rappel des défauts de paiement.


## Techniques utilisées :

★régularisation,

★ajustement des hyperparamètres,

★validation croisée K-Fold,

★ajustement du seuil de décision.


## 📈 Performances du modèle

<img width="289" height="105" alt="classification report xgboost" src="https://github.com/user-attachments/assets/c7b22eff-0eac-4a5e-8155-26a5cfdcdd73" />


Le modèle présente :

- une bonne capacité de généralisation,
  
- une réduction significative du surapprentissage,
  
- un compromis équilibré entre détection du risque et précision.


## 📉 Courbe ROC

La courbe ROC permet d’évaluer la capacité du modèle à distinguer les clients défaillants des clients solvables.

<img width="401" height="333" alt="courbe AUC" src="https://github.com/user-attachments/assets/4a63d717-8ba5-46c1-a413-38d1509f9d01" />


## 📉 Courbe d'apprentissage 

<img width="440" height="344" alt="courbe apprentissage" src="https://github.com/user-attachments/assets/40f7b0c8-4d97-4158-ad8c-dec86841dfa0" />



## 🧠 Interprétabilité avec SHAP

Le projet intègre SHAP afin d’expliquer les décisions du modèle.

SHAP permet de :

comprendre pourquoi un client est considéré à risque,

identifier les variables les plus influentes,

améliorer la transparence du système.

Exemple d’explication SHAP

## 💻 Application Streamlit

Une application web interactive a été développée avec Streamlit.

## Fonctionnalités :

saisie des informations client,

prédiction du risque de défaut,

génération d’un score de crédit,

segmentation du risque,

explication des prédictions avec SHAP.


# Aperçu de l’application

