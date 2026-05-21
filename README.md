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

- régularisation,

- ajustement des hyperparamètres,

- validation croisée K-Fold,

- ajustement du seuil de décision.


## 📈 Performances du modèle

<img width="289" height="105" alt="classification report xgboost" src="https://github.com/user-attachments/assets/c7b22eff-0eac-4a5e-8155-26a5cfdcdd73" />


Le modèle présente :

- une bonne capacité de généralisation,
  
- une réduction significative du surapprentissage,
  
- un compromis équilibré entre détection du risque et précision.


## 📉 Courbe ROC

La courbe ROC permet d’évaluer la capacité du modèle à distinguer les clients défaillants des clients solvables.

<img width="401" height="333" alt="courbe AUC" src="https://github.com/user-attachments/assets/4a63d717-8ba5-46c1-a413-38d1509f9d01" />

Le modèle obtient un score AUC d’environ **0.75**, indiquant une bonne capacité de discrimination.  
Cela signifie que le modèle est capable de différencier correctement les clients à risque dans une majorité des cas.


## 📉 Courbe d'apprentissage 

<img width="440" height="344" alt="courbe apprentissage" src="https://github.com/user-attachments/assets/40f7b0c8-4d97-4158-ad8c-dec86841dfa0" />


La courbe d’apprentissage montre l’évolution des performances du modèle XGBoost sur les données d’entraînement et de validation.

On observe que les scores d’entraînement et de validation restent relativement **proches**, ce qui indique une **réduction du surapprentissage** et une **bonne capacité de généralisation du modèle**.
Le modèle présente des performances stables avec une validation croisée relativement cohérente entre les différents folds.


## 🧠 Interprétabilité avec SHAP

Le projet intègre SHAP afin d’expliquer les décisions du modèle.

SHAP permet de :

★ comprendre pourquoi un client est considéré à risque,

★ identifier les variables les plus influentes,

★ améliorer la transparence du système.



## 💻 Application Streamlit

Une application web interactive a été développée avec Streamlit.

## Fonctionnalités :

- saisie des informations client,

- prédiction du risque de défaut,

- génération d’un score de crédit,

- segmentation du risque,

- explication des prédictions avec SHAP.


## Aperçu de l’application

<img width="502" height="383" alt="image" src="https://github.com/user-attachments/assets/29859a1a-761a-475a-9f2f-f7f6ed585b0b" />

<img width="500" height="386" alt="image" src="https://github.com/user-attachments/assets/80010182-d6b9-473d-b311-f80851212366" />

<img width="518" height="178" alt="image" src="https://github.com/user-attachments/assets/880e4082-51fb-4f1b-bf9d-8a94d1f1667a" />

<img width="469" height="346" alt="image" src="https://github.com/user-attachments/assets/72903d5a-59a6-44cf-b997-e8dedea3c2b0" />






## Pour lancer l'application 
1- **Pre requis (installer les depandances)** 

```bash
pip install -r requirements.txt
```

2- **lancer**

```bash
streamlit run app.py
```

## 📌 Disclaimer
Le jeu de données utilisé dans ce projet est un dataset simulé généré à des fins éducatives et de démonstration.
Il ne contient aucune donnée réelle de clients.

## 👨‍💻 Auteur

Projet réalisé par **Bolade Mamadou** dans le cadre d’un projet de Data Science appliqué au scoring de crédit et à l’analyse du risque financier.




