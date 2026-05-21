import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# =====================================================
# CHARGEMENT
# =====================================================

model = joblib.load("xgboost_credit_scoring.pkl")
features = joblib.load("features.pkl")

# SHAP explainer
explainer = shap.Explainer(model)

# =====================================================
# TITRE
# =====================================================

st.title("Système de Scoring Crédit")
st.write("Prédiction du risque de défaut de paiement")

# =====================================================
# FORMULAIRE
# =====================================================

age = st.number_input("Age", 18, 100, 30)

revenu_mensuel = st.number_input(
    "Revenu mensuel",
    0,
    10000000,
    200000
)

montant_credit = st.number_input(
    "Montant du crédit",
    0,
    100000000,
    500000
)

duree_credit = st.number_input(
    "Durée du crédit (mois)",
    1,
    360,
    12
)

anciennete_emploi = st.number_input(
    "Ancienneté emploi",
    0,
    50,
    2
)

historique_credit_mois = st.number_input(
    "Historique crédit (mois)",
    0,
    300,
    12
)

personnes_charge = st.number_input(
    "Personnes à charge",
    0,
    20,
    0
)

defaut_anterieur = st.selectbox(
    "Défaut antérieur",
    [0,1]
)

mobile_money = st.selectbox(
    "Mobile money",
    [0,1]
)

zone_urbaine = st.selectbox(
    "Zone urbaine",
    [0,1]
)

genre = st.selectbox(
    "Genre",
    ["Homme", "Femme"]
)

garantie = st.selectbox(
    "Garantie",
    ["Oui", "Non"]
)

education = st.selectbox(
    "Education",
    ["Primaire", "Secondaire", "Universitaire"]
)

# =====================================================
# PREDICTION
# =====================================================

if st.button("Prédire"):

    # =================================================
    # DATAFRAME
    # =================================================

    data = {
        "age": age,
        "revenu_mensuel": revenu_mensuel,
        "montant_credit": montant_credit,
        "duree_credit": duree_credit,
        "anciennete_emploi": anciennete_emploi,
        "historique_credit_mois": historique_credit_mois,
        "personnes_charge": personnes_charge,
        "defaut_anterieur": defaut_anterieur,
        "mobile_money": mobile_money,
        "zone_urbaine": zone_urbaine,
        "genre": genre,
        "garantie": garantie,
        "education": education
    }

    df = pd.DataFrame([data])

    # =================================================
    # ENCODAGE
    # =================================================

    df = pd.get_dummies(df)

    # alignement colonnes
    df = df.reindex(columns=features, fill_value=0)

    # =================================================
    # PREDICTION
    # =================================================

    prediction = model.predict(df)[0]

    proba = model.predict_proba(df)[0][1]

    # score crédit
    credit_score = int((1 - proba) * 1000)

    # =================================================
    # RESULTATS
    # =================================================

    st.subheader("Résultats")

    st.write(f"Probabilité de défaut : {proba:.2%}")

    st.write(f"Score crédit : {credit_score}")

    # =================================================
    # SEGMENT RISQUE
    # =================================================

    if credit_score >= 800:
        st.success("Faible risque")

    elif credit_score >= 600:
        st.warning("Risque moyen")

    else:
        st.error("Risque élevé")

    # =================================================
    # PREDICTION FINALE
    # =================================================

    if prediction == 1:
        st.error("Client à risque de défaut")

    else:
        st.success("Client solvable")

    # =================================================
    # SHAP VALUES
    # =================================================

    st.subheader("Explication de la prédiction")

    shap_values = explainer(df)

    fig, ax = plt.subplots(figsize=(10,5))

    shap.plots.waterfall(
        shap_values[0],
        max_display=10,
        show=False
    )

    st.pyplot(fig)

    st.write("""
    Rouge → augmente le risque de défaut
    
    Bleu → réduit le risque de défaut
    """)