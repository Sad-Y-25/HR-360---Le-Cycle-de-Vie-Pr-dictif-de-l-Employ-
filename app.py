
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu
import os
import ast

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Insight HR", layout="wide")

# --- 2. CSS DARK MODE & STYLES ---
st.markdown("""
<style>
    /* RESET GLOBAL */
    .stApp { background-color: #000000; color: #FFFFFF; font-family: 'Helvetica Neue', sans-serif; }
    
    /* HERO SECTION (AI HARDWARE / CIRCUIT IMAGE) */
    .hero {
        position: relative;
        /* Image Hardware/Circuit Bleu Sombre */
        background-image: url('https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=1740&auto=format&fit=crop'); 
        background-size: cover;
        background-position: center;
        height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 40px;
        border-bottom: 1px solid #333;
    }
    .hero::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.5); /* Assombrir pour le texte */
    }
    .hero-content {
        position: relative;
        z-index: 2;
        text-align: center;
    }
    .hero-title {
        font-size: 5rem;
        font-weight: 900;
        color: #FFFFFF;
        letter-spacing: 5px;
        text-shadow: 0 0 20px rgba(0,0,0,0.8);
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1.5rem;
        color: #DDDDDD;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 10px;
    }

    /* INPUTS NOIRS */
    .stTextInput input, .stNumberInput input, .stSelectbox > div > div, .stSlider {
        background-color: #000000 !important; 
        color: #FFFFFF !important; 
        border: 1px solid #333333 !important;
        border-radius: 0px !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #FFFFFF !important;
    }
    
    /* OUTPUT DIVS (Style Unifié : Bordure Blanche, Taille Raisonnable) */
    .res-card {
        background: #000000;
        border: 2px solid #FFFFFF; /* BORDURE BLANCHE */
        padding: 25px;
        text-align: center;
        margin-bottom: 15px;
        color: #FFFFFF;
    }
    
    /* FEATURE BOXES (Accueil) */
    .feature-box {
        background: #0A0A0A;
        padding: 30px;
        border: 1px solid #333;
        transition: all 0.3s;
    }
    .feature-box:hover { border-color: #FFF; }
    
    /* BUTTONS */
    .stButton button {
        background-color: #FFFFFF;
        color: #000000;
        border-radius: 0px;
        font-weight: bold;
        text-transform: uppercase;
        border: none;
        padding: 12px 24px;
        width: 100%;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #CCCCCC;
    }

    /* REMOVE STUFF */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. CHARGEMENT DONNÉES ---
@st.cache_resource
def load_data():
    models = {'classification': {}, 'regression': {}, 'clustering': None, 'association': None}
    path = "Models/"
    
    if os.path.exists(path):
        # A. Classification
        for f in [x for x in os.listdir(path) if "modele_classif_" in x]:
            name = f.replace('modele_classif_', '').replace('.pkl', '').replace('_', ' ').upper()
            models['classification'][name] = joblib.load(path + f)
            
        # B. Régression
        if os.path.exists(path + "modele_salaire_lineaire.pkl"):
            models['regression']['Simple'] = joblib.load(path + "modele_salaire_lineaire.pkl")
        if os.path.exists(path + "modele_salaire_avance.pkl"):
            models['regression']['Multiple'] = joblib.load(path + "modele_salaire_avance.pkl")
            
        # C. Clustering
        if os.path.exists(path + "modele_kmeans.pkl"):
            models['clustering'] = joblib.load(path + "modele_kmeans.pkl")
            
        # D. Association
        if os.path.exists(path + "regles_satisfaction.pkl"):
            models['association'] = joblib.load(path + "regles_satisfaction.pkl")
        elif os.path.exists(path + "regles_association.csv"):
            models['association'] = pd.read_csv(path + "regles_association.csv")
            
    return models

data = load_data()

# --- 4. NAVIGATION ---
selected = option_menu(
    menu_title=None,
    options=["Home", "Régression", "Prédiction Départ", "Segmentation", "Satisfaction"],
    icons=["house", "currency-dollar", "shield-exclamation", "grid-3x3", "diagram-3"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "black", "border-bottom": "1px solid #333"},
        "nav-link": {"font-size": "13px", "color": "#888", "margin":"0px", "text-transform": "uppercase"},
        "nav-link-selected": {"background-color": "white", "color": "black"},
    }
)

# --- 5. UTILS ---
def format_input(input_dict, artefacts):
    df = pd.DataFrame([input_dict])
    if isinstance(artefacts, dict) and 'features' in artefacts:
        cols = artefacts['features']
        df_encoded = pd.get_dummies(df)
        df_final = df_encoded.reindex(columns=cols, fill_value=0)
        if 'scaler' in artefacts:
            df_final = pd.DataFrame(artefacts['scaler'].transform(df_final), columns=cols)
        return df_final
    return pd.get_dummies(df)

# =============================================================================
# HOME
# =============================================================================
if selected == "Home":
    st.markdown("""
    <div class="hero">
        <div class="hero-content">
            <h1 class="hero-title">INSIGHT HR</h1>
            <div class="hero-subtitle">DATA INTELLIGENCE SUITE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="feature-box">
            <div style="font-size:30px; margin-bottom:10px;">⚖️</div>
            <div class="feature-title">JUSTICE SALARIALE</div>
            <p class="feature-desc">Estimation précise basée sur le marché.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-box">
            <div style="font-size:30px; margin-bottom:10px;">🛡️</div>
            <div class="feature-title">SÉCURISATION TALENTS</div>
            <p class="feature-desc">Anticipation des départs à risque.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-box">
            <div style="font-size:30px; margin-bottom:10px;">🔍</div>
            <div class="feature-title">PROFILAGE RH</div>
            <p class="feature-desc">Segmentation automatique des équipes.</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# RÉGRESSION
# =============================================================================
elif selected == "Régression":
    st.header("ESTIMATION SALARIALE")
    st.info("ℹ️ **Note :** Le modèle 'Simple' ne prend en compte QUE l'expérience globale. Le modèle 'Multiple' analyse tout le profil.")
    st.markdown("---")

    with st.form("reg"):
        c1, c2, c3 = st.columns(3)
        exp = c1.number_input("EXPÉRIENCE (Années)", 0, 40, 5)
        level = c2.slider("NIVEAU (1-5)", 1, 5, 2)
        edu = c3.selectbox("FORMATION (1-5)", [1, 2, 3, 4, 5])
        
        c4, c5 = st.columns(2)
        role = c4.selectbox("POSTE", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager", "Manufacturing Director", "Sales Representative"])
        dept = c5.selectbox("DÉPARTEMENT", ["Sales", "Research & Development", "Human Resources"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        calc = st.form_submit_button("CALCULER L'ESTIMATION")
        
    if calc:
        inputs = {'TotalWorkingYears': exp, 'JobLevel': level, 'JobRole': role, 'Department': dept, 'Education': edu, 'Age': 30}
        
        c1, c2 = st.columns(2)
        
        # Simple
        if 'Simple' in data['regression']:
            model = data['regression']['Simple']
            try:
                X_simp = pd.DataFrame([[exp]], columns=['TotalWorkingYears'])
                pred = model.predict(X_simp)[0] if not isinstance(model, dict) else model['model'].predict(X_simp)[0]
                with c1:
                    st.markdown(f"""
                    <div class="res-card">
                        <div style="color:#AAA; font-size:12px; margin-bottom:5px;">RÉGRESSION SIMPLE</div>
                        <div style="font-size:35px; font-weight:bold;">{pred:,.0f} $</div>
                    </div>
                    """, unsafe_allow_html=True)
            except: pass
            
        # Multiple
        if 'Multiple' in data['regression']:
            model = data['regression']['Multiple']
            X_mult = format_input(inputs, model)
            pred = model['model'].predict(X_mult)[0]
            with c2:
                st.markdown(f"""
                <div class="res-card" style="border-color:#FFFFFF">
                    <div style="color:#AAA; font-size:12px; margin-bottom:5px;">RÉGRESSION MULTIPLE</div>
                    <div style="font-size:35px; font-weight:bold;">{pred:,.0f} $</div>
                </div>
                """, unsafe_allow_html=True)

# =============================================================================
# PRÉDICTION DÉPART (Classification)
# =============================================================================
elif selected == "Prédiction Départ":
    st.header("ANALYSE DE RISQUE")
    st.markdown("---")
    
    with st.expander("PROFIL COLLABORATEUR", expanded=True):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("AGE", 18, 65, 30)
        income = c2.number_input("SALAIRE ($)", 1000, 20000, 5000)
        overtime = c3.selectbox("HEURES SUPP.", ["Yes", "No"])
        
        c4, c5, c6 = st.columns(3)
        distance = c4.slider("DISTANCE (km)", 1, 50, 10)
        years = c5.number_input("ANCIENNETÉ", 0, 40, 5)
        satisfaction = c6.slider("SATISFACTION (1-4)", 1, 4, 2)
        
        st.markdown("<br>", unsafe_allow_html=True)
        btn = st.button("LANCER L'AUDIT")
        
    if btn:
        inputs = {'Age': age, 'MonthlyIncome': income, 'OverTime': overtime, 'DistanceFromHome': distance, 'YearsAtCompany': years, 'JobSatisfaction': satisfaction, 'TotalWorkingYears': years+2, 'NumCompaniesWorked': 1, 'JobLevel': 2, 'StockOptionLevel': 0, 'YearsSinceLastPromotion': 1}
        
        cols = st.columns(3)
        i = 0
        for name, artefact in data['classification'].items():
            model = artefact['model'] if isinstance(artefact, dict) else artefact
            X = format_input(inputs, artefact)
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0][1]
                    res = 1 if proba > 0.5 else 0
                else:
                    res = model.predict(X)[0]
                    proba = 1.0 if res == 1 else 0.0
                
                # EXCEPTIONS COULEURS (ROUGE/VERT UNIQUEMENT ICI)
                border_color = "#EF4444" if res == 1 else "#10B981" # Rouge ou Vert
                text_color = "#EF4444" if res == 1 else "#10B981"
                txt = "DÉPART PROBABLE" if res == 1 else "COLLABORATEUR STABLE"
                
                with cols[i % 3]:
                    st.markdown(f"""
                    <div style="background:black; border: 2px solid {border_color}; padding:20px; text-align:center; margin-bottom:10px;">
                        <div style="font-size:12px; color:#AAA; margin-bottom:10px;">{name}</div>
                        <div style="font-size:18px; font-weight:bold; color:{text_color}; text-transform:uppercase;">{txt}</div>
                        <div style="font-size:12px; margin-top:5px; color:white;">Confiance: {proba:.0%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                i+=1
            except: pass

# =============================================================================
# SEGMENTATION (CLUSTERING) 
# =============================================================================
elif selected == "Segmentation":
    st.header("PROFILAGE AUTOMATIQUE (K-MEANS)")
    st.markdown("---")
    
    if data['clustering']:
        art = data['clustering']
        # Inputs avec valeurs par défaut critiques pour influencer le cluster
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("AGE", 18, 60, 25)
        inc = c2.number_input("REVENU MENSUEL", 1000, 20000, 3000)
        exp = c3.number_input("EXPÉRIENCE", 0, 40, 2)
        c4, c5 = st.columns(2)
        yrs = c4.number_input("ANCIENNETÉ SOC.", 0, 40, 1)
        lvl = c5.slider("NIVEAU (1-5)", 1, 5, 1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("IDENTIFIER LE SEGMENT"):
            inp = {'Age': age, 'MonthlyIncome': inc, 'TotalWorkingYears': exp, 'YearsAtCompany': yrs, 'JobLevel': lvl}
            # Ordre strict
            X = pd.DataFrame([inp])[['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'JobLevel']]
            if 'scaler' in art: 
                X = art['scaler'].transform(X)
            cid = art['model'].predict(X)[0]
            label = art.get('labels', {}).get(cid, f"CLUSTER {cid}")
            
            st.markdown(f"""
            <div class="res-card">
                <h2 style="color:white; font-size:40px; margin-bottom:10px;">{label}</h2>
                <p style="color:#AAA; margin-bottom:5px;">PROFIL ID : {cid}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Modèle K-Means manquant.")

# =============================================================================
# SATISFACTION (ASSOCIATION) 
# =============================================================================
elif selected == "Satisfaction":
    st.header("ANALYSE DE SATISFACTION")
    st.caption("Déduit la satisfaction basée sur la présence de facteurs positifs.")
    st.markdown("---")
    
    if data['association'] is None:
        st.error("Règles non trouvées.")
    else:
        with st.form("assoc"):
            c1, c2 = st.columns(2)
            dept = c1.selectbox("DÉPARTEMENT", ['Sales', 'Research & Development', 'Human Resources'])
            level = c2.selectbox("NIVEAU", [1, 2, 3, 4, 5])
            ot = c1.selectbox("HEURES SUPP", ['Oui', 'Non'])
            wl = c2.selectbox("ÉQUILIBRE VIE (1-4)", [1, 2, 3, 4])
            env = c1.selectbox("ENVIRONNEMENT (1-4)", [1, 2, 3, 4])
            inv = c2.selectbox("IMPLICATION (1-4)", [1, 2, 3, 4])
            st.markdown("<br>", unsafe_allow_html=True)
            check = st.form_submit_button("ANALYSER")
        
        if check:
            # GÉNÉRATION DES TAGS (UNIQUEMENT CEUX QUI EXISTENT DANS LE MODÈLE)
            tags = set()
            tags.add(f"Dep_{dept}")
            if ot == 'Non': tags.add("Pas_Heures_Sup")
            if level >= 3: tags.add("Niveau_Expert")
            if wl >= 3: tags.add("Equilibre_Sain")
            if env >= 3: tags.add("Bon_Environnement")
            if inv >= 3: tags.add("Bonnes_Relations")
            
            # FILTRE DE SÉCURITÉ : Si on n'a QUE le département, on arrête tout.
            # Cela empêche la règle "Si Sales -> Happy" de s'activer pour un profil stressé.
            if len(tags) <= 1:
                st.error("⚠️ PROFIL À RISQUE DÉTECTÉ")
                st.markdown("""
                <div class="res-card" style="border-color: #EF4444;">
                    <h2 style="color: #EF4444;">FAIBLE SATISFACTION (PROBABLE)</h2>
                    <p style="color: #AAA;">Aucun facteur protecteur (Équilibre, Environnement, etc.) n'a été détecté.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                rules = data['association']
                matches = []
                if isinstance(rules, pd.DataFrame):
                    for idx, r in rules.iterrows():
                        # Nettoyage set
                        ra = r['antecedents']
                        ants = set(ast.literal_eval(ra.replace("frozenset({", "{").replace("})", "}"))) if isinstance(ra, str) else set(ra)
                        rc = r['consequents']
                        cons = list(ast.literal_eval(rc.replace("frozenset({", "{").replace("})", "}")))[0] if isinstance(rc, str) else list(rc)[0]
                        
                        if "SATISFACTION" in str(cons):
                            if ants.issubset(tags):
                                matches.append({'R': cons, 'Conf': r['confidence'], 'L': r['lift'], 'Ants': list(ants)})
                
                if matches:
                    best = sorted(matches, key=lambda x: (len(x['Ants']), x['L']), reverse=True)[0]
                    res = str(best['R']).replace("JobSatisfaction_", "").replace("_", " ")
                    
                    # AFFICHAGE avec design 2nd version
                    st.markdown(f"""
                    <div class="res-card" style="border-color: #10B981;">
                        <h2 style="color:#10B981; font-size:40px; margin-bottom:10px;">{res}</h2>
                        <p style="color:#10B981; margin-bottom:5px;">FACTEURS DÉCLENCHEURS : {best['Ants']}</p>
                        <small style="color:#AAA">FIABILITÉ : {best['Conf']:.0%}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Aucune règle exacte trouvée pour cette combinaison.")
                    st.write("Tags détectés : ", list(tags))