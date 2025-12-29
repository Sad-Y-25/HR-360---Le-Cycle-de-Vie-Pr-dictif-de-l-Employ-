# 📊 HR Analytics : Analyse, Prédiction et Stratégie RH

Ce projet vise à transformer les données brutes des ressources humaines en outils d'aide à la décision. Il combine l'analyse exploratoire, la prédiction de salaire par Machine Learning et l'étude des facteurs de satisfaction via l'analyse d'association.

---

## 🚀 État d’Avancement du Projet (Mis à jour)

### ✅ Phase 1 : Structuration & EDA
* Nettoyage complet du dataset (1470 employés, 35 variables).
* Analyse visuelle des corrélations (Impact du `JobLevel` et de l'expérience sur le revenu).

### ✅ Phase 2 : Modélisation Prédictive (Régression)
Nous avons implémenté et comparé deux approches pour l'estimation des salaires :

* **Régression Linéaire Simple :** Modèle de base (Expérience uniquement). 
    * *Résultat :* MAPE ~40%, R² ~0.49. (Insuffisant pour la production).
* **Régression Linéaire Multiple :** Modèle avancé (Expérience, Poste, Département).
    * *Résultat :* **MAPE ~20%, R² ~0.87**. Ce modèle est retenu pour l'interface finale.

### ✅ Phase 3 : Analyse de Tendance (ARIMA)
* Transformation du dataset en série temporelle pour analyser l'évolution du salaire moyen.
* **Précision :** MAPE de **10.79%**. 
* **Usage :** Projection de la masse salariale et trajectoires de carrière à long terme.

### ✅ Phase 4 : Analyse d'Association (Apriori)
* **But :** Identifier les "Micro-climats de satisfaction".
* **Résultat :** Extraction de 3 550 règles d'association.
* **Insight clé :** La satisfaction maximale est fortement liée à la synergie entre le `JobLevel_2` et le département `Sales`, ainsi qu'à l'implication en `R&D`.

---

## 🏆 Benchmark des Modèles

| Modèle | Objectif | R² | MAPE | Status |
| :--- | :--- | :---: | :---: | :--- |
| **Régression Simple** | Baseline / Pédagogie | 0.49 | 39.5% | ❌ Rejeté |
| **Régression Multiple** | Prédiction Individuelle | **0.87** | **20.1%** | ✅ Retenu |
| **ARIMA** | Tendance de Carrière | N/A | **10.8%** | 📈 Analytique |

---

## 🛠️ Technologies Utilisées

* **Langage :** Python 3.13
* **Analyse :** Pandas, NumPy
* **Visualisation :** Matplotlib, Seaborn
* **Machine Learning :** Scikit-Learn (Régression), Statsmodels (ARIMA)
* **Data Mining :** Mlxtend (Apriori)
* **Déploiement :** Streamlit (Interface GUI en cours)

---

## 🧭 La "Boussole du Bien-être" (Insights Apriori)

Grâce à l'algorithme Apriori, nous avons classé les facteurs les plus influents sur la satisfaction :
1. **Bon Environnement de travail** (Facteur présent dans 80% des profils "Heureux")
2. **Équilibre Vie Pro/Perso**
3. **Absence d'Heures Supplémentaires**

---

## 🔜 Prochaines Étapes

1. **Interface Streamlit :** Création d'un dashboard interactif permettant de simuler un salaire en temps réel via le modèle de régression multiple.
2. **Classification :** Implémentation du modèle Random Forest pour prédire l'Attrition (départs volontaires).
3. **Clustering :** Segmentation des employés pour identifier les profils à haut potentiel.