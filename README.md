# 📊 HR Analytics : Analyse et Prédiction de l'Attrition des Employés

Ce projet vise à analyser les données des ressources humaines afin de comprendre les facteurs influençant le départ des employés (**Attrition**) et de visualiser les tendances salariales et démographiques au sein de l'entreprise.

La méthodologie adoptée repose sur une approche rigoureuse de **structuration des données**, d’**exploration visuelle (EDA)** et de **préparation à la modélisation prédictive**.

---

## 📂 Structure du Projet

Le projet est organisé en plusieurs phases :

1. **Structuration & Nettoyage des Données**
2. **Analyse Exploratoire & Visualisation (EDA)**
3. **Modélisation Prédictive (à venir)**

---

## 🛠️ Installation et Prérequis

Assurez-vous d’avoir **Python 3.x** installé, puis installez les bibliothèques nécessaires :

## 💾 Jeu de Données

- **Nom du fichier** : `WA_Fn-UseC_-HR-Employee-Attrition.csv`
- **Taille** : 1470 employés, 35 variables
- **Variable cible** : `Attrition` (Yes / No)

### 📌 Colonnes importantes
- `Age`
- `MonthlyIncome`
- `JobRole`
- `YearsAtCompany`
- `OverTime`
- `JobLevel`

---

## 🚀 État d’Avancement du Projet

### ✅ Phase 1 : Structuration des Données

Conformément aux bonnes pratiques en **Data Engineering**, les étapes suivantes ont été réalisées :

#### 🔹 Nettoyage
- Suppression des colonnes constantes :
  - `EmployeeCount`
  - `Over18`
  - `StandardHours`
- Suppression des colonnes non pertinentes

#### 🔹 Prétraitement
- Gestion des valeurs manquantes
- Suppression des doublons

#### 🔹 Feature Engineering
- Création de la variable `Revenu_Annuel`
- Encodage numérique de la variable `Attrition`

#### 🔹 Gestion des Outliers
- Détection et filtrage des valeurs aberrantes du salaire
- Méthode utilisée : **IQR (Interquartile Range)**

---

## 📊 Phase 2 : Visualisation des Données (Data Visualization)

### 📈 Visualisations avec Matplotlib
- **Histogramme** : Distribution des âges  
  _(Population majoritairement Junior–Senior)_
- **Diagramme en barres** : Répartition des employés par département  
  _(Dominance du département R&D)_
- **Scatter plot avec régression** : Corrélation entre l’expérience et le revenu
- **Graphique 3D** : Analyse combinée _(Âge, Ancienneté, Revenu)_

---

### 🎨 Visualisations Avancées avec Seaborn

#### 🔹 Scatterplot multivarié
- Identification des profils à risque de départ
- Départs concentrés chez :
  - Les jeunes employés
  - Les bas salaires

#### 🔹 Lineplot
- Analyse de l’équité salariale entre :
  - Hommes
  - Femmes

#### 🔹 Relplot (Faceting)
Comparaison des dynamiques salariales par :
- Département
- Niveau hiérarchique (`JobLevel`)

---

## 📊 Insights Clés (Résultats Préliminaires)

### 🔹 Profil des départs
Les employés les plus susceptibles de quitter l’entreprise sont :
- Jeunes
- Ayant une faible ancienneté
- Avec un salaire inférieur à la moyenne

### 🔹 Salaire
- La progression salariale dépend fortement du **JobLevel**
- L’ancienneté seule n’explique pas significativement le salaire

### 🔹 Genre
- Aucune disparité salariale significative observée entre hommes et femmes

---

## 🔜 Prochaines Étapes (Roadmap)

Le projet évoluera vers l’implémentation de modèles de **Machine Learning**, répartis en trois axes :

### 🔹 Classification – Prédiction de l’Attrition
- Régression Logistique
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (K-NN)

### 🔹 Clustering – Segmentation des Employés
- K-Means
- Clustering Hiérarchique

### 🔹 Régression – Prédiction du Salaire
- Régression Linéaire
- Séries Temporelles (ARIMA)

---


