# Feature Engineering AML — Plan d'implementation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Creer un notebook qui ajoute 7 features comportementales et compare deux modeles (Decision Tree et XGBoost) pour ameliorer le recall AML (baseline 43%).

**Architecture:** Un seul notebook Jupyter dans le repo `aml-feature-engineering`. Le notebook charge le dataset 800k, fait le split train/test AVANT le calcul des features (pas de data leakage), construit les features sur le train set, les applique au test set par lookup, puis entraine deux modeles (Decision Tree et XGBoost) et compare les resultats.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

---

## Structure des fichiers

| Action | Fichier | Responsabilite |
|---|---|---|
| Creer | `feature_engineering_aml.ipynb` | Notebook principal — tout le pipeline |
| Existant | `docs/specs/2026-05-07-feature-engineering-design.md` | Spec de reference |

---

### Task 1: Setup du notebook — imports et chargement des donnees

**Files:**
- Create: `feature_engineering_aml.ipynb`

- [ ] **Step 1: Creer le notebook avec la cellule markdown d'introduction**

```markdown
# Feature Engineering Comportemental — AML Detection

**Objectif :** Ameliorer le recall du Decision Tree (baseline 43%) en ajoutant des features comportementales.

**Methode :** Split train/test AVANT le calcul des features pour eviter le data leakage temporel.

**Spec :** voir `docs/specs/2026-05-07-feature-engineering-design.md`
```

- [ ] **Step 2: Cellule imports**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    recall_score, precision_score, f1_score
)
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
```

- [ ] **Step 3: Cellule chargement des donnees**

```python
import os

CSV_FULL = '../SAML-D_sample_800k.csv'
CSV_SAMPLE = '../projet_final_merged/data/SAML-D_sample_1k.csv'

if os.path.exists(CSV_FULL):
    csv_path = CSV_FULL
    print(f"Dataset complet : {csv_path}")
elif os.path.exists(CSV_SAMPLE):
    csv_path = CSV_SAMPLE
    print(f"Echantillon : {csv_path}")
else:
    raise FileNotFoundError("Aucun dataset trouve.")

df = pd.read_csv(csv_path)
print(f"Shape : {df.shape}")
print(f"Transactions suspectes : {df['Is_laundering'].sum()} ({df['Is_laundering'].mean()*100:.4f}%)")
```

- [ ] **Step 4: Executer et verifier**

Run: executer les 3 cellules
Expected: `Shape : (800000, 12)` et `Transactions suspectes : 833 (0.1041%)`

---

### Task 2: Modele baseline — Decision Tree avec les 6 features originales

**Files:**
- Modify: `feature_engineering_aml.ipynb`

- [ ] **Step 1: Cellule markdown**

```markdown
## 1. Modele Baseline — Decision Tree (6 features originales)

Reproduction du modele du projet initial pour avoir une reference de comparaison.
```

- [ ] **Step 2: Cellule baseline**

```python
# --- Baseline : meme modele que le projet initial ---
df_baseline = df.copy()

# Encoding categoriel
label_encoders = {}
cat_cols = ['Payment_type', 'Payment_currency', 'Received_currency',
            'Sender_bank_location', 'Receiver_bank_location']
for col in cat_cols:
    le = LabelEncoder()
    df_baseline[col] = le.fit_transform(df_baseline[col])
    label_encoders[col] = le

# Features et target
baseline_features = ['Amount', 'Payment_type', 'Payment_currency',
                     'Received_currency', 'Sender_bank_location',
                     'Receiver_bank_location']
X_base = df_baseline[baseline_features]
y = df_baseline['Is_laundering']

# Split
X_train_base, X_test_base, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Entrainement
tree_baseline = DecisionTreeClassifier(
    max_depth=10, class_weight='balanced', random_state=RANDOM_STATE
)
tree_baseline.fit(X_train_base, y_train)

# Evaluation
y_pred_base = tree_baseline.predict(X_test_base)
recall_baseline = recall_score(y_test, y_pred_base)
print(f"=== BASELINE ===")
print(f"Recall test : {recall_baseline:.3f}")
print(f"\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred_base))
print(f"\n{classification_report(y_test, y_pred_base, zero_division=0)}")
```

- [ ] **Step 3: Executer et verifier**

Run: executer la cellule
Expected: Recall test ~ 0.425 (43%), matrice de confusion avec TP=71, FN=96

---

### Task 3: Split train/test et construction des features sur le train

**Files:**
- Modify: `feature_engineering_aml.ipynb`

- [ ] **Step 1: Cellule markdown**

```markdown
## 2. Feature Engineering — Construction sur le train set uniquement

Les features sont calculees sur le train set (80%) puis appliquees au test set par lookup.
Cela evite le data leakage temporel (le modele ne voit pas d'informations futures).

### Features creees :
1. `sender_tx_count` / `receiver_tx_count` — frequence d'activite du compte
2. `is_sender_risky_country` / `is_receiver_risky_country` — pays a risque
3. `receiver_smurfing_score` — pattern smurfing (petit montant + beaucoup d'expediteurs)
4. `sender_risky_payment_count` / `receiver_risky_payment_count` — nombre de paiements a risque
```

- [ ] **Step 2: Cellule split train/test AVANT feature engineering**

```python
# --- Split AVANT le calcul des features ---
# On garde les index pour pouvoir calculer les features sur le train uniquement

df_work = df.copy()

# On a besoin des colonnes originales (non encodees) pour le feature engineering
# Le split se fait sur les index
train_idx, test_idx = train_test_split(
    df_work.index, test_size=0.2, random_state=RANDOM_STATE,
    stratify=df_work['Is_laundering']
)

df_train = df_work.loc[train_idx].copy()
df_test = df_work.loc[test_idx].copy()

print(f"Train : {len(df_train)} lignes ({df_train['Is_laundering'].sum()} suspectes)")
print(f"Test  : {len(df_test)} lignes ({df_test['Is_laundering'].sum()} suspectes)")
```

- [ ] **Step 3: Executer et verifier**

Run: executer la cellule
Expected: Train ~640000 lignes (666 suspectes), Test ~160000 lignes (167 suspectes)

---

### Task 4: Feature 1 — Frequence (sender_tx_count, receiver_tx_count)

**Files:**
- Modify: `feature_engineering_aml.ipynb`

- [ ] **Step 1: Cellule calcul frequence**

```python
# --- Feature 1 : Frequence ---
# Calculer le nombre de transactions par compte sur le TRAIN uniquement
sender_counts = df_train.groupby('Sender_account').size()
receiver_counts = df_train.groupby('Receiver_account').size()

# Appliquer au train
df_train['sender_tx_count'] = df_train['Sender_account'].map(sender_counts)
df_train['receiver_tx_count'] = df_train['Receiver_account'].map(receiver_counts)

# Appliquer au test par lookup (comptes absents = 0)
df_test['sender_tx_count'] = df_test['Sender_account'].map(sender_counts).fillna(0).astype(int)
df_test['receiver_tx_count'] = df_test['Receiver_account'].map(receiver_counts).fillna(0).astype(int)

# Verification
print("=== sender_tx_count ===")
print(f"Train — mean: {df_train['sender_tx_count'].mean():.1f}, max: {df_train['sender_tx_count'].max()}")
print(f"Test  — mean: {df_test['sender_tx_count'].mean():.1f}, max: {df_test['sender_tx_count'].max()}")
print(f"Test  — comptes absents du train : {(df_test['sender_tx_count'] == 0).sum()}")
```

- [ ] **Step 2: Executer et verifier**

Run: executer la cellule
Expected: moyennes similaires train/test, quelques comptes absents du train dans le test

---

### Task 5: Feature 2 — Geographique (is_sender_risky_country, is_receiver_risky_country)

**Files:**
- Modify: `feature_engineering_aml.ipynb`

- [ ] **Step 1: Cellule pays a risque**

```python
# --- Feature 2 : Pays a risque ---
# Pays identifies dans l'EDA (seuils bases sur le taux global de 0.104%)
SENDER_RISKY = {'Albania', 'Italy', 'Netherlands'}       # seuil 3x (> 0.31%)
RECEIVER_RISKY = {'Nigeria', 'Albania', 'Morocco', 'Mexico'}  # seuil 5x (> 0.52%)

# Appliquer au train et test (pas de leakage, ce sont des constantes metier)
for dataset in [df_train, df_test]:
    dataset['is_sender_risky_country'] = dataset['Sender_bank_location'].isin(SENDER_RISKY).astype(int)
    dataset['is_receiver_risky_country'] = dataset['Receiver_bank_location'].isin(RECEIVER_RISKY).astype(int)

# Verification
print("=== Pays a risque ===")
print(f"Train — sender risky : {df_train['is_sender_risky_country'].sum()} ({df_train['is_sender_risky_country'].mean()*100:.2f}%)")
print(f"Train — receiver risky : {df_train['is_receiver_risky_country'].sum()} ({df_train['is_receiver_risky_country'].mean()*100:.2f}%)")
print(f"\nTaux suspect parmi sender risky (train) :")
print(df_train.groupby('is_sender_risky_country')['Is_laundering'].mean() * 100)
```

- [ ] **Step 2: Executer et verifier**

Run: executer la cellule
Expected: le taux suspect parmi sender risky > 0.31%, parmi non-risky ~ 0.10%

---

### Task 6: Feature 3 — Smurfing score

**Files:**
- Modify: `feature_engineering_aml.ipynb`

- [ ] **Step 1: Cellule smurfing score**

```python
# --- Feature 3 : Smurfing Score ---
# Smurfing = petit montant (< 5K) + beaucoup d'expediteurs vers un meme destinataire
# Taux suspect : 0.644% (6x la moyenne) vs 0.042% pour petits montants + peu d'expediteurs

# Etape 1 : identifier les transactions "petit montant" dans le train
small_amount_train = df_train[df_train['Amount'] < 5000]

# Etape 2 : pour chaque receiver, compter le nb d'expediteurs uniques (petits montants seulement)
receiver_unique_senders = small_amount_train.groupby('Receiver_account')['Sender_account'].nunique()
median_senders = receiver_unique_senders.median()
print(f"Mediane du nb d'expediteurs uniques (petits montants) : {median_senders}")

# Etape 3 : pour chaque receiver, compter le nb de transactions "smurfing"
# = transactions recues avec montant < 5K ET provenant de plus d'expediteurs que la mediane
smurfing_receivers = set(receiver_unique_senders[receiver_unique_senders > median_senders].index)

def calc_smurfing_score(dataset, smurfing_receivers_set):
    """Compte le nb de transactions smurfing recues par chaque receiver dans le dataset."""
    is_smurfing_tx = (dataset['Amount'] < 5000) & (dataset['Receiver_account'].isin(smurfing_receivers_set))
    smurfing_counts = is_smurfing_tx.groupby(dataset['Receiver_account']).sum()
    return smurfing_counts

smurfing_scores_train = calc_smurfing_score(df_train, smurfing_receivers)

# Appliquer au train et test
df_train['receiver_smurfing_score'] = df_train['Receiver_account'].map(smurfing_scores_train).fillna(0).astype(int)
df_test['receiver_smurfing_score'] = df_test['Receiver_account'].map(smurfing_scores_train).fillna(0).astype(int)

# Verification
print(f"\n=== Smurfing Score ===")
print(f"Train — mean: {df_train['receiver_smurfing_score'].mean():.2f}, max: {df_train['receiver_smurfing_score'].max()}")
print(f"Train — % avec score > 0 : {(df_train['receiver_smurfing_score'] > 0).mean()*100:.1f}%")
print(f"\nTaux suspect par smurfing score (train) :")
print(df_train.groupby(df_train['receiver_smurfing_score'] > 0)['Is_laundering'].mean() * 100)
```

- [ ] **Step 2: Executer et verifier**

Run: executer la cellule
Expected: le taux suspect parmi smurfing_score > 0 devrait etre nettement superieur a la moyenne (0.104%)

---

### Task 7: Feature 4 — Risky payment count

**Files:**
- Modify: `feature_engineering_aml.ipynb`

- [ ] **Step 1: Cellule risky payment count**

```python
# --- Feature 4 : Risky Payment Count ---
# "Transaction a risque" = Payment_type in {Cash Deposit, Cash Withdrawal, Cross-border}
# Taux suspects > 2x la moyenne (0.63%, 0.47%, 0.28% vs 0.104%)

RISKY_PAYMENT_TYPES = {'Cash Deposit', 'Cash Withdrawal', 'Cross-border'}

# Marquer les transactions a risque dans le train
df_train['is_risky_payment'] = df_train['Payment_type'].isin(RISKY_PAYMENT_TYPES).astype(int)

# Compter le nb de paiements a risque par compte (sur le train uniquement)
sender_risky_counts = df_train.groupby('Sender_account')['is_risky_payment'].sum()
receiver_risky_counts = df_train.groupby('Receiver_account')['is_risky_payment'].sum()

# Appliquer au train
df_train['sender_risky_payment_count'] = df_train['Sender_account'].map(sender_risky_counts)
df_train['receiver_risky_payment_count'] = df_train['Receiver_account'].map(receiver_risky_counts)

# Appliquer au test par lookup (comptes absents = 0)
df_test['sender_risky_payment_count'] = df_test['Sender_account'].map(sender_risky_counts).fillna(0).astype(int)
df_test['receiver_risky_payment_count'] = df_test['Receiver_account'].map(receiver_risky_counts).fillna(0).astype(int)

# Nettoyage colonne temporaire
df_train.drop('is_risky_payment', axis=1, inplace=True)

# Verification
print("=== Risky Payment Count ===")
print(f"Train — sender mean: {df_train['sender_risky_payment_count'].mean():.2f}")
print(f"Train — receiver mean: {df_train['receiver_risky_payment_count'].mean():.2f}")
print(f"\nTaux suspect par sender_risky_payment_count > 0 (train) :")
print(df_train.groupby(df_train['sender_risky_payment_count'] > 0)['Is_laundering'].mean() * 100)
```

- [ ] **Step 2: Executer et verifier**

Run: executer la cellule
Expected: le taux suspect parmi risky_payment_count > 0 devrait etre superieur a la moyenne

---

### Task 8: Encoding et preparation des features

**Files:**
- Modify: `feature_engineering_aml.ipynb`

- [ ] **Step 1: Cellule markdown**

```markdown
## 3. Modeles ameliores — Decision Tree et XGBoost

Deux modeles entraines avec les 13 features (6 originales + 7 nouvelles) pour comparer :
- **Decision Tree** (max_depth=10) — comparaison directe avec le baseline
- **XGBoost** — modele ensembliste performant sur donnees tabulaires desequilibrees
```

- [ ] **Step 2: Cellule encoding et preparation**

```python
# --- Encoding des colonnes categorielles ---
cat_cols = ['Payment_type', 'Payment_currency', 'Received_currency',
            'Sender_bank_location', 'Receiver_bank_location']

for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([df_train[col], df_test[col]]))
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

# --- Features ---
new_features = ['sender_tx_count', 'receiver_tx_count',
                'is_sender_risky_country', 'is_receiver_risky_country',
                'receiver_smurfing_score',
                'sender_risky_payment_count', 'receiver_risky_payment_count']

all_features = baseline_features + new_features

X_train_new = df_train[all_features]
X_test_new = df_test[all_features]
y_train_new = df_train['Is_laundering']
y_test_new = df_test['Is_laundering']

print(f"Features utilisees ({len(all_features)}) : {all_features}")
print(f"X_train shape : {X_train_new.shape}")
print(f"X_test shape  : {X_test_new.shape}")
```

- [ ] **Step 3: Executer et verifier**

Run: executer la cellule
Expected: 13 features, X_train ~640000 lignes

---

### Task 9: Modele 1 — Decision Tree ameliore

**Files:**
- Modify: `feature_engineering_aml.ipynb`

- [ ] **Step 1: Cellule Decision Tree**

```python
# --- Decision Tree avec nouvelles features ---
tree_new = DecisionTreeClassifier(
    max_depth=10, class_weight='balanced', random_state=RANDOM_STATE
)
tree_new.fit(X_train_new, y_train_new)

y_pred_tree = tree_new.predict(X_test_new)
recall_tree = recall_score(y_test_new, y_pred_tree)

print(f"=== DECISION TREE (13 features) ===")
print(f"Recall BASELINE : {recall_baseline:.3f}")
print(f"Recall NOUVEAU  : {recall_tree:.3f}")
print(f"Amelioration    : {(recall_tree - recall_baseline)*100:+.1f} points")
print(f"\nMatrice de confusion :")
print(confusion_matrix(y_test_new, y_pred_tree))
print(f"\n{classification_report(y_test_new, y_pred_tree, zero_division=0)}")
```

- [ ] **Step 2: Executer et verifier**

Run: executer la cellule
Expected: Recall > 0.43 (amelioration par rapport au baseline)

---

### Task 10: Modele 2 — XGBoost

**Files:**
- Modify: `feature_engineering_aml.ipynb`

- [ ] **Step 1: Cellule XGBoost**

```python
# --- XGBoost avec nouvelles features ---
# scale_pos_weight compense le desequilibre : nb_negatifs / nb_positifs
n_neg = (y_train_new == 0).sum()
n_pos = (y_train_new == 1).sum()

xgb = XGBClassifier(
    max_depth=6,
    n_estimators=200,
    learning_rate=0.1,
    scale_pos_weight=n_neg / n_pos,
    random_state=RANDOM_STATE,
    eval_metric='aucpr',
    use_label_encoder=False
)
xgb.fit(X_train_new, y_train_new)

y_pred_xgb = xgb.predict(X_test_new)
recall_xgb = recall_score(y_test_new, y_pred_xgb)

print(f"=== XGBOOST (13 features) ===")
print(f"Recall BASELINE      : {recall_baseline:.3f}")
print(f"Recall Decision Tree : {recall_tree:.3f}")
print(f"Recall XGBoost       : {recall_xgb:.3f}")
print(f"\nMatrice de confusion :")
print(confusion_matrix(y_test_new, y_pred_xgb))
print(f"\n{classification_report(y_test_new, y_pred_xgb, zero_division=0)}")
```

- [ ] **Step 2: Executer et verifier**

Run: executer la cellule
Expected: Recall comparable ou superieur au Decision Tree

---

### Task 11: Comparaison train vs test et importance des features

**Files:**
- Modify: `feature_engineering_aml.ipynb`

- [ ] **Step 1: Cellule markdown**

```markdown
## 4. Analyse des resultats
```

- [ ] **Step 2: Cellule comparaison train vs test (les deux modeles)**

```python
# --- Comparaison Train vs Test (sur-apprentissage) ---
y_train_pred_tree = tree_new.predict(X_train_new)
y_train_pred_xgb = xgb.predict(X_train_new)

print("=== Decision Tree — Train vs Test ===")
for name, train_val, test_val in [
    ("Recall", recall_score(y_train_new, y_train_pred_tree), recall_score(y_test_new, y_pred_tree)),
    ("Precision", precision_score(y_train_new, y_train_pred_tree, zero_division=0), precision_score(y_test_new, y_pred_tree, zero_division=0)),
    ("F1", f1_score(y_train_new, y_train_pred_tree), f1_score(y_test_new, y_pred_tree)),
]:
    print(f"  {name:10s} train={train_val:.3f}  | test={test_val:.3f}")

print(f"\n=== XGBoost — Train vs Test ===")
for name, train_val, test_val in [
    ("Recall", recall_score(y_train_new, y_train_pred_xgb), recall_score(y_test_new, y_pred_xgb)),
    ("Precision", precision_score(y_train_new, y_train_pred_xgb, zero_division=0), precision_score(y_test_new, y_pred_xgb, zero_division=0)),
    ("F1", f1_score(y_train_new, y_train_pred_xgb), f1_score(y_test_new, y_pred_xgb)),
]:
    print(f"  {name:10s} train={train_val:.3f}  | test={test_val:.3f}")
```

- [ ] **Step 3: Cellule importance des features (les deux modeles)**

```python
# --- Importance des features ---
importance_tree = pd.Series(tree_new.feature_importances_, index=all_features).sort_values(ascending=False)
importance_xgb = pd.Series(xgb.feature_importances_, index=all_features).sort_values(ascending=False)

print("=== Importance — Decision Tree ===")
for feat, imp in importance_tree.items():
    marker = " <-- NEW" if feat in new_features else ""
    print(f"  {feat:35s} {imp*100:5.1f}%{marker}")

print(f"\n=== Importance — XGBoost ===")
for feat, imp in importance_xgb.items():
    marker = " <-- NEW" if feat in new_features else ""
    print(f"  {feat:35s} {imp*100:5.1f}%{marker}")
```

- [ ] **Step 4: Executer et verifier**

Run: executer les 2 cellules
Expected: les nouvelles features apparaissent dans le classement d'importance des deux modeles

---

### Task 12: Visualisations

**Files:**
- Modify: `feature_engineering_aml.ipynb`

- [ ] **Step 1: Cellule matrice de confusion comparative (3 modeles)**

```python
# --- Visualisation : Matrices de confusion — Baseline vs DT vs XGBoost ---
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

for ax, y_true, y_pred, title in [
    (axes[0], y_test, y_pred_base, "Baseline\n(6 features)"),
    (axes[1], y_test_new, y_pred_tree, "Decision Tree\n(13 features)"),
    (axes[2], y_test_new, y_pred_xgb, "XGBoost\n(13 features)")
]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall_val = tp / (tp + fn)
    annot = np.array([
        [f"TN\n{tn:,}", f"FP\n{fp:,}"],
        [f"FN\n{fn}", f"TP\n{tp}"]
    ])
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=["Predit Normal", "Predit Suspect"],
                yticklabels=["Reel Normal", "Reel Suspect"],
                annot_kws={"fontsize": 12, "fontweight": "bold"}, ax=ax)
    ax.set_title(f"{title}\nRecall = {recall_val*100:.1f}%",
                 fontsize=12, fontweight="bold")

plt.suptitle("Comparaison des 3 modeles",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Cellule importance des features (2 modeles cote a cote)**

```python
# --- Visualisation : Importance des features — DT vs XGBoost ---
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, importance, title in [
    (axes[0], importance_tree, "Decision Tree"),
    (axes[1], importance_xgb, "XGBoost")
]:
    imp_sorted = importance.sort_values()
    colors = ['#E74C3C' if f in new_features else '#2E86C1' for f in imp_sorted.index]
    ax.barh(imp_sorted.index, imp_sorted.values * 100, color=colors)
    for i, (feat, val) in enumerate(imp_sorted.items()):
        ax.text(val * 100 + 0.3, i, f"{val*100:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Importance (%)")
    ax.set_title(f"{title}\nRouge = nouvelles features", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("Importance des features par modele", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Cellule recap comparatif**

```python
# --- Tableau recap ---
print("=" * 60)
print(f"{'Modele':<25} {'Recall':>8} {'Precision':>10} {'F1':>8}")
print("-" * 60)
print(f"{'Baseline (6 feat.)':<25} {recall_baseline:>8.3f} {precision_score(y_test, y_pred_base, zero_division=0):>10.4f} {f1_score(y_test, y_pred_base):>8.4f}")
print(f"{'Decision Tree (13 feat.)':<25} {recall_tree:>8.3f} {precision_score(y_test_new, y_pred_tree, zero_division=0):>10.4f} {f1_score(y_test_new, y_pred_tree):>8.4f}")
print(f"{'XGBoost (13 feat.)':<25} {recall_xgb:>8.3f} {precision_score(y_test_new, y_pred_xgb, zero_division=0):>10.4f} {f1_score(y_test_new, y_pred_xgb):>8.4f}")
print("=" * 60)
```

- [ ] **Step 4: Executer et verifier**

Run: executer les 3 cellules
Expected: 3 matrices de confusion, 2 bar charts d'importance, tableau recap

---

### Task 13: Validation croisee

**Files:**
- Modify: `feature_engineering_aml.ipynb`

- [ ] **Step 1: Cellule markdown**

```markdown
## 5. Validation croisee

Un seul split 80/20 peut donner un recall variable selon le hasard du split (167 cas positifs seulement dans le test).
La validation croisee 5-fold donne une estimation plus robuste.
```

- [ ] **Step 2: Cellule validation croisee**

```python
# --- Validation croisee 5-fold ---
# On recalcule les features sur tout le dataset pour la CV
# (sklearn gere le split interne, mais les features sont calculees globalement
# — c'est une approximation acceptable pour Jedha, cf. section Regard metier du spec)

df_cv = df.copy()

# Recalculer toutes les features sur tout le dataset
df_cv['sender_tx_count'] = df_cv.groupby('Sender_account')['Sender_account'].transform('count')
df_cv['receiver_tx_count'] = df_cv.groupby('Receiver_account')['Receiver_account'].transform('count')
df_cv['is_sender_risky_country'] = df_cv['Sender_bank_location'].isin(SENDER_RISKY).astype(int)
df_cv['is_receiver_risky_country'] = df_cv['Receiver_bank_location'].isin(RECEIVER_RISKY).astype(int)

# Smurfing score
small_cv = df_cv[df_cv['Amount'] < 5000]
recv_unique_cv = small_cv.groupby('Receiver_account')['Sender_account'].nunique()
median_cv = recv_unique_cv.median()
smurfing_recv_cv = set(recv_unique_cv[recv_unique_cv > median_cv].index)
is_smurf_cv = (df_cv['Amount'] < 5000) & (df_cv['Receiver_account'].isin(smurfing_recv_cv))
smurf_counts_cv = is_smurf_cv.groupby(df_cv['Receiver_account']).sum()
df_cv['receiver_smurfing_score'] = df_cv['Receiver_account'].map(smurf_counts_cv).fillna(0).astype(int)

# Risky payment count
df_cv['is_risky_pmt'] = df_cv['Payment_type'].isin(RISKY_PAYMENT_TYPES).astype(int)
df_cv['sender_risky_payment_count'] = df_cv.groupby('Sender_account')['is_risky_pmt'].transform('sum')
df_cv['receiver_risky_payment_count'] = df_cv.groupby('Receiver_account')['is_risky_pmt'].transform('sum')
df_cv.drop('is_risky_pmt', axis=1, inplace=True)

# Encoding
for col in cat_cols:
    df_cv[col] = LabelEncoder().fit_transform(df_cv[col])

X_cv = df_cv[all_features]
y_cv = df_cv['Is_laundering']

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
models_cv = {
    'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(max_depth=6, n_estimators=200, learning_rate=0.1,
                             scale_pos_weight=(y_cv == 0).sum() / (y_cv == 1).sum(),
                             random_state=RANDOM_STATE, eval_metric='aucpr', use_label_encoder=False)
}

print(f"=== Validation croisee 5-fold ===")
print(f"Baseline recall : {recall_baseline:.3f}\n")
for name, model in models_cv.items():
    scores = cross_val_score(model, X_cv, y_cv, cv=cv, scoring='recall')
    print(f"{name}:")
    print(f"  Recall par fold : {scores}")
    print(f"  Recall moyen    : {scores.mean():.3f} (+/- {scores.std():.3f})")
    print(f"  Amelioration    : {(scores.mean() - recall_baseline)*100:+.1f} points\n")
```

- [ ] **Step 3: Executer et verifier**

Run: executer la cellule
Expected: recall moyen > 0.43, ecart-type raisonnable

---

### Task 14: Conclusion et commit

**Files:**
- Modify: `feature_engineering_aml.ipynb`

- [ ] **Step 1: Cellule markdown conclusion**

```markdown
## 6. Conclusion

| Metrique | Baseline (6 feat.) | Decision Tree (13 feat.) | XGBoost (13 feat.) |
|---|---|---|---|
| Recall test | 43% | XX% | XX% |
| Recall CV (5-fold) | — | XX% | XX% |

### Features les plus importantes (nouvelles) :
- [a remplir apres execution]

### Limites :
- **Data leakage temporel** : la validation croisee utilise les features calculees sur tout le dataset (approximation acceptable pour un projet Jedha). Les modeles principaux (Tasks 9-10) calculent les features sur le train uniquement.
```

- [ ] **Step 2: Commit**

```bash
cd "/Users/nash/Documents/jedha/projet final/aml-feature-engineering"
git add feature_engineering_aml.ipynb
git commit -m "Add feature engineering notebook with 7 behavioral features"
```

- [ ] **Step 3: Push**

```bash
git push origin main
```
