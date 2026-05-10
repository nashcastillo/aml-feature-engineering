# Correction du data leakage dans les listes "à risque"

**Date :** 2026-05-09
**Projet :** aml-feature-engineering
**Auteur :** Nashely Castillo
**Spec parent :** `2026-05-07-feature-engineering-design.md`

---

## 1. Problème

Le notebook `feature_engineering_aml.ipynb` définit trois listes de modalités "à risque" qui sont utilisées pour construire les features `is_sender_risky_country`, `is_receiver_risky_country` et `sender_risky_payment_count` / `receiver_risky_payment_count` :

```python
SENDER_RISKY = {'Albania', 'Italy', 'Netherlands'}
RECEIVER_RISKY = {'Nigeria', 'Albania', 'Morocco', 'Mexico'}
RISKY_PAYMENT_TYPES = {'Cash Deposit', 'Cash Withdrawal', 'Cross-border'}
```

Le commentaire associé indique « ce sont des constantes métier ». En réalité, ces listes ont été obtenues par EDA sur le **dataset complet** (800k lignes, test inclus), avec des seuils définis comme des multiples de la moyenne globale (3x pour sender, 5x pour receiver, 2x pour payment_type).

C'est un **data leakage** : les listes encodent indirectement de l'information sur le test set utilisé ensuite pour évaluer le modèle. Ce leakage existe à trois endroits :

1. Section 2 — split simple train/test utilisé pour la comparaison baseline vs nouveau modèle
2. Section 5 — validation croisée 5-fold
3. Section 6 — tuning d'hyperparamètres par CV

## 2. Objectif

Calculer ces trois listes **uniquement à partir du train**, dans tous les contextes où elles sont utilisées, sans modifier ni la définition métier (mêmes seuils 3x / 5x / 2x) ni la nature des features (binaire 0/1).

## 3. Solution — Approche A (recalcul dynamique)

### 3.1 Composant : fonction `compute_risky_lists()`

Une seule fonction encapsule les trois règles métier du spec d'origine :

```python
def compute_risky_lists(df_train,
                        sender_threshold_x=3,
                        receiver_threshold_x=5,
                        payment_threshold_x=2):
    """
    Identifie les modalités à risque à partir du train uniquement.
    Reproduit les règles du spec : seuils en multiples de la moyenne globale.

    Retourne 3 sets : (sender_risky_countries, receiver_risky_countries, risky_payment_types)
    """
    global_rate = df_train['Is_laundering'].mean()

    sender_rates = df_train.groupby('Sender_bank_location')['Is_laundering'].mean()
    sender_risky = set(sender_rates[sender_rates > sender_threshold_x * global_rate].index)

    receiver_rates = df_train.groupby('Receiver_bank_location')['Is_laundering'].mean()
    receiver_risky = set(receiver_rates[receiver_rates > receiver_threshold_x * global_rate].index)

    payment_rates = df_train.groupby('Payment_type')['Is_laundering'].mean()
    risky_payments = set(payment_rates[payment_rates > payment_threshold_x * global_rate].index)

    return sender_risky, receiver_risky, risky_payments
```

**Choix de design :**
- Sets en sortie (compatible avec `.isin()`, sémantique correcte).
- Seuils paramétrables avec valeurs par défaut alignées sur le spec d'origine.
- Aucun effet de bord (pas de modification de `df_train`).
- Pas de seuil de volume minimum (V2 si nécessaire — voir section 6).

### 3.2 Intégration

Quatre modifications dans le notebook :

**Modification 1** — Nouvelle cellule placée juste avant la section 2 (« Feature Engineering — Construction sur le train set uniquement ») contenant la définition de `compute_risky_lists()`.

**Modification 2** — Cellule "Pays à risque" (section 2) : remplacer les sets hardcodés par un appel à la fonction.

```python
# AVANT
SENDER_RISKY = {'Albania', 'Italy', 'Netherlands'}
RECEIVER_RISKY = {'Nigeria', 'Albania', 'Morocco', 'Mexico'}

# APRÈS
SENDER_RISKY, RECEIVER_RISKY, RISKY_PAYMENT_TYPES = compute_risky_lists(df_train)
print(f"Sender risqués (calculés sur le train) : {sorted(SENDER_RISKY)}")
print(f"Receiver risqués (calculés sur le train) : {sorted(RECEIVER_RISKY)}")
print(f"Types de paiement risqués : {sorted(RISKY_PAYMENT_TYPES)}")
```

**Modification 3** — Cellule "Risky Payment Count" (section 2) : supprimer la définition redondante `RISKY_PAYMENT_TYPES = {...}` (la variable est déjà disponible depuis la modification 2).

**Modification 4** — Fonction `build_features_on_fold()` (section 5) : recalculer les listes en début de fonction à partir du fold train uniquement.

```python
def build_features_on_fold(df_fold_train, df_fold_test):
    # NOUVEAU : recalcul des listes risquées sur le fold train uniquement
    sender_risky, receiver_risky, risky_payments = compute_risky_lists(df_fold_train)

    # ... features 1 (frequence) inchangée ...

    # Feature 2 : géographique — utilise les sets locaux du fold
    for ds in [df_fold_train, df_fold_test]:
        ds['is_sender_risky_country'] = ds['Sender_bank_location'].isin(sender_risky).astype(int)
        ds['is_receiver_risky_country'] = ds['Receiver_bank_location'].isin(receiver_risky).astype(int)

    # ... feature 3 (smurfing) inchangée ...

    # Feature 4 : risky payment count — utilise risky_payments local
    df_fold_train['is_risky_pmt'] = df_fold_train['Payment_type'].isin(risky_payments).astype(int)
    # ... reste inchangé ...
```

## 4. Vérifications

Trois checks ajoutés au notebook pour valider la correction.

### 4.1 Comparaison aux listes hardcodées d'origine

Juste après la modification 2 :

```python
expected_sender = {'Albania', 'Italy', 'Netherlands'}
expected_receiver = {'Nigeria', 'Albania', 'Morocco', 'Mexico'}
expected_payments = {'Cash Deposit', 'Cash Withdrawal', 'Cross-border'}

print(f"Diff sender   : ajoutés={SENDER_RISKY - expected_sender}, retirés={expected_sender - SENDER_RISKY}")
print(f"Diff receiver : ajoutés={RECEIVER_RISKY - expected_receiver}, retirés={expected_receiver - RECEIVER_RISKY}")
print(f"Diff payments : ajoutés={RISKY_PAYMENT_TYPES - expected_payments}, retirés={expected_payments - RISKY_PAYMENT_TYPES}")
```

L'écart attendu est faible (le train principal contient 80% du dataset, les pays à fort taux suspect le restent). Toute différence est documentée pour la soutenance.

### 4.2 Comparaison du recall avant/après

Le recall test du Decision Tree 13 features (actuellement 63.5%) ne devrait pas s'effondrer. Tolérance : ±3 points. Une chute brutale (ex : 50%) signalerait que les listes hardcodées contenaient des « trouvailles » du test set, ce qui validerait rétroactivement la critique du leakage.

### 4.3 Stabilité des listes en CV

Dans la boucle CV de la section 5, afficher les listes calculées par fold avant l'appel à `build_features_on_fold()` :

```python
for fold, (train_idx, test_idx) in enumerate(cv.split(df_cv, y_cv)):
    df_fold_train = df_cv.iloc[train_idx].copy()
    df_fold_test = df_cv.iloc[test_idx].copy()

    s, r, p = compute_risky_lists(df_fold_train)
    print(f"Fold {fold+1} — sender={sorted(s)}, receiver={sorted(r)}, payments={sorted(p)}")

    df_fold_train, df_fold_test = build_features_on_fold(df_fold_train, df_fold_test)
    # ... reste inchangé ...
```

On s'attend à des listes stables (les mêmes pays reviennent dans tous les folds). Un pays qui n'apparaît que dans un seul fold = bruit statistique potentiel.

## 5. Ce qui ne change pas

- Structure du notebook (sections 1 à 8 conservées)
- Les autres features (`sender_tx_count`, `receiver_tx_count`, `receiver_smurfing_score`)
- Le pipeline d'encodage (`OrdinalEncoder` fit sur le train uniquement)
- Les modèles (Decision Tree, Random Forest, XGBoost)
- Les hyperparamètres et la grille de tuning
- Les métriques (recall prioritaire, précision indicative, courbe PR)

## 6. Limites et travail futur

- **Pas de seuil de volume minimum.** Un pays avec très peu de transactions dans le train du fold pourrait apparaître dans la liste par hasard statistique. Si la vérification 4.3 révèle des pays présents dans seulement 1-2 folds sur 5, ajouter une contrainte du type « au moins 100 transactions dans le train ».
- **Les seuils 3x / 5x / 2x sont conservés tels quels.** Ils viennent du spec d'origine sans justification statistique forte. Une analyse de sensibilité (faire varier les seuils, comparer le recall) sortirait du périmètre de cette correction et fera l'objet d'un travail séparé si nécessaire.
- **Approche binaire conservée.** Une feature continue (target encoding du taux historique de blanchiment par pays) a été envisagée puis écartée pour préserver la lisibilité métier (contrainte « explicable en 1 phrase » du spec d'origine).

## 7. Critère de succès

- La fonction `compute_risky_lists()` est appelée dans la section 2 et dans `build_features_on_fold()` ; aucune autre constante hardcodée pour les listes.
- Les trois vérifications de la section 4 sont exécutées et leurs sorties documentées.
- Le recall test du Decision Tree 13 features reste dans une fenêtre de ±3 points par rapport à l'état initial (63.5%). Toute déviation hors de cette fenêtre est analysée et expliquée.
