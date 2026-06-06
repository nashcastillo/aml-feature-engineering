# Correction du data leakage dans les listes "à risque" — Plan d'implémentation

> **Pour l'exécution :** ce plan est conçu pour une exécution interactive avec l'utilisateur dans Jupyter. Les modifications du notebook sont faites avec NotebookEdit ; l'utilisateur ré-exécute les cellules dans son Jupyter et partage les outputs.

**Goal :** Remplacer les trois listes hardcodées (`SENDER_RISKY`, `RECEIVER_RISKY`, `RISKY_PAYMENT_TYPES`) par un calcul fait sur le train uniquement, partout où elles sont utilisées dans le notebook.

**Architecture :** Une fonction unique `compute_risky_lists(df_train)` encapsule les règles métier (seuils 3x / 5x / 2x la moyenne globale du train). Elle est appelée (a) une fois sur le train principal en section 2, (b) à chaque fold dans `build_features_on_fold()` en section 5.

**Tech Stack :** Jupyter Notebook, pandas, scikit-learn, XGBoost. Notebook : `feature_engineering_aml.ipynb`. Spec : `docs/specs/2026-05-09-fix-risky-lists-leakage-design.md`.

**Fichier modifié :** un seul fichier — `feature_engineering_aml.ipynb` :
- 1 nouvelle cellule insérée (fonction + tests inline)
- Cellule `0e8b08a1` modifiée (Pays à risque)
- Cellule `49469cdc` modifiée (Risky Payment Count)
- Cellule `jnb2sya219s` modifiée (build_features_on_fold + boucle CV)

---

### Task 1 : Définir `compute_risky_lists()` avec tests inline

**Files :**
- Modifier : `feature_engineering_aml.ipynb` (insérer 1 nouvelle cellule code juste avant la cellule `0e8b08a1`)

- [ ] **Step 1 : Insérer la nouvelle cellule via NotebookEdit**

Position : avant la cellule `0e8b08a1` (cellule "Pays à risque").

Contenu de la cellule :

```python
# --- Fonction reutilisable : calcul des listes "a risque" sur le train uniquement ---
# Remplace les anciennes listes hardcodees (issues d'une EDA full-dataset = data leakage).
# Memes regles metier : seuils en multiples de la moyenne globale du train.

def compute_risky_lists(df_train,
                        sender_threshold_x=3,
                        receiver_threshold_x=5,
                        payment_threshold_x=2):
    """
    Identifie les modalites a risque a partir du train uniquement.
    Retourne 3 sets : (sender_risky_countries, receiver_risky_countries, risky_payment_types).
    """
    global_rate = df_train['Is_laundering'].mean()

    sender_rates = df_train.groupby('Sender_bank_location')['Is_laundering'].mean()
    sender_risky = set(sender_rates[sender_rates > sender_threshold_x * global_rate].index)

    receiver_rates = df_train.groupby('Receiver_bank_location')['Is_laundering'].mean()
    receiver_risky = set(receiver_rates[receiver_rates > receiver_threshold_x * global_rate].index)

    payment_rates = df_train.groupby('Payment_type')['Is_laundering'].mean()
    risky_payments = set(payment_rates[payment_rates > payment_threshold_x * global_rate].index)

    return sender_risky, receiver_risky, risky_payments


# --- Tests inline : verifier le contrat de la fonction ---
_s, _r, _p = compute_risky_lists(df_train)
assert isinstance(_s, set) and isinstance(_r, set) and isinstance(_p, set), "Doit retourner 3 sets"
assert len(_s) > 0 and len(_r) > 0 and len(_p) > 0, "Aucune liste ne doit etre vide sur le train principal"

# Monotonie des seuils : seuil plus bas => plus d'elements
_s_loose, _, _ = compute_risky_lists(df_train, sender_threshold_x=1)
_s_strict, _, _ = compute_risky_lists(df_train, sender_threshold_x=10)
assert len(_s_loose) >= len(_s) >= len(_s_strict), "Monotonie des seuils violee"

# Pas d'effet de bord sur df_train
_cols_avant = set(df_train.columns)
compute_risky_lists(df_train)
assert set(df_train.columns) == _cols_avant, "compute_risky_lists ne doit pas modifier df_train"

print("compute_risky_lists() : 3 assertions passees")
del _s, _r, _p, _s_loose, _s_strict, _cols_avant
```

- [ ] **Step 2 : Demander à l'utilisateur d'exécuter la nouvelle cellule dans Jupyter**

Output attendu :
```
compute_risky_lists() : 3 assertions passees
```

Si une assertion échoue → arrêt, diagnostic.

- [ ] **Step 3 : Validation**

L'utilisateur partage l'output. Confirmer que le print est apparu sans AssertionError.

---

### Task 2 : Remplacer les listes hardcodées dans la cellule "Pays à risque"

**Files :**
- Modifier : `feature_engineering_aml.ipynb`, cellule `0e8b08a1`

- [ ] **Step 1 : Remplacer le contenu de la cellule `0e8b08a1` via NotebookEdit**

Nouveau contenu :

```python
# --- Feature 2 : Pays a risque ---
# Listes calculees sur le train uniquement (anti-leakage).
# Memes regles que dans le spec d'origine : seuils 3x / 5x / 2x la moyenne globale.

SENDER_RISKY, RECEIVER_RISKY, RISKY_PAYMENT_TYPES = compute_risky_lists(df_train)

print(f"Sender risques (calcules sur le train) : {sorted(SENDER_RISKY)}")
print(f"Receiver risques (calcules sur le train) : {sorted(RECEIVER_RISKY)}")
print(f"Types de paiement risques : {sorted(RISKY_PAYMENT_TYPES)}")

# Comparaison avec les listes hardcodees d'origine (issues de l'EDA full-dataset)
_expected_sender = {'Albania', 'Italy', 'Netherlands'}
_expected_receiver = {'Nigeria', 'Albania', 'Morocco', 'Mexico'}
_expected_payments = {'Cash Deposit', 'Cash Withdrawal', 'Cross-border'}
print(f"\nDiff sender   : ajoutes={SENDER_RISKY - _expected_sender}, retires={_expected_sender - SENDER_RISKY}")
print(f"Diff receiver : ajoutes={RECEIVER_RISKY - _expected_receiver}, retires={_expected_receiver - RECEIVER_RISKY}")
print(f"Diff payments : ajoutes={RISKY_PAYMENT_TYPES - _expected_payments}, retires={_expected_payments - RISKY_PAYMENT_TYPES}")

# Appliquer au train et test
for dataset in [df_train, df_test]:
    dataset['is_sender_risky_country'] = dataset['Sender_bank_location'].isin(SENDER_RISKY).astype(int)
    dataset['is_receiver_risky_country'] = dataset['Receiver_bank_location'].isin(RECEIVER_RISKY).astype(int)

# Verification
print("\n=== Pays a risque ===")
print(f"Train - sender risky : {df_train['is_sender_risky_country'].sum()} ({df_train['is_sender_risky_country'].mean()*100:.2f}%)")
print(f"Train - receiver risky : {df_train['is_receiver_risky_country'].sum()} ({df_train['is_receiver_risky_country'].mean()*100:.2f}%)")
print(f"\nTaux suspect parmi sender risky (train) :")
print(df_train.groupby('is_sender_risky_country')['Is_laundering'].mean() * 100)
```

- [ ] **Step 2 : Exécuter la cellule dans Jupyter**

Output attendu : prints des listes calculées + diff par rapport aux listes d'origine + statistiques inchangées (le format est le même qu'avant).

- [ ] **Step 3 : Validation**

Vérifier deux choses dans l'output partagé par l'utilisateur :

1. Les listes "Sender risques" / "Receiver risques" / "Types de paiement risques" sont non vides.
2. Les diffs sont **petits** (idéalement vides, ou 1-2 modalités d'écart). Si la diff est massive (ex : 10 pays différents), c'est un signal qu'il faut investiguer.

Documenter les listes finales — livrable du rapport.

---

### Task 3 : Supprimer la définition redondante dans "Risky Payment Count"

**Files :**
- Modifier : `feature_engineering_aml.ipynb`, cellule `49469cdc`

- [ ] **Step 1 : Modifier le contenu de la cellule `49469cdc` via NotebookEdit**

Une seule ligne à supprimer (la ligne `RISKY_PAYMENT_TYPES = {...}`). Le reste de la cellule est inchangé. Nouveau contenu complet :

```python
# --- Feature 4 : Risky Payment Count ---
# RISKY_PAYMENT_TYPES est deja calcule en section "Pays a risque" via compute_risky_lists().

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
print(f"Train - sender mean: {df_train['sender_risky_payment_count'].mean():.2f}")
print(f"Train - receiver mean: {df_train['receiver_risky_payment_count'].mean():.2f}")
print(f"\nTaux suspect par sender_risky_payment_count > 0 (train) :")
print(df_train.groupby(df_train['sender_risky_payment_count'] > 0)['Is_laundering'].mean() * 100)
```

- [ ] **Step 2 : Exécuter la cellule dans Jupyter**

Output attendu : identique à l'exécution précédente (mêmes statistiques mean / taux suspect). Si les listes calculées en task 2 sont identiques aux listes d'origine, les chiffres seront strictement identiques.

- [ ] **Step 3 : Validation**

Pas d'erreur `NameError`. Statistiques cohérentes.

---

### Task 4 : Re-exécuter les modèles de la section 3 — vérification du recall

**Files :**
- `feature_engineering_aml.ipynb` — exécution sans modification des cellules :
  - `c1ottfim8gg` (encoding + features)
  - `at7ur7ayhkn` (Decision Tree 13 features)
  - `j8lpp7zrem` (XGBoost)
  - `31tddinq14b` (Random Forest)

- [ ] **Step 1 : Demander à l'utilisateur d'exécuter ces 4 cellules dans l'ordre**

- [ ] **Step 2 : Récupérer les recall test des 3 modèles**

Output attendu (référence avant correction) :
- Decision Tree : recall test = 0.635
- Random Forest : recall test = 0.575
- XGBoost : recall test = 0.527

- [ ] **Step 3 : Valider le critère de succès du spec (section 7)**

Le recall test du Decision Tree doit rester dans `[0.605, 0.665]` (±3 points autour de 63.5%).

Cas possibles :
- **Recall identique ou très proche** → les listes calculées sont identiques aux listes hardcodées. La critique du leakage était méthodologique, pas quantitative. C'est bien : le code est rigoureux, et la perf est inchangée.
- **Recall en baisse de 1-3 points** → léger overfit du choix de listes sur le test set d'origine. Acceptable.
- **Recall en chute > 3 points** → les anciennes listes contenaient des "trouvailles" du test. Confirmer la critique du leakage en investigant la cellule `0e8b08a1`.
- **Recall en hausse** → bonus inattendu (le seuil de la nouvelle liste capture mieux). À documenter.

---

### Task 5 : Modifier `build_features_on_fold()` pour recalcul dynamique en CV

**Files :**
- Modifier : `feature_engineering_aml.ipynb`, cellule `jnb2sya219s`

- [ ] **Step 1 : Modifier le contenu de la cellule `jnb2sya219s` via NotebookEdit**

Cellule contenant à la fois la fonction `build_features_on_fold` ET la boucle CV. Nouveau contenu complet :

```python
# --- Validation croisee 5-fold PROPRE ---
# Les features (ET les listes risquees) sont recalculees a l'interieur de chaque fold
# pour eviter tout data leakage temporel.

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

def build_features_on_fold(df_fold_train, df_fold_test):
    """Calcule les 7 features sur le fold train et les applique au fold test par lookup.
    Recalcule aussi les listes risquees sur le fold train uniquement."""

    # 0. Recalcul des listes risquees sur le fold train (anti-leakage)
    sender_risky, receiver_risky, risky_payments = compute_risky_lists(df_fold_train)

    # 1. Frequence
    sender_counts = df_fold_train.groupby('Sender_account').size()
    receiver_counts = df_fold_train.groupby('Receiver_account').size()
    df_fold_train['sender_tx_count'] = df_fold_train['Sender_account'].map(sender_counts)
    df_fold_train['receiver_tx_count'] = df_fold_train['Receiver_account'].map(receiver_counts)
    df_fold_test['sender_tx_count'] = df_fold_test['Sender_account'].map(sender_counts).fillna(0).astype(int)
    df_fold_test['receiver_tx_count'] = df_fold_test['Receiver_account'].map(receiver_counts).fillna(0).astype(int)

    # 2. Geographique (utilise les listes locales du fold)
    for ds in [df_fold_train, df_fold_test]:
        ds['is_sender_risky_country'] = ds['Sender_bank_location'].isin(sender_risky).astype(int)
        ds['is_receiver_risky_country'] = ds['Receiver_bank_location'].isin(receiver_risky).astype(int)

    # 3. Smurfing score (2 dimensions : petits montants + expediteurs uniques)
    small = df_fold_train[df_fold_train['Amount'] < 5000]
    recv_stats = small.groupby('Receiver_account').agg(
        n_senders=('Sender_account', 'nunique'),
        n_tx=('Amount', 'count')
    )
    recv_stats['smurfing_score'] = recv_stats['n_senders'] * recv_stats['n_tx']
    df_fold_train['receiver_smurfing_score'] = df_fold_train['Receiver_account'].map(recv_stats['smurfing_score']).fillna(0)
    df_fold_test['receiver_smurfing_score'] = df_fold_test['Receiver_account'].map(recv_stats['smurfing_score']).fillna(0)

    # 4. Risky payment count (utilise risky_payments local)
    df_fold_train['is_risky_pmt'] = df_fold_train['Payment_type'].isin(risky_payments).astype(int)
    s_risky = df_fold_train.groupby('Sender_account')['is_risky_pmt'].sum()
    r_risky = df_fold_train.groupby('Receiver_account')['is_risky_pmt'].sum()
    df_fold_train['sender_risky_payment_count'] = df_fold_train['Sender_account'].map(s_risky)
    df_fold_train['receiver_risky_payment_count'] = df_fold_train['Receiver_account'].map(r_risky)
    df_fold_test['sender_risky_payment_count'] = df_fold_test['Sender_account'].map(s_risky).fillna(0).astype(int)
    df_fold_test['receiver_risky_payment_count'] = df_fold_test['Receiver_account'].map(r_risky).fillna(0).astype(int)
    df_fold_train.drop('is_risky_pmt', axis=1, inplace=True)

    # Encoding avec OrdinalEncoder (fit uniquement sur train)
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    oe.fit(df_fold_train[cat_cols])
    df_fold_train[cat_cols] = oe.transform(df_fold_train[cat_cols])
    df_fold_test[cat_cols] = oe.transform(df_fold_test[cat_cols])

    return df_fold_train, df_fold_test


# --- CV manuelle ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
df_cv = df.copy()
y_cv = df_cv['Is_laundering']

results = {'Decision Tree': [], 'Random Forest': [], 'XGBoost': []}

for fold, (train_idx, test_idx) in enumerate(cv.split(df_cv, y_cv)):
    df_fold_train = df_cv.iloc[train_idx].copy()
    df_fold_test = df_cv.iloc[test_idx].copy()

    # Affichage des listes calculees pour ce fold (verification stabilite)
    _s, _r, _p = compute_risky_lists(df_fold_train)
    print(f"Fold {fold+1} listes risquees - sender={sorted(_s)}, receiver={sorted(_r)}, payments={sorted(_p)}")

    df_fold_train, df_fold_test = build_features_on_fold(df_fold_train, df_fold_test)

    X_tr = df_fold_train[all_features]
    X_te = df_fold_test[all_features]
    y_tr = df_fold_train['Is_laundering']
    y_te = df_fold_test['Is_laundering']

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=RANDOM_STATE)
    dt.fit(X_tr, y_tr)
    results['Decision Tree'].append(recall_score(y_te, dt.predict(X_te)))

    # Random Forest
    rf_cv = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced',
                                   n_jobs=-1, random_state=RANDOM_STATE)
    rf_cv.fit(X_tr, y_tr)
    results['Random Forest'].append(recall_score(y_te, rf_cv.predict(X_te)))

    # XGBoost
    n_neg = (y_tr == 0).sum()
    n_pos = (y_tr == 1).sum()
    xgb_cv = XGBClassifier(max_depth=6, n_estimators=200, learning_rate=0.1,
                           scale_pos_weight=n_neg / n_pos,
                           random_state=RANDOM_STATE, eval_metric='aucpr', use_label_encoder=False)
    xgb_cv.fit(X_tr, y_tr)
    results['XGBoost'].append(recall_score(y_te, xgb_cv.predict(X_te)))

    print(f"Fold {fold+1} - DT: {results['Decision Tree'][-1]:.3f}, RF: {results['Random Forest'][-1]:.3f}, XGB: {results['XGBoost'][-1]:.3f}\n")

print(f"=== Validation croisee 5-fold (features + listes recalculees par fold) ===")
print(f"Baseline recall : {recall_baseline:.3f}\n")
for name, scores in results.items():
    scores = np.array(scores)
    print(f"{name}:")
    print(f"  Recall par fold : {scores}")
    print(f"  Recall moyen    : {scores.mean():.3f} (+/- {scores.std():.3f})")
    print(f"  Amelioration    : {(scores.mean() - recall_baseline)*100:+.1f} points\n")
```

Différences avec l'ancienne version :
- Ligne ajoutée `sender_risky, receiver_risky, risky_payments = compute_risky_lists(df_fold_train)` au début de la fonction.
- `SENDER_RISKY` / `RECEIVER_RISKY` / `RISKY_PAYMENT_TYPES` (variables globales) → `sender_risky` / `receiver_risky` / `risky_payments` (variables locales du fold) dans les blocs features 2 et 4.
- Bloc d'affichage `compute_risky_lists` ajouté dans la boucle CV pour la vérification de stabilité.
- Reste strictement identique.

- [ ] **Step 2 : Validation syntaxique**

Pas d'exécution dans cette étape — on valide uniquement que la modification ne casse pas la cellule (pas d'erreur de parsing). NotebookEdit échouera si la cellule est mal formée.

---

### Task 6 : Re-exécuter la CV 5-fold pour vérifier la stabilité

**Files :**
- `feature_engineering_aml.ipynb` — exécution de la cellule `jnb2sya219s` modifiée à la task 5

- [ ] **Step 1 : Demander à l'utilisateur d'exécuter la cellule en Jupyter**

Durée estimée : 2 à 5 minutes (5 folds × 3 modèles × 640k lignes).

- [ ] **Step 2 : Vérifier la stabilité des listes par fold**

Output attendu : 5 lignes "Fold X listes risquees - ..." avec les listes calculées par fold.

Critère :
- Les pays / types qui apparaissent dans **tous les folds** sont stables → fiables.
- Les pays / types qui apparaissent dans **1 fold sur 5** sont du bruit statistique → noter pour V2 (seuil de volume minimum).

- [ ] **Step 3 : Comparer les recall CV à la baseline**

Output attendu (avant correction, indicatif — le notebook d'origine n'a pas exécuté cette section) :

Référence approximative attendue : recall CV moyen DT / RF / XGB autour de 0.55-0.65.

Critère de succès : aucun modèle ne s'effondre brutalement (chute > 10 points par rapport au split simple). Sinon → investigation.

---

### Task 7 (optionnelle) : Commit des modifications

- [ ] **Step 1 : Demander à l'utilisateur s'il veut committer**

Si oui :

```bash
git add docs/specs/2026-05-09-fix-risky-lists-leakage-design.md \
        docs/plans/2026-05-09-fix-risky-lists-leakage.md \
        feature_engineering_aml.ipynb
git commit -m "fix: remove data leakage from risky lists

Replace hardcoded SENDER_RISKY / RECEIVER_RISKY / RISKY_PAYMENT_TYPES
(derived from full-dataset EDA) with compute_risky_lists() called on
the train split only. Applied to both the main train/test split (section 2)
and the per-fold computation in build_features_on_fold (section 5).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Hors scope (V2 si nécessaire)

- Ajouter un seuil de volume minimum (ex : "au moins 100 transactions") dans `compute_risky_lists()` si la task 6 révèle des pays bruités.
- Analyse de sensibilité aux seuils 3x / 5x / 2x.
- Mise à jour du tuning d'hyperparamètres (sections 6/7) — sera traité dans le point suivant de la critique.

## Références

- Spec : `docs/specs/2026-05-09-fix-risky-lists-leakage-design.md`
- Spec parent : `docs/specs/2026-05-07-feature-engineering-design.md`
- Notebook : `feature_engineering_aml.ipynb`
