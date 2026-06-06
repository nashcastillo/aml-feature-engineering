# Calibration honnête du seuil compliance — Plan d'implémentation

> **Pour les agents :** SOUS-COMPÉTENCE REQUISE : utiliser `superpowers:executing-plans` ou `superpowers:subagent-driven-development` pour implémenter ce plan tâche par tâche. Les étapes utilisent la syntaxe checkbox (`- [ ]`) pour le suivi.

**Objectif** : Éliminer le biais méthodologique actuel (seuil compliance calculé directement sur le test set). Calibrer le seuil 80% recall sur un validation set held-out tiré du train ; reporter le recall effectif obtenu sur le test.

**Architecture** : Split temporel à 3 segments — `train_inner` (80% du train, ~204 j), `val` (20% du train, ~51 j), `test` (inchangé, ~65 j). Toutes les features et le modèle final tournent sur `train_inner`. Le seuil 80% recall est figé sur `val`, puis appliqué sur `test` avec mesure du recall effectif.

**Tech Stack** : Python 3.11, pandas, scikit-learn, LightGBM, PyTorch, NetworkX.

**Spec source** : `docs/specs/2026-06-06-honest-threshold-calibration-design.md`

**Positions actuelles des cellules code clés** (après l'audit pédagogique 9f628e7) :

| Cell | Contenu |
|---|---|
| 4 | Split temporel train/test |
| 7 | Feature 1 : frequence + cold start |
| 8 | `compute_risky_lists()` |
| 9 | Feature 2 : pays risqué |
| 10 | Smurfing + fan-out |
| 11 | Risky payment count |
| 12 | Graph features (PageRank) |
| 46 | Autoencoder PyTorch |
| 48 | LGBM+AE final |
| 57 | Baseline rule-based |
| 61 | Récap final ML vs RB |
| 63 | Stratification par typologie |

---

## Task 1 : Split temporel à 3 segments (cell 4)

**Fichier** : `feature_engineering_aml.ipynb` cell 4

**Pourquoi** : créer le segment `val` strictement chronologique entre `train_inner` et `test`, sans gap temporel.

- [ ] **Étape 1.1 : Lire la cell 4 actuelle**

Récupérer le code existant pour identifier les noms exacts de `df_train` / `df_test` et la logique de split.

- [ ] **Étape 1.2 : Remplacer la cell 4 par le nouveau split à 3 segments**

Code à insérer en complément du split existant (garder `df_train` / `df_test` comme variables intermédiaires, ajouter `df_train_inner` et `df_val`) :

```python
# --- Split temporel AVANT le calcul des features ---
# Tri chronologique strict pour respecter l'anti-leakage temporel
df_sorted = df.sort_values(['Date', 'Time']).reset_index(drop=True)

# Split 80/20 train/test (inchangé : conserve le benchmark historique)
split_idx = int(len(df_sorted) * 0.80)
df_train = df_sorted.iloc[:split_idx].copy()
df_test  = df_sorted.iloc[split_idx:].copy()

# NOUVEAU : sous-split du train pour la calibration honnête du seuil
# train_inner = premiers 80% du train (~204 jours)
# val         = derniers 20% du train (~51 jours, held-out)
split_idx_inner = int(len(df_train) * 0.80)
df_train_inner = df_train.iloc[:split_idx_inner].copy()
df_val         = df_train.iloc[split_idx_inner:].copy()

print(f"=== Split temporel à 3 segments ===")
print(f"df_train_inner : {len(df_train_inner):>7,} tx — {df_train_inner['Date'].min()} → {df_train_inner['Date'].max()}")
print(f"df_val         : {len(df_val):>7,} tx — {df_val['Date'].min()} → {df_val['Date'].max()}")
print(f"df_test        : {len(df_test):>7,} tx — {df_test['Date'].min()} → {df_test['Date'].max()}")
print(f"\nVérif laundering rate :")
print(f"  train_inner : {df_train_inner['Is_laundering'].mean()*100:.3f}%")
print(f"  val         : {df_val['Is_laundering'].mean()*100:.3f}%")
print(f"  test        : {df_test['Is_laundering'].mean()*100:.3f}%")
```

- [ ] **Étape 1.3 : Exécuter la cell**

Vérification : `df_train_inner` ~512k tx, `df_val` ~128k tx, `df_test` ~160k tx. Les 3 segments ont un taux de laundering proche de 0.1%.

- [ ] **Étape 1.4 : Commit**

```bash
git add feature_engineering_aml.ipynb
git commit -m "Stage calibration seuil (1/7) : split temporel 3 segments train_inner/val/test"
```

---

## Task 2 : Recalculer toutes les features sur train_inner (cells 7-12 + 14)

**Fichiers** : cells 7, 8, 9, 10, 11, 12, 14

**Pourquoi** : si le seuil est calibré sur `val`, aucune feature de `val` ne doit contenir d'information dérivée de `val` elle-même. Cela inclut l'encodage catégoriel (`OrdinalEncoder` fit) qui doit aussi se faire sur `train_inner` uniquement.

- [ ] **Étape 2.0 : Cell 14 — encoding catégoriel fit sur train_inner uniquement**

Remplacer la cell 14 par :

```python
# --- Encoding des colonnes categorielles ---
cat_cols = ['Payment_type', 'Payment_currency', 'Received_currency',
            'Sender_bank_location', 'Receiver_bank_location']

# Fit sur train_inner uniquement (anti-leakage : val et test ne participent pas)
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
oe.fit(df_train_inner[cat_cols])
df_train_inner[cat_cols] = oe.transform(df_train_inner[cat_cols])
df_val[cat_cols]         = oe.transform(df_val[cat_cols])
df_test[cat_cols]        = oe.transform(df_test[cat_cols])

# --- Features ---
new_features = ['sender_tx_count', 'receiver_tx_count',
                'is_sender_risky_country', 'is_receiver_risky_country',
                'receiver_smurfing_score', 'sender_fanout_score',
                'sender_risky_payment_count', 'receiver_risky_payment_count',
                'sender_pagerank', 'receiver_pagerank',
                'sender_out_degree', 'receiver_in_degree']

all_features = baseline_features + new_features

# Matrices pour les 3 segments
X_train_inner_new = df_train_inner[all_features]
X_val_new         = df_val[all_features]
X_test_new        = df_test[all_features]
y_train_inner_new = df_train_inner['Is_laundering']
y_val_new         = df_val['Is_laundering']
y_test_new        = df_test['Is_laundering']

# Garder X_train_new / y_train_new comme alias sur train_inner pour les cells aval
# qui n'auraient pas encore été migrées (transition douce)
X_train_new = X_train_inner_new
y_train_new = y_train_inner_new

print(f"Features utilisees ({len(all_features)}) : {all_features}")
print(f"X_train_inner shape : {X_train_inner_new.shape}")
print(f"X_val shape         : {X_val_new.shape}")
print(f"X_test shape        : {X_test_new.shape}")
```

Vérification : les 3 shapes ont le même nombre de colonnes ; positifs dans val et test cohérents avec la spec (~50 et ~210 respectivement).

- [ ] **Étape 2.1 : Cell 7 — frequence + cold start sur train_inner**

Remplacer le bloc qui calcule `sender_counts` et `receiver_counts`. Le code existant utilise `df_train.groupby(...)`. Changer pour `df_train_inner.groupby(...)` et ajouter le mapping sur `df_val` :

```python
# --- Feature 1 : Frequence + Cold Start (Stage 2B) ---
# POURQUOI metier : la frequence est un signal AML classique (comptes hyperactifs
# depassent le profil KYC particulier). Le cold start (tx_count <= 3) cible le compte
# mule tout juste ouvert : taux laundering 6x la moyenne sur SAML-D cote sender ->
# vigilance KYC renforcee defendable face a l'ACPR.
# Calculer le nombre de transactions par compte sur le TRAIN_INNER uniquement
sender_counts   = df_train_inner.groupby('Sender_account').size()
receiver_counts = df_train_inner.groupby('Receiver_account').size()

# Appliquer aux 3 segments (train_inner + val + test)
for dataset in [df_train_inner, df_val, df_test]:
    dataset['sender_tx_count']   = dataset['Sender_account'].map(sender_counts).fillna(0).astype(int)
    dataset['receiver_tx_count'] = dataset['Receiver_account'].map(receiver_counts).fillna(0).astype(int)

NEW_ACCOUNT_THRESHOLD = 3
for dataset in [df_train_inner, df_val, df_test]:
    dataset['is_new_account_sender']   = (dataset['sender_tx_count']   <= NEW_ACCOUNT_THRESHOLD).astype(int)
    dataset['is_new_account_receiver'] = (dataset['receiver_tx_count'] <= NEW_ACCOUNT_THRESHOLD).astype(int)

# Vérification
print("=== sender_tx_count ===")
print(f"train_inner — mean: {df_train_inner['sender_tx_count'].mean():.1f}, max: {df_train_inner['sender_tx_count'].max()}")
print(f"val         — comptes absents du train_inner : {(df_val['sender_tx_count'] == 0).sum()}")
print(f"test        — comptes absents du train_inner : {(df_test['sender_tx_count'] == 0).sum()}")
print(f"\n=== Cold start (Stage 2B) ===")
print(f"val  — is_new_account_sender   : {df_val['is_new_account_sender'].sum():,} tx ({100*df_val['is_new_account_sender'].mean():.1f}%)")
print(f"test — is_new_account_sender   : {df_test['is_new_account_sender'].sum():,} tx ({100*df_test['is_new_account_sender'].mean():.1f}%)")
```

Exécuter et vérifier : taux cold start similaire entre val et test.

- [ ] **Étape 2.2 : Cell 8 — appel `compute_risky_lists` sur train_inner**

Cell 8 contient la définition de `compute_risky_lists`. Aucun changement de signature. L'appel se fait en cell 9 — voir étape 2.3.

- [ ] **Étape 2.3 : Cell 9 — pays risqué sur train_inner**

Remplacer la ligne `SENDER_RISKY, RECEIVER_RISKY, RISKY_PAYMENT_TYPES = compute_risky_lists(df_train)` par `... = compute_risky_lists(df_train_inner)`. Étendre le mapping aux 3 segments :

```python
SENDER_RISKY, RECEIVER_RISKY, RISKY_PAYMENT_TYPES = compute_risky_lists(df_train_inner)

print(f"Sender risques (calcules sur train_inner) : {sorted(SENDER_RISKY)}")
print(f"Receiver risques (calcules sur train_inner) : {sorted(RECEIVER_RISKY)}")
print(f"Types de paiement risques : {sorted(RISKY_PAYMENT_TYPES)}")

for dataset in [df_train_inner, df_val, df_test]:
    dataset['is_sender_risky_country']   = dataset['Sender_bank_location'].isin(SENDER_RISKY).astype(int)
    dataset['is_receiver_risky_country'] = dataset['Receiver_bank_location'].isin(RECEIVER_RISKY).astype(int)
```

Garder l'expected check des listes (debug initial) sans modification.

- [ ] **Étape 2.4 : Cell 10 — smurfing + fan-out sur train_inner**

Remplacer `small = df_train[df_train['Amount'] < 5000]` par `small = df_train_inner[df_train_inner['Amount'] < 5000]` et étendre le mapping :

```python
small = df_train_inner[df_train_inner['Amount'] < 5000]

n_senders_par_receiver = small.groupby('Receiver_account')['Sender_account'].nunique()
n_receivers_par_sender = small.groupby('Sender_account')['Receiver_account'].nunique()

for dataset in [df_train_inner, df_val, df_test]:
    dataset['receiver_smurfing_score'] = dataset['Receiver_account'].map(n_senders_par_receiver).fillna(0).astype(int)
    dataset['sender_fanout_score']     = dataset['Sender_account'].map(n_receivers_par_sender).fillna(0).astype(int)
```

Vérifier que les distributions affichées sont cohérentes (top 10% > moyenne).

- [ ] **Étape 2.5 : Cell 11 — risky payment count sur train_inner**

Remplacer `is_risky = df_train['Payment_type'].isin(...)` par `df_train_inner`. Étendre le mapping :

```python
is_risky = df_train_inner['Payment_type'].isin(RISKY_PAYMENT_TYPES).astype(int)
sender_risky_counts   = is_risky.groupby(df_train_inner['Sender_account']).sum()
receiver_risky_counts = is_risky.groupby(df_train_inner['Receiver_account']).sum()

for dataset in [df_train_inner, df_val, df_test]:
    dataset['sender_risky_payment_count']   = dataset['Sender_account'].map(sender_risky_counts).fillna(0).astype(int)
    dataset['receiver_risky_payment_count'] = dataset['Receiver_account'].map(receiver_risky_counts).fillna(0).astype(int)
```

- [ ] **Étape 2.6 : Cell 12 — graph features sur train_inner**

Remplacer `edges = df_train.groupby(['Sender_account', 'Receiver_account'])` par `df_train_inner`. Étendre le mapping :

```python
edges = df_train_inner.groupby(['Sender_account', 'Receiver_account']).size().reset_index(name='weight')
G = nx.DiGraph()
G.add_weighted_edges_from(edges.itertuples(index=False, name=None))

pagerank   = nx.pagerank(G, weight='weight')
in_degree  = dict(G.in_degree())
out_degree = dict(G.out_degree())

for dataset in [df_train_inner, df_val, df_test]:
    dataset['sender_pagerank']    = dataset['Sender_account'].map(pagerank).fillna(0)
    dataset['receiver_pagerank']  = dataset['Receiver_account'].map(pagerank).fillna(0)
    dataset['sender_out_degree']  = dataset['Sender_account'].map(out_degree).fillna(0).astype(int)
    dataset['receiver_in_degree'] = dataset['Receiver_account'].map(in_degree).fillna(0).astype(int)
```

Garder le diagnostic discriminance sur `df_train_inner`.

- [ ] **Étape 2.7 : Exécuter cells 7→12 dans l'ordre, vérifier l'absence d'erreur de mapping**

Tous les segments (`train_inner`, `val`, `test`) doivent avoir les mêmes colonnes feature.

- [ ] **Étape 2.8 : Commit**

```bash
git add feature_engineering_aml.ipynb
git commit -m "Stage calibration seuil (2/7) : recalcul features sur train_inner (cells 7-12)"
```

---

## Task 3 : Autoencoder sur train_inner (cell 46)

**Fichier** : cell 46 (code Stage 2.2 Account Autoencoder)

**Pourquoi** : l'autoencoder doit apprendre les profils sur `train_inner` uniquement ; les embeddings de `val` et `test` viennent du lookup.

- [ ] **Étape 3.1 : Remplacer les groupby sur df_train par df_train_inner**

Dans la cell 46, remplacer :
- `sender_stats = df_train.groupby('Sender_account').agg(...)` → `df_train_inner.groupby(...)`
- `receiver_stats = df_train.groupby('Receiver_account').agg(...)` → `df_train_inner.groupby(...)`

- [ ] **Étape 3.2 : Étendre le mapping des embeddings à val et test**

À la fin de la cell, là où `sender_embs_tr` et `recv_embs_tr` sont calculés sur `df_train`, ajouter le calcul équivalent pour `df_val` et `df_test`. Modifier la dernière partie de la cell :

```python
# Mapper sur transactions : sender_emb (8) + receiver_emb (8) = 16 features par segment
def map_embeddings(df_segment):
    s = np.array([emb_dict.get(a, DEFAULT_EMB) for a in df_segment['Sender_account'].values], dtype=np.float32)
    r = np.array([emb_dict.get(a, DEFAULT_EMB) for a in df_segment['Receiver_account'].values], dtype=np.float32)
    return np.hstack([s, r])  # (n_tx, 16)

ae_tr  = map_embeddings(df_train_inner)
ae_val = map_embeddings(df_val)
ae_te  = map_embeddings(df_test)

print(f"Embeddings train_inner : {ae_tr.shape}")
print(f"Embeddings val         : {ae_val.shape}")
print(f"Embeddings test        : {ae_te.shape}")
```

- [ ] **Étape 3.3 : Exécuter la cell**

Vérification : l'autoencoder converge sur ~30 epochs, les 3 shapes affichées sont cohérentes (16 colonnes chacune).

- [ ] **Étape 3.4 : Commit**

```bash
git add feature_engineering_aml.ipynb
git commit -m "Stage calibration seuil (3/7) : autoencoder sur train_inner + embeddings val"
```

---

## Task 4 : LGBM+AE + calibration seuil sur val + report test (cell 48)

**Fichier** : cell 48 (code Stage 2.3 LGBM+AE final)

**Pourquoi** : c'est le coeur du fix. Le modèle s'entraîne sur `train_inner`, le seuil 80% est calibré sur `val`, puis appliqué sur `test` pour mesurer le recall effectif.

- [ ] **Étape 4.1 : Construire les matrices X pour les 3 segments**

Au début de la cell 48, remplacer la construction de `X_train_v2` / `X_test_v2` par 3 segments. **On réutilise les variables définies en cell 14** (`X_train_inner_new`, `X_val_new`, `X_test_new`) pour rester cohérent avec `all_features` (18 colonnes : baseline + nouvelles features) :

```python
# Concatenation 18 features (all_features de cell 14) + 16 embeddings AE = 34 features
X_train_inner_v2 = np.hstack([X_train_inner_new.values, ae_tr])
X_val_v2         = np.hstack([X_val_new.values,         ae_val])
X_test_v2        = np.hstack([X_test_new.values,        ae_te])

print(f"=== Stage 2.3 : LGBM calibre + AE (34 features, 3 segments) ===")
print(f"train_inner : {X_train_inner_v2.shape}, positifs={y_train_inner_new.sum()}")
print(f"val         : {X_val_v2.shape}, positifs={y_val_new.sum()}")
print(f"test        : {X_test_v2.shape}, positifs={y_test_new.sum()}")
```

- [ ] **Étape 4.2 : Fit du modèle sur train_inner uniquement**

```python
lgbm_ae = CalibratedClassifierCV(
    LGBMClassifier(**lgbm_best_params, class_weight='balanced',
                   random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
    method='sigmoid', cv=3
)
lgbm_ae.fit(X_train_inner_v2, y_train_inner_new)
```

- [ ] **Étape 4.3 : Calibration du seuil sur val (held-out)**

```python
# Calibration du seuil compliance sur val
proba_val = lgbm_ae.predict_proba(X_val_v2)[:, 1]
_, seuil_compliance, _ = volume_at_recall(y_val_new, proba_val, 0.80)
print(f"\nSeuil compliance calibre sur val : {seuil_compliance:.4f}")
```

- [ ] **Étape 4.4 : Application du seuil figé sur test + mesure du recall effectif**

```python
# Application du seuil FIGE sur test : on mesure le recall et le volume effectifs
proba_test = lgbm_ae.predict_proba(X_test_v2)[:, 1]
mask_alert_test         = proba_test >= seuil_compliance
recall_test_compliance  = (mask_alert_test & (y_test_new == 1)).sum() / (y_test_new == 1).sum()
vol_test_compliance     = int(mask_alert_test.sum())
ap_test                 = average_precision_score(y_test_new, proba_test)

N_SEM_TEST = (df_test['Date'].max() - df_test['Date'].min()).days / 7

print(f"\n=== Resultats LGBM+AE — calibration honnete ===")
print(f"AP test                       : {ap_test:.4f}")
print(f"Seuil compliance (fige sur val) : {seuil_compliance:.4f}")
print(f"Vol@seuil (test)              : {vol_test_compliance:,} alertes ({vol_test_compliance/N_SEM_TEST:,.0f}/sem)")
print(f"Recall@seuil (test)           : {recall_test_compliance:.1%}")
print(f"\nLecture compliance : seuil calibre sur 51 jours held-out (val), recall mesure sur test = {recall_test_compliance:.1%}")
print(f"(versus calibration directe sur test qui aurait donne 80.0% — gap = bias methodologique evite)")

# Variables pour les cellules aval (recap + stratif typo)
proba_ml = proba_test
seuil_ml = seuil_compliance
vol_ml_80 = vol_test_compliance
recall_ml_80 = recall_test_compliance
```

Le caveat « seuil calibré sur test » est désormais supprimé du commentaire en haut de la cell — remplacer par : « seuil calibré sur val held-out, recall mesuré sur test ».

- [ ] **Étape 4.5 : Exécuter la cell**

Vérification : `recall_test_compliance` ∈ [0.73, 0.80] (attendu autour de 0.75-0.78). `vol_test_compliance / N_SEM_TEST` proche de 3 000-3 500 alertes/sem.

- [ ] **Étape 4.6 : Commit**

```bash
git add feature_engineering_aml.ipynb
git commit -m "Stage calibration seuil (4/7) : LGBM+AE sur train_inner, seuil fige sur val, report test"
```

---

## Task 5 : Rule-based sur train_inner (cell 57)

**Fichier** : cell 57 (baseline rule-based)

**Pourquoi** : pour cohérence anti-leakage, les agrégats utilisés par le rule-based (sender_risky_pay_counts, intl_in_dataset diagnostic) doivent venir de `train_inner` également. Les comptages `sender_tx_count` et `receiver_tx_count` viennent déjà de la cell 7 mise à jour — automatique.

- [ ] **Étape 5.1 : Remplacer df_train par df_train_inner dans le recalcul des counts**

Dans la cell 57, remplacer :

```python
is_risky_pay_train = df_train['Payment_type'].isin(risky_payments_total).astype(int)
sender_risky_pay_counts = is_risky_pay_train.groupby(df_train['Sender_account']).sum()
```

par :

```python
is_risky_pay_train = df_train_inner['Payment_type'].isin(risky_payments_total).astype(int)
sender_risky_pay_counts = is_risky_pay_train.groupby(df_train_inner['Sender_account']).sum()
```

De même pour la ligne diagnostic `intl_in_dataset = INTL_HIGH_RISK & set(df_train['Sender_bank_location'...])` → `df_train_inner`.

- [ ] **Étape 5.2 : Exécuter la cell**

Vérification : le tableau rule-based imprime `Rule-based traditionnel (7 regles)` avec un recall qui peut légèrement varier (les seuils des features dépendent de train_inner). Recall RB attendu : 55-65%, volume 9 000-11 000 alertes/sem.

- [ ] **Étape 5.3 : Commit**

```bash
git add feature_engineering_aml.ipynb
git commit -m "Stage calibration seuil (5/7) : rule-based sur agregats train_inner (cell 57)"
```

---

## Task 6 : Récap final + stratification typologie utilisent le seuil figé (cells 61, 63)

**Fichiers** : cells 61 (récap final), 63 (stratification typologie)

**Pourquoi** : ces deux cellules appellent actuellement `volume_at_recall(y_test, proba_ml, 0.80)` qui re-calibre sur test. Elles doivent maintenant utiliser les variables `seuil_ml`, `vol_ml_80`, `recall_ml_80` définies en cell 48.

- [ ] **Étape 6.1 : Cell 61 — remplacer la recalibration sur test par les variables figées**

Dans la cell 61, remplacer :

```python
# Volume ML a recall 80% (calcule dynamiquement)
vol_ml_80, _, _ = volume_at_recall(y_test_new, proba_ml, 0.80)
sem_ml_80 = vol_ml_80 / N_SEM_TEST
```

par :

```python
# vol_ml_80, recall_ml_80, seuil_ml viennent de la cell 48 (calibration sur val)
sem_ml_80 = vol_ml_80 / N_SEM_TEST
```

Modifier également la ligne du tableau qui affiche `ML L1 a recall 80%` :

```python
print(f"{'ML L1 cible compliance (seuil fige val)':<45s} {vol_ml_80:>10,d} {sem_ml_80:>10,.0f} {recall_ml_80:>8.1%}")
```

(L'affichage du recall n'est plus 80.0% exactement, mais la vraie valeur mesurée sur test.)

- [ ] **Étape 6.2 : Cell 61 — adapter le caveat méthodologique**

Remplacer le bloc « Caveat méthodologique » par :

```python
print("\n--- Methodologie ---")
print(f"Seuil compliance calibre sur val held-out (51 jours, derniers 20% du train).")
print(f"Recall mesure sur test : {recall_ml_80:.1%} (vs cible 80% calibree en val).")
print(f"Le gap {0.80 - recall_ml_80:+.1%} reflete la generalisation val -> test.")
print(f"Ces chiffres sont defendables sans biais 'seuil choisi sur test'.")
```

- [ ] **Étape 6.3 : Cell 63 — utiliser seuil_ml au lieu de recalibrer**

Dans la cell 63, remplacer :

```python
_, seuil_80, _ = volume_at_recall(y_test_new, proba_ml, 0.80)
y_pred_80 = (proba_ml >= seuil_80).astype(int)
```

par :

```python
# Seuil fige sur val (cell 48), applique sur test pour stratifier par typologie
y_pred_80 = (proba_ml >= seuil_ml).astype(int)
print(f"Seuil de probabilite (fige sur val) : {seuil_ml:.4f}")
print(f"Recall global teste                : {recall_ml_80:.1%}")
```

Mettre à jour le titre du graphe pour refléter « recall global ~76% » au lieu de « 80% ».

- [ ] **Étape 6.4 : Exécuter cells 61 et 63**

Vérification : le tableau récap affiche une ligne ML L1 avec recall mesuré (pas 80% exactement) ; le barplot stratification affiche le recall global figé.

- [ ] **Étape 6.5 : Commit**

```bash
git add feature_engineering_aml.ipynb
git commit -m "Stage calibration seuil (6/7) : cells aval (recap + stratif) utilisent seuil fige"
```

---

## Task 7 : Mise à jour README

**Fichier** : `README.md`

**Pourquoi** : les chiffres affichés dans la section Résultats reflètent l'ancienne calibration sur test (recall 80.0% exactement). Mettre à jour avec le recall mesuré sur test au seuil figé sur val.

- [ ] **Étape 7.1 : Récupérer les vrais chiffres depuis l'output de la cell 61**

Après exécution complète du notebook, noter :
- Volume ML cible compliance (alertes/sem)
- Recall test mesuré au seuil val (probablement 75-79%)
- Seuil de probabilité figé

- [ ] **Étape 7.2 : Mettre à jour le tableau Résultats**

Dans le README, remplacer la ligne `ML cible compliance (recall 80 %)` par la valeur mesurée :

```markdown
| **ML cible compliance (seuil calibré val)** | **<volume>** | **<recall>** | — |
```

- [ ] **Étape 7.3 : Mettre à jour la lecture compliance**

Adapter le point 3 :

```markdown
3. Le rule-based **ne peut pas dépasser** <recall RB> % de recall. Le ML calibré atteint <recall ML> % de recall sur test (cible 80 % calibrée sur 51 jours de validation held-out) avec <volume> alertes / semaine.
```

- [ ] **Étape 7.4 : Ajouter une note méthodologique sous le tableau**

```markdown
> **Note méthodologique** : le seuil compliance 80 % est calibré sur un validation set held-out (derniers 51 jours du train, jamais vu pendant l'entraînement). Le recall reporté sur test est celui effectivement mesuré avec ce seuil figé — pas un choix a posteriori sur le test set. Voir [`docs/specs/2026-06-06-honest-threshold-calibration-design.md`](docs/specs/2026-06-06-honest-threshold-calibration-design.md).
```

- [ ] **Étape 7.5 : Commit final**

```bash
git add README.md feature_engineering_aml.ipynb
git commit -m "Stage calibration seuil (7/7) : README aligne sur recall mesure (seuil val held-out)"
```

---

## Self-review (effectuée)

1. **Couverture spec** : les 4 sections de la spec (3.1 split, 3.2 features, 3.3 calibration, 3.4 cellules aval) ont chacune un Task correspondant (1, 2, 3+4, 6). Task 5 (rule-based) ajouté pour cohérence anti-leakage non explicitement listé dans la spec mais cohérent. Task 7 (README) couvre la mise à jour documentaire. ✓
2. **Placeholders** : aucun TBD / TODO / "à compléter". Tous les blocs de code sont concrets. ✓
3. **Cohérence types** : `train_inner` / `val` / `test` utilisés partout ; `seuil_compliance` / `vol_test_compliance` / `recall_test_compliance` définis en cell 48 et réutilisés en cells 61, 63. ✓

## Caveats hors scope

- Le choix d'architecture LGBM+AE vs ensemble S1.3 reste fait sur test (mention honnête conservée en cell 48). Hors scope de ce plan.
- Les hyperparamètres LGBM `lgbm_best_params` ont été tunés en CV TimeSeriesSplit sur `df_train` complet. On les conserve pour éviter une re-explosion de scope (tuning anti-leakage strict aurait nécessité d'isoler val avant le tuning original). À mentionner comme caveat résiduel dans la note méthodologique.

## Notes pour l'agent qui exécute

- Le format `df_train_inner` plutôt que `df_train` n'est PAS un copier-coller mécanique : certaines cellules contiennent des diagnostics (`df_train.groupby(...)['Is_laundering'].mean()`) qui doivent passer à `df_train_inner` pour rester cohérents.
- À chaque commit, vérifier que le notebook est sauvegardé via le script `json.dump(nb, f, ensure_ascii=False, indent=1)` si modification programmatique — pour préserver les em-dash et accents (préférence utilisatrice).
- Le notebook ne contient pas de tests pytest ; la « vérification » de chaque étape passe par les prints diagnostic intégrés dans la cell modifiée.
- Pas de signature `Co-Authored-By: Claude` dans les commits portfolio (préférence utilisatrice).
