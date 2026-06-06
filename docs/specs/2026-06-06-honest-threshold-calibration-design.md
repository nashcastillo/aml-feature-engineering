# Calibration honnête du seuil compliance (recall 80%) sur validation set

**Date :** 2026-06-06
**Projet :** aml-feature-engineering
**Auteur :** Nashely Castillo
**Statut :** Spec en cours de validation

---

## 1. Contexte et motivation

Le pipeline ML actuel (LGBM+AE calibré, cellule 46 du notebook) reporte un volume de **3 177 alertes/sem au recall 80%** sur le test set. Ce chiffre est central dans le récit du projet :

> « ML L1 à recall 80% (cible compliance) → 3 177 alertes/sem »

**Problème méthodologique identifié** : ce seuil de 80% est calculé directement sur le test set via `volume_at_recall(y_test, proba_ml, 0.80)` aux cellules 46, 54, 55. C'est un **optimistic bias** classique : on choisit a posteriori le seuil qui donne exactement 80% de recall sur les données qui devraient servir uniquement à évaluer.

**Conséquence portfolio / défendabilité ACPR** : un inspecteur ou un recruteur ML averti repère ce biais en 30 secondes. Le chiffre 3 177/sem n'est pas reproductible en production où le seuil doit être fixé **avant** de voir les vraies données.

## 2. Objectif

Calibrer le seuil compliance (recall 80%) sur un **validation set strictement held-out** issu du train, puis **mesurer et reporter** le recall et le volume effectivement obtenus sur le test set avec ce seuil figé.

Critère de succès : tous les chiffres communiqués (recall, volume, ratios vs rule-based) reposent sur un seuil calibré hors test. Le caveat « seuil calibré sur test » disparaît du notebook et du README.

## 3. Conception

### 3.1 Split temporel à 3 segments

La cellule 6 actuelle produit un split chronologique 80/20 (train/test). Le nouveau split crée un segment intermédiaire :

```
df complet (~320 jours)
├── train_inner : 80% du train     ≈ 204 jours    (2022-10 → ~2023-04-25)
├── val         : 20% du train     ≈ 51 jours     (~2023-04-25 → 2023-06-19)
└── test        : inchangé         ≈ 65 jours     (2023-06-19 → 2023-08-23)
```

Contraintes :
- Les trois segments sont strictement ordonnés chronologiquement (val > train_inner, test > val).
- Aucun gap temporel : val commence à la fin de train_inner, test commence à la fin de val.
- Le test set conserve sa définition d'origine (anti-rétro-compatibilité du benchmark).

### 3.2 Recalcul des features sur train_inner uniquement

**Principe anti-leakage** : si le seuil est calibré sur val, les features utilisées sur val ne doivent contenir aucune information dérivée de val. Toutes les agrégations passent donc à `df_train_inner`.

Cellules concernées :

| Cell | Feature(s) | Modification |
|---|---|---|
| 7 | `sender_tx_count`, `receiver_tx_count`, `is_new_account_*` | `groupby` sur `df_train_inner` ; mapping sur val + test |
| 8 | `compute_risky_lists()` | Appel avec `df_train_inner` |
| 9 | `is_sender_risky_country`, `is_receiver_risky_country` | Mapping sur val + test |
| 10 | `receiver_smurfing_score`, `sender_fanout_score` | `small = df_train_inner[df_train_inner['Amount'] < 5000]` |
| 11 | `sender_risky_payment_count`, `receiver_risky_payment_count` | Comptage sur `df_train_inner` |
| 12 | `sender_pagerank`, `receiver_pagerank`, degrees | `nx.DiGraph()` construit sur `df_train_inner` |
| 45 | Account Autoencoder (10 features per-compte → 8-dim) | `node_df` construit sur `df_train_inner` ; AE entraîné dessus ; embeddings mappés val + test |

### 3.3 Calibration du seuil sur val, report sur test

Cellule 46 (LGBM+AE) modifiée :

```python
# Fit sur train_inner uniquement
lgbm_ae.fit(X_train_inner_v2, y_train_inner)

# Calibration du seuil compliance sur val (held-out)
proba_val = lgbm_ae.predict_proba(X_val_v2)[:, 1]
_, seuil_compliance, _ = volume_at_recall(y_val, proba_val, 0.80)

# Report sur test : seuil FIGÉ, on mesure recall et volume effectifs
proba_test = lgbm_ae.predict_proba(X_test_v2)[:, 1]
mask_alert_test = proba_test >= seuil_compliance
recall_test_compliance = (mask_alert_test & (y_test == 1)).sum() / (y_test == 1).sum()
vol_test_compliance = int(mask_alert_test.sum())
```

Attendu : `recall_test_compliance` ∈ [0.75, 0.79] (légère perte due à la généralisation val → test). `vol_test_compliance` proche de 3 177/sem ± marge.

### 3.4 Mise à jour des cellules aval

- **Cellule 54 (recap final)** : remplacer l'appel `volume_at_recall(y_test, proba_ml, 0.80)` par les variables figées `vol_test_compliance` et `recall_test_compliance`. Le tableau affiche la ligne `ML L1 @ seuil compliance` avec le recall mesuré (et non 80% exactement). Ajouter une note méthodo : « seuil calibré sur validation set (51 jours held-out), recall mesuré sur test ».
- **Cellule 55 (stratification typologie)** : utiliser `seuil_compliance` au lieu de recalculer un seuil sur test. Le barplot par typologie reste valide.
- **Cellule 51 (rule-based)** : inchangée (les règles n'ont pas de seuil ML).

## 4. Caveat résiduel hors scope

Le **choix d'architecture LGBM+AE vs ensemble S1.3** reste fait sur test (mentionné honnêtement cellule 46). Cette spec ne corrige pas ce biais — il faudrait un second split pour le choix de modèle, ce qui réduirait encore le train_inner et n'est pas justifié pour un dataset de cette taille. La mention honnête en cellule 46 est conservée.

## 5. Cellules à modifier — récapitulatif

| Cell | Type de modification | Effort |
|---|---|---|
| 6 | Ajout d'un sous-split chronologique train → train_inner / val | structurel |
| 7-12 | Substitution `df_train` → `df_train_inner`, ajout du mapping val | mécanique |
| 45 | `node_df` sur train_inner ; mapping AE étendu à val + test | mécanique |
| 46 | Fit sur train_inner, calibration seuil sur val, report test | structurel |
| 54 | Lecture des variables figées au lieu d'un nouvel appel | mécanique |
| 55 | Lecture du seuil figé | mécanique |
| README.md | Mise à jour des chiffres et de la note méthodo | documentaire |

## 6. Critères de succès

1. Le notebook s'exécute sans erreur de bout en bout.
2. Le pipeline n'utilise plus jamais `y_test` pour calibrer un seuil.
3. Le tableau final (cellule 54) affiche un recall test mesuré (probablement entre 75% et 79%) au lieu de 80% exactement.
4. Le README et le backlog reflètent les nouveaux chiffres et l'absence de biais.
5. La cellule 46 ne mentionne plus le caveat « seuil calibré sur test » (mais conserve le caveat sur le choix d'architecture).

## 7. Risques et points d'attention

- **Risque 1** : `train_inner` est plus court (~204 j vs 255 j) → moins de signal pour entraîner le modèle, AP test potentiellement plus bas. Ordre de grandeur estimé : −5 à −10% d'AP. Acceptable pour la rigueur gagnée.
- **Risque 2** : Le recall val 80% peut ne pas généraliser parfaitement au test → recall test < 80%. C'est le résultat attendu et la valeur ajoutée méthodologique : on montre exactement de combien.
- **Risque 3** : Certaines features (PageRank, smurfing) peuvent voir leur distribution changer sur train_inner plus court. À surveiller dans les `print` de diagnostic.
- **Risque 4** : L'autoencoder est ré-entraîné sur ~80% des comptes — embeddings différents pour les comptes très peu fréquents. Diagnostic à conserver dans la cellule 45 (taille `all_accounts`).
