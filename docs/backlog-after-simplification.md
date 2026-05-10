# Backlog après pass de simplification

**Date de création :** 2026-05-10
**Projet :** aml-feature-engineering
**Contexte :** liste des tâches identifiées pendant la critique du notebook et la pass de simplification, à traiter une fois le notebook propre.

L'objectif est de prioriser : on attaque dans l'ordre, pas tout en même temps.

---

## P1 — Améliorations méthodologiques (impact fort sur les résultats)

### 1. Score de smurfing plus discriminant
**Constat :** la cellule `51b3a838` montre que la tranche "top 10%" du score a un taux suspect (0.092%) **plus faible** que la tranche "0 < score ≤ P90" (0.123%). Le score `n_senders × n_tx` n'est pas assez sélectif — un compte légitime très actif (peu de senders, beaucoup de tx) peut avoir un score élevé sans être du smurfing.

**Pistes :**
- Tester `n_senders / n_tx` (ratio = "diversité par transaction") au lieu du produit.
- Filtrer plus strictement : `score = n_senders × n_tx` mais **uniquement** si `n_senders > médiane` (cible explicite : beaucoup de sources).
- Ajouter une feature séparée `n_senders_distincts` non multipliée par `n_tx`.

**Comment valider :** comparer le taux suspect par tranche après modif. La "top 10%" doit redevenir clairement plus suspecte que la moyenne.

### 2. Fan-out côté sender (layering)
**Idée :** symétrique du smurfing receiver. `sender_fanout_score = n_receivers_uniques × n_tx` côté sender, sur petits montants.

**Why :** capture le pattern de **layering** (dispersion des fonds), complémentaire du collecteur. L'EDA d'origine avait conclu que le pattern réseau sender seul n'était pas discriminant — mais sans le croiser avec les petits montants. Test à refaire.

**Comment valider :** suivre le même protocole que le smurfing receiver (calcul sur le train uniquement, lookup sur le test, comparaison taux suspect par tranche).

### 3. Split temporel au lieu de split random
**Constat :** le `train_test_split(random_state=42)` mélange les transactions du dataset. Pour un système AML opérationnel, on évalue sur des transactions **postérieures** au train.

**Plan :** remplacer le split random par un split sur la colonne `Date` (par ex. 80% premières dates → train, 20% dernières → test).

**Comment valider :** le recall test va probablement baisser (le modèle apprend des comptes du futur quand le split est random). C'est attendu et plus représentatif de la réalité prod.

### 4. Comparer les modèles par Average Precision, pas par recall@0.5
**Constat :** dans la section 4 du notebook, on compare DT (recall 63.5%) vs RF (57.5%) vs XGB (52.7%) **au seuil 0.5**. Mais ce seuil est arbitraire. La métrique correcte est l'**Average Precision** (aire sous la courbe PR), déjà calculée en section 8.

**Plan :** ajouter une colonne "AP" au tableau récap section 4. Le ranking des modèles peut s'inverser.

### 5. Point de décision — Les modèles actuels suffisent-ils ? — ✅ FAIT (2026-05-10)

**Résultat du gate après tuning :**
- XGB TUNE recall CV = **0.762** ≥ 60% ✓
- XGB TUNE AP = **0.0681** ≥ 0.05 ✓

**Décision : on garde les modèles actuels (XGB TUNE). P3 (modèles alternatifs) non déclenché.**

Si plus tard les améliorations P1.3 (split temporel) font chuter le recall sous 60%, on rouvre P3.

---

## P2 — Bugs / propreté méthodologique

### 6. Baseline avec encoding fit sur full dataset
**Constat :** cellule `77da55b8` (baseline) fait `oe.fit_transform(df_baseline[cat_cols])` sur le dataset complet **avant** le split. C'est incohérent avec la section 2 (qui fit l'encoder sur df_train uniquement). Le baseline a donc un mini-leakage.

**Plan :** déplacer le `fit_transform` après le split, fit sur `X_train_base` uniquement, transform sur `X_test_base`. Le recall baseline peut baisser de quelques dixièmes de point — à documenter.

### 7. `fillna(0)` pour les comptes inconnus du test → biais
**Constat :** un compte présent dans le test mais absent du train reçoit `sender_tx_count=0`. Or, par construction, ce compte fait au moins 1 transaction (celle du test). Mettre 0 introduit un biais.

**Plan :** utiliser la **médiane** du train comme fallback. Ajouter aussi une feature binaire `is_unknown_account` pour que le modèle distingue "compte connu peu actif" de "compte jamais vu".

### 8. Smurfing : seuil 5000 non normalisé par devise
**Constat :** `df_train['Amount'] < 5000` filtre les "petits montants", mais 5000 USD ≠ 5000 INR ≠ 5000 EUR. La feature ne capture pas la même chose selon la devise.

**Plan :** soit convertir tous les montants en EUR via une table de change (idéal mais lourd), soit calculer un seuil **par devise** (ex : P10 des montants par `Payment_currency`).

### 9. Volume minimum dans `compute_risky_lists()`
**Constat :** un pays avec très peu de transactions dans le train pourrait apparaître dans la liste par hasard statistique (ex : 1 sur 5 = 20%). Mentionné comme "V2" dans le spec leakage.

**Plan :** ajouter un paramètre `min_volume=100` qui exige au moins 100 transactions dans le groupe avant d'être candidat à la liste risquée.

---

## P3 — Modèles alternatifs à comparer

> **À déclencher uniquement** si le gate P1.5 (point de décision) conclut que les modèles actuels ne suffisent pas. Sinon, sauter cette section.

### 10. Anomaly detection non supervisée
- **Isolation Forest** : naturel sur 0.1% de positifs.
- **Autoencoder** (avec reconstruction error comme score d'anomalie).

Bénéfice : ne dépend pas du label `Is_laundering`, donc plus robuste si la qualité du label est douteuse (ce qui est probable dans un dataset synthétique).

### 11. LightGBM / CatBoost
**Pourquoi :** souvent meilleurs que XGBoost sur déséquilibre + categorielles. CatBoost gère nativement les categorielles (pas besoin d'OrdinalEncoder).

### 12. SMOTE / undersampling vs `class_weight='balanced'`
**À tester :** rééchantillonner le train (SMOTE pour générer des positifs synthétiques, ou undersampler les négatifs) et comparer au `class_weight='balanced'` actuel.

---

## P4 — Pour la soutenance

### 13. Recommandation d'un seuil opérationnel chiffré — ✅ FAIT (2026-05-10)

**Décision finale après tuning :**

| Critère | Valeur |
|---|---|
| Modèle final | **XGB TUNE** (`max_depth=3, learning_rate=0.05, min_child_weight=1, n_estimators=200`) |
| Recall CV 5-fold | 0.762 (+/- 0.006) — le plus stable des 3 |
| Average Precision | 0.0681 — 3× DT, 1.4× RF |
| **Seuil opérationnel recommandé** | **0.7** |
| Recall au seuil 0.7 | 61.7% (vs 42.5% baseline) |
| Précision au seuil 0.7 | 1.34% |
| Alertes par an | ~7 700 |
| Dossiers/vrai cas | 74 |
| Charge analystes | ~0.4 ETP/an |

**Pourquoi seuil 0.7 :** compromis entre recall et volume d'alertes traitables. À seuil 0.5, on détecte 76% mais 155 dossiers/cas (1 ETP). À seuil 0.8, recall 46% mais 42 dossiers/cas (0.2 ETP).

**Pourquoi XGB et pas DT :** DT a un recall légèrement supérieur sur le split simple (0.772) mais en CV XGB est meilleur (0.762 vs 0.734) et beaucoup plus stable (std 0.006 vs 0.041). DT a aussi une AP 3× plus faible (0.0209 vs 0.0681), ce qui signifie que DT ne marche qu'au seuil par défaut, alors que XGB est exploitable sur toute la courbe.

**À faire encore :** ajouter cette recommandation directement dans le notebook (markdown encadré au-dessus de la section 8).

### 14. Documenter les listes finales obtenues sur le train
**Constat :** la pass anti-leakage révèle des modalités risquées légèrement différentes des listes hardcodées d'origine (ex : `India` apparaît dans `RECEIVER_RISKY` calculée sur le train, alors qu'elle était absente de la liste full-dataset).

**Plan :** dans le markdown de section 2, ajouter un encadré "Listes finales identifiées sur le train (anti-leakage)" avec les 3 listes et un commentaire sur les différences avec l'EDA d'origine. C'est un beau "vendable" pour la soutenance.

---

## Ordre de traitement suggéré

1. **D'abord** finir la pass de simplification en cours (priorité absolue : code lisible).
2. **P1.1 (smurfing score) + P1.2 (fan-out sender)** ensemble — c'est le même fichier de cellule, c'est cohérent.
3. **P2.6 (baseline encoding)** — vite fait, élimine un dernier mini-leakage.
4. **P1.3 (split temporel)** — gros impact sur les chiffres, à faire avant la soutenance.
5. **P1.4 (Average Precision)** — facile à ajouter, change le ranking.
6. **P1.5 (point de décision)** — évaluer si les modèles actuels suffisent ou s'il faut basculer sur P3.
7. Si P1.5 dit "basculer" → **P3** (modèles alternatifs). Sinon, sauter P3.
8. **P4.13 + P4.14** — finitions soutenance.
9. P2 (autres bugs) = bonus si temps disponible.

---

## Ce qui n'est PAS dans le backlog (laissé tel quel)

- Refacto en module Python `.py` séparé : ajoute une couche d'abstraction qui complique la lecture pour un débutant.
- Optimisations de performance avancées : le notebook tourne en quelques minutes, pas un sujet.
- Réécriture des modèles avec des frameworks deep learning : hors scope du projet Jedha.
