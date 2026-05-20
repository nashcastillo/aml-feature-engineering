# Réduire le volume d'alertes à recall ≥ 80%

**Date :** 2026-05-13
**Projet :** aml-feature-engineering
**Auteur :** Nashely Castillo
**Statut :** Validé par l'utilisatrice le 2026-05-13

---

## 1. Contexte et motivation

L'utilisatrice est compliance officer AML par métier (10 ans d'expérience). La motivation centrale du projet : démontrer que le ML peut résoudre le problème historique des systèmes AML conventionnels — **trop d'alertes à faible valeur** qui saturent les équipes L2 sans détecter assez de cas réels.

**Benchmark opérationnel WU (Western Union)** : ~600 alertes/semaine traitées au niveau L2, déjà décrit comme saturant.

**État actuel du notebook (après P1.1+P1.2+P1.3+P1.5)** :
- Modèle gagnant : RF TUNE
- Recall test : 0.800
- AP : 0.0511
- Volume alertes à seuil 0.5 : ~7100/semaine sur test set (×12 la capacité WU L2)

**Constat critique** : optimiser l'AP seul, ou viser les 600/sem de WU comme cible stricte, sont deux objectifs mal posés. Le **vrai dilemme compliance** est ailleurs.

## 2. Objectif

**Réduire le volume d'alertes tout en maintenant un recall ≥ 80% sur le test set.**

- Le **recall 80%** est un plancher non-négociable (défendable face à un régulateur ACPR/TRACFIN — "on ne peut pas rater 20%+ des cas suspects").
- Le **volume** est la variable à minimiser sous cette contrainte.
- La contrainte "atteindre 600/sem" est **abandonnée** : irréaliste sur ce dataset sans features KYC/historique/graphe.

## 3. Métriques

### Métrique opérationnelle principale (NEW)

`volume_at_recall_80` : nombre d'alertes par semaine nécessaires pour atteindre recall = 80% sur le test set. C'est la métrique opérationnelle que l'utilisatrice comprend, et celle qui parlera au jury Jedha.

### Métrique d'optimisation (changement de tuning)

**AP (Average Precision)** : maximise la courbe PR entière → améliore le tradeoff recall/volume à tous les seuils → permet d'atteindre `volume_at_recall_80` minimal.

### Métriques de support (déjà présentes)

- Recall global @ seuil 0.5
- Precision @ seuil 0.5
- Tableau seuils complet
- Recall@K (où K = budget alertes)

## 4. Plan en 4 semaines

### Semaine 1 — Refocus méthodologique (3-5 jours)

| Tâche | Coût code | Livrable |
|---|---|---|
| Ajouter métrique `volume_at_recall_80` au tableau seuils | ~10 lignes | Première mesure publiable |
| Tuner par AP au lieu de recall@0.5 (P1.4) | 1 ligne | Tuning cohérent avec défense |
| Migrer StratifiedKFold → TimeSeriesSplit dans `manual_cv_tuning` | 2 lignes | Méthodo CV alignée production |

**Vérification préalable TimeSeriesSplit** : compter les positifs dans chaque fold (n_splits=3). Si un fold a < 30 positifs, conserver StratifiedKFold et documenter le compromis.

**Livrable S1** : 3 modèles (DT, RF, XGB) re-tunés avec méthodologie propre. Premier chiffre `volume_at_recall_80` mesuré.

### Semaine 2 — Pousser le modèle (3-5 jours)

| Tâche | Coût code | Livrable |
|---|---|---|
| Ajouter LightGBM (4ème modèle) | ~30 lignes | Comparaison 4 modèles |
| Calibration des probabilités sur le winner (`CalibratedClassifierCV`, méthode `sigmoid` par défaut, `isotonic` testé en comparatif) | ~5 lignes | Seuil interprétable probabilistement |
| **Optionnel** : BalancedRandomForest (5ème modèle) | ~3 lignes | Couverture famille bagging+undersampling |

**Livrable S2** : modèle final identifié. `volume_at_recall_80` du modèle final mesuré et calibré.

### Semaine 3 — Narratif comparatif (keystone soutenance) (3-5 jours)

| Tâche | Coût code | Livrable |
|---|---|---|
| Baseline rule-based simple sur le même dataset (4-5 règles métier) | ~30 lignes | Point de comparaison ancré sur les données |
| Comparaison ML vs rule-based à volume égal ET à recall égal | ~20 lignes | Tableau central de la soutenance |
| Tableau récap final : pour chaque modèle (recall, volume@recall80, AP, alertes/sem) | ~20 lignes | Slide-ready |

**Règles candidates pour la baseline rule-based** :
- `amount > 10_000`
- `(amount > 1_000 AND sender_bank_location IN risky_countries)`
- `(amount > 1_000 AND payment_type IN risky_payments)`
- `sender_tx_count > 100 OR receiver_tx_count > 100`
- `receiver_smurfing_score > P90_train OR sender_fanout_score > P90_train` (P90 calculé sur train pour anti-leakage)

→ Le set final sera ajusté lors de l'implémentation S3 selon ce qui donne un bon point de comparaison (recall similaire ou volume similaire au ML).

**Livrable S3** : chiffre central de la soutenance, par exemple : "À volume égal, mon ML L1 atteint 80% recall vs ~30% pour rule-based — soit **2.5× plus de cas détectés**".

### Semaine 4 — Soutenance

- Slide deck (focus narratif compliance + chiffres centraux)
- Répétitions oral
- Anticipation questions jury

## 5. Choix méthodologiques

### Architecture
- **Single-stage ML** comme L1 (pas de L2 ML ni de L2 règles strict)
- L2 mentionné comme "next step" dans la soutenance, en notant les features manquantes (KYC, graphe relationnel, historique multi-mois)

### Split des données
- Split train/test temporel 80/20 (conservé depuis P1.3)
- CV interne `TimeSeriesSplit(n_splits=3)` (changement S1)

### Anti-leakage
- Features calculées sur train uniquement, appliquées sur test via `.map().fillna(0).astype(int)` (conservé)
- Fonction `build_features_on_fold` pour features anti-leakage en CV (conservée, à adapter si TimeSeriesSplit déstabilise certains folds)
- Tuning sur `df_train` uniquement, jamais sur `df` complet (corrigé P1.5)

### Modèles à comparer
- DT, RF, XGB (conservés, re-tunés)
- LightGBM (ajouté S2)
- BalancedRandomForest (optionnel S2)

### Gestion du déséquilibre
- `class_weight='balanced'` (DT, RF, BalancedRF) — conservé
- `scale_pos_weight = n_neg/n_pos` (XGB, LGBM) — conservé
- Pas de SMOTE/oversampling (apport marginal, risque sur split temporel)

### Tuning
- Grille manuelle 3-fold via `manual_cv_tuning`
- **Optimisation par AP** (changement S1)
- CV temporel via `TimeSeriesSplit` (changement S1)

### Seuil de décision
- Avant calibration : seuil 0.5 par défaut
- Après calibration : choix du seuil **par contrainte de recall** (recall ≥ 80%), volume minimisé
- Tableau seuils étendu avec `alertes/sem` et `volume_at_recall_80`

## 6. Ce qu'on N'inclut PAS et pourquoi

| Hors scope | Raison |
|---|---|
| Atteindre les 600 alertes/sem de WU comme cible stricte | Irréaliste sur ce dataset sans features supplémentaires (KYC, historique, graphe) |
| L2 filter à base de règles **strict** sur la sortie ML | Risque de jeter des vrais positifs → recall < 80% |
| SMOTE / oversampling agressif | Apport marginal sur ce déséquilibre, risque sur split temporel |
| Optuna (tuning bayésien) | `manual_cv_tuning` existant suffit, pas de bénéfice à compliquer |
| MLflow | Outil pro mais hors-scope projet académique solo |
| CatBoost / Autoencoder / One-Class SVM | Over-engineering, gain hypothétique vs coût implémentation |
| Réécriture from-scratch du notebook | ~80% du code actuel est solide et défendable |

## 7. Risques et mitigations

| Risque | Mitigation |
|---|---|
| TimeSeriesSplit donne des folds avec trop peu de positifs | Vérifier avant migration : compter positifs par fold (cible ≥ 30). Si insuffisant, conserver StratifiedKFold et documenter |
| Tuning par AP fait baisser recall à seuil 0.5 | Ne plus mesurer recall@0.5 isolé. Mesurer `recall_at_volume_target` et `volume_at_recall_80` |
| LightGBM perd contre les modèles existants | Documenter comme test négatif (utile soutenance : "j'ai testé, ça n'a pas apporté") |
| Calibration `isotonic` overfit (peu de données positives) | Tester aussi `sigmoid` (plus paramétrique). Comparer AP avant/après calibration |
| Rule-based baseline trop simpliste ou trop sophistiquée | Justifier par l'expérience métier ("règles typiques que je vois en production WU"). Itérer 2-3 versions si besoin |
| 4 semaines trop court pour tout livrer | Plan S1+S3 sont les MUST. S2 LightGBM et BalancedRF sont des SHOULD. Calibration est un MUST (peu coûteux) |

## 8. Critères de succès

À la fin du projet, le notebook doit présenter :

1. ✓ **Modèle final avec recall test ≥ 80%** (contrainte dure, non-négociable)
2. ✓ **`volume_at_recall_80` mesuré** sur le modèle final
3. ✓ **Comparaison à un baseline rule-based** sur le même dataset, à volume égal ou à recall égal
4. ✓ **Méthodologie cohérente** : split temporel + CV temporel + tuning par AP
5. ✓ **Notebook propre**, exécuté de bout en bout sans erreur, sans data leakage
6. ✓ **Narratif soutenance** centré sur : "ML L1 atteint 80% recall vs ~30% rule-based, à volume comparable — 2.5× plus de cas suspects détectés"

## 9. Référence backlog

Cette spec ferme :
- **P1.4** (tuner par AP) — S1
- **Nouveau P1.6** (TimeSeriesSplit interne au tuning) — S1
- **Nouveau P1.7** (métrique volume_at_recall_80) — S1
- **P3.11** (LightGBM) — S2
- **Nouveau P3.12** (calibration probabilités) — S2
- **Nouveau P3.13** (BalancedRandomForest) — S2 optionnel
- **Nouveau P3.14** (baseline rule-based comparatif) — S3
- **P4.13** (mise à jour finale du modèle gagnant) — S3

Le backlog `docs/backlog-after-simplification.md` sera mis à jour à la fin de chaque semaine.
