# Backlog après pass de simplification

**Date de création :** 2026-05-10
**Dernière mise à jour :** 2026-06-06 — Stage 2.4 SMOTE sur LGBM+AE documenté comme finding négatif
**Projet :** aml-feature-engineering

> **Renommage 2026-05-20** : ce qu'on appelait « **GNN embeddings** » dans les versions précédentes est en réalité un **autoencoder MLP per-compte** (pas de message passing graphique). Renommé partout en « **Account Autoencoder (AE)** » pour honnêteté technique. La composante « graphe » réelle du projet vient du Stage 1.2 (NetworkX : PageRank + degrees), déjà intégré aux 18 features de base.

> **Règles R7 / R8 désactivées 2026-05-27** : deux règles velocity glissante ont été testées puis retirées car SAML-D ne simule pas ces patterns (0 vrai positif quel que soit le seuil) :
> - **R7 burst 24h** : `sender_tx_24h > 3 OU receiver_tx_24h > 3` — typologie structuring court terme (3+ envois en 24h via MSB = frais récurrents incompatibles avec usage légitime particulier)
> - **R8 velocity 28j** : `sender_tx_28d > 30 OU receiver_tx_28d > 30` — typologie velocity mensuelle (30+ envois/mois = signal LCB-FT MSB)
>
> Diagnostic test : sur 210 transactions laundering du test set, R7 capte 0 TP même en variant le seuil de 3 à 15. Le dataset synthétique SAML-D ne génère pas ces patterns temporels. **À réactiver en production sur données MSB réelles** où ces typologies sont bien documentées (guides Tracfin / GAFI). Le code de calcul des features `sender_tx_24h`, `receiver_tx_24h`, `sender_tx_28d`, `receiver_tx_28d` via `groupby().rolling(window, closed='left')` est conservé dans l'historique git (commit `f575753`) pour réutilisation directe.

> **Liste pays sanctions 2026-05-21** : la règle R2 du baseline rule-based utilise une **UNION 4 sources officielles** (41 pays uniques, mai 2026) :
> - **ONU** Security Council sanctions regimes (14 pays sous sanctions actives, avril 2026) — [un.org Consolidated List](https://main.un.org/securitycouncil/en/content/un-sc-consolidated-list)
> - **OFAC** sanctions programs incl. Ethiopia EO 14046 (8 pays apportés en plus de l'ONU) — [ofac.treasury.gov](https://ofac.treasury.gov/sanctions-programs-and-country-information)
> - **GAFI/FATF** black + grey list (plénière 13 février 2026 : 3 black + 22 grey) — [fatf-gafi.org](https://www.fatf-gafi.org/en/countries/black-and-grey-lists.html)
> - **UE** Règlement délégué 2016/1675 amendé par 2025/1184, 2026/46 et 2026/83 (26 pays) — [eur-lex 2026/46](http://data.europa.eu/eli/reg_del/2026/46/oj)
>
> **10 pays retirés** des listes en 2024-2026 après réformes (Barbados, Cayman Islands, Gibraltar, Jamaica, Jordan, Panama, Philippines, Senegal, Uganda, UAE) — exclus conformément aux dernières plénières GAFI et règlements UE.
>
> Cette liste **complète** la liste interne `SENDER_RISKY` calculée sur train (Albania, Italy, Netherlands…) via `set | set`. Lecture : « ma liste combine 4 sources réglementaires officielles à jour + enrichissement interne par retour d'expérience sur les typologies observées ».

---

## État final 2026-05-19 — Synthèse complète terminée

**Modèle final retenu :** LightGBM tuné + calibration sigmoid + **Account Autoencoder (AE) embeddings**

| Critère | Valeur |
|---|---|
| Hyperparamètres LGBM | `max_depth=5, learning_rate=0.1, min_child_samples=10` |
| AE (Account Autoencoder) | MLP PyTorch sur 10 features per-compte → 8-dim latent (425k comptes) |
| Méthodologie tuning | TimeSeriesSplit + optimisation AP |
| Calibration | Sigmoid (Platt scaling, cv=3) |
| Features | **34** (18 base + 16 AE = 8 sender + 8 receiver) |
| AP test (calibré) | **0.6415** |
| **Volume @ recall 80%** | **3,128 alertes/sem** |
| Seuil opérationnel | 0.2433 = probabilité calibrée 24.3% (= 185× le taux de base) |

**Comparaison vs Baseline rule-based** (5 règles métier combinées OR, mesurées sur le même test set) :

| Système | Alertes/sem | Recall |
|---|---|---|
| Rule-based (5 règles) | 10,406 | 62.9% |
| **ML L1 à volume égal** | 10,406 | **98.6%** |
| **ML L1 à recall égal** | **21** | 62.9% |
| **ML L1 à recall 80%** (cible compliance) | 3,128 | **80.0%** |

**3 chiffres cles du projet :**
1. À volume égal : ML détecte **1.6× plus** de cas suspects
2. À recall égal (62.9%) : ML utilise **490× moins** de volume — précision 66% sur top 200 alertes
3. Cible compliance recall 80% : Rule-based plafonne à 62.9%, ML L1 atteint 80% avec 3,128 alertes/sem (3.3× moins que rule-based à recall équivalent)

**⚠️ Caveat méthodologique :** le ratio 490× au point recall 62.9% est particulièrement élevé et reflète aussi le **caractère synthétique** de SAML-D. Sur des données bancaires réelles, ce ratio serait probablement inférieur. À mentionner honnêtement dans toute communication du projet.

**Trajectoire complète depuis la baseline initiale :**

| Étape | Vol/sem @ recall 80% |
|---|---|
| Baseline S2 (14 features, RF) | 5,950 |
| Stage 1.2 graph features (NetworkX) | 5,678 (-5%) |
| Stage 1.3 ensemble rank-avg RF+XGB+LGBM | 4,264 (-28%) |
| **Stage 2.2+2.3 LGBM + AE (34 features)** | **3,128 (-47%)** |

---

## Chronologie complète des stages

| Date | Stage | Résultat |
|---|---|---|
| 2026-05-10 | Pass de simplification du notebook | ✅ |
| 2026-05-11 | P1.1 smurfing amélioré + P1.2 fan-out sender | ✅ |
| 2026-05-11 | P1.3 split temporel 80/20 | ✅ |
| 2026-05-13 | P1.4 tuning par AP + TimeSeriesSplit CV | ✅ |
| 2026-05-15 | Métrique `volume_at_recall_80` | ✅ |
| 2026-05-15 | Lever 1 Phase A features temporelles | ❌ reverted (négatif) |
| 2026-05-15 | LightGBM 4ème modèle | ✅ |
| 2026-05-15 | Calibration sigmoid des probabilités | ✅ |
| 2026-05-15 | Baseline rule-based + comparaison ML vs RB | ✅ |
| 2026-05-17 | Stage 1.1 velocity features (tenure_days) | ❌ reverted (négatif) |
| 2026-05-17 | Stage 1.2 graph features NetworkX (PageRank + degrees) | ✅ |
| 2026-05-17 | Stage 1.3 ensemble rank-average RF+XGB+LGBM | ✅ |
| 2026-05-17 | Stage 3 anomaly detection benchmarks (IF, OCSVM, LOF) | ❌ non-adoptés |
| 2026-05-17 | Stage 4.1 SHAP analysis | ✅ |
| 2026-05-17 | Stage 4.2 Stratification par typologie de blanchiment | ✅ |
| 2026-05-18 | Stage 2.1 VAE embeddings (PyTorch) | ❌ non-adopté (RF+VAE seul, mais ensemble capture déjà le signal) |
| 2026-05-19 | Stage 2.2 Account Autoencoder embeddings (MLP PyTorch, anciennement "GNN") | ✅ winner |
| 2026-05-19 | Stage 2.3 Feature fusion (concatenate 18+16) + LGBM final | ✅ winner |
| 2026-05-19 | Visualisations finales (PR curve, confusion matrix, recall par typologie) | ✅ |

---

## Findings négatifs documentés (preuve de rigueur expérimentale)

### Lever 1 Phase A — Features temporelles + behavioral (testées et revertées)
**Testé** : `hour_of_day`, `day_of_week`, `amount_log`, `amount_zscore_sender`, `amount_zscore_receiver`.
**Résultat** : AP test −26%, volume@recall80 +4% (régression).
**Cause** : SAML-D est synthétique, le diagnostic montre distribution suspect quasi-plate par tranche horaire (0.088% – 0.108%). Les patterns temporels production AML n'existent pas sur ce dataset.
**Conclusion** : reverté. Argument projet "next step = vraies données KYC/historique/graphe".

### Stage 1.1 — Velocity features sender_tenure_days (testée et revertée)
**Testé** : `sender_tenure_days`, `receiver_tenure_days` (jours depuis première apparition du compte dans train).
**Résultat** : AP test −78%, volume@recall80 +1.5% (régression sévère).
**Cause** : biais sémantique train/test (tenure=0 signifie "première apparition" en train mais "jamais vu" en test). Le modèle confond les deux.
**Conclusion** : reverté. Documenté comme leçon méthodologique sur les features temporelles dans un split temporel.

### Stage 2.1 — VAE embeddings (testée, non adoptée)
**Testé** : VAE PyTorch (encoder 32→16→8 latent, decoder symétrique) sur les 18 features standardisées.
**Résultat** : RF+VAE seul = 4,565/sem (-19.6% vs RF seul à 5,678/sem).
**MAIS** : l'ensemble S1.3 sans VAE atteignait déjà 4,264/sem. VAE n'apporte pas de gain marginal au-dessus de l'ensemble.
**Conclusion** : non adopté dans le notebook final. L'Account Autoencoder (Stage 2.2) capture le signal structurel plus efficacement.

### Stage 3 — Anomaly detection (Isolation Forest, OCSVM, LOF — non adoptées)
**Testé** : approches non-supervisées du paper Kungu et al. 2026 (IF, OCSVM, LOF).
**Résultat** : Vol@R80 = 7,207 / 9,956 / 9,686 sem respectivement. Tous 2-3× pires que l'ensemble supervised.
**Conclusion** : confirme que supervised >> unsupervised quand des labels existent. Argument méthodologique pour positionner notre approche.

### LightGBM avec `scale_pos_weight=n_neg/n_pos`
**Testé** : config standard XGB (`scale_pos_weight=1026`).
**Résultat** : AP=0.0015 (catastrophique). LightGBM amplifie scale_pos_weight différemment de XGBoost.
**Fix** : `class_weight='balanced'` (sklearn-style). AP CV=0.0229, AP test=0.0551, volume@recall80=6,892/sem.
**Conclusion** : leçon méthodologique sur la transposition d'hyperparamètres entre frameworks ML.

### Stage 2.4 — SMOTE sur LGBM+AE (testé, contre-productif sur pipeline final)
**Date** : 2026-06-06.
**Contexte** : étude de sensibilité au rééchantillonnage (cellule 56 du notebook) → SMOTE bat `class_weight='balanced'` sur les 3 modèles tunés en CV TimeSeriesSplit (LGBM +105%, RF +157%, XGB +16% AP). Validation hold-out sur LGBM seul confirme un gain réel mais beaucoup plus modeste (+14% AP test). Question naturelle : ce gain persiste-t-il une fois les 16 embeddings autoencoder ajoutés (pipeline final LGBM+AE) ?
**Testé** : `imblearn.Pipeline(SMOTE, LGBM)` wrappée dans `CalibratedClassifierCV(method='sigmoid', cv=3)`. Architecture anti-leakage stricte — SMOTE appliqué uniquement sur le train de chaque split interne, calibration apprise sur le held-out non resamplé, seuils interprétables comme probabilité calibrée sur la vraie distribution 0.1%.
**Résultat** :

| Métrique | baseline class_weight | SMOTE | Δ |
|---|---|---|---|
| AP test | **0.6055** | 0.5097 | **−16 %** |
| Vol@R80 (alertes) | 35,315 | 41,287 | +17 % |
| Vol@R80 (alertes/sem) | **3,803** | 4,446 | +643/sem |
| Seuil calibré (proba) | 0.063 | 0.002 | écrasé |

**Cause hypothèse 1 — Espace AE non-linéaire** : SMOTE interpole linéairement entre 2 vrais positifs dans un espace 34-dim dont 16 dimensions sont des embeddings autoencoder (manifold non-linéaire des features compte). Les positifs synthétiques se retrouvent hors-manifold, statistiquement absurdes — un "compte moyen" entre 2 blanchisseurs n'existe pas dans l'embedding AE.

**Cause hypothèse 2 — Calibration sigmoid saturée** : le seuil calibré SMOTE tombe à 0.002 (vs 0.063 baseline). Même avec calibration apprise sur held-out non-SMOTE, le modèle SMOTE écrase les probabilités vers zéro. La sigmoid ne redresse pas sur cette plage. `method='isotonic'` corrigerait peut-être mais resterait une rustine.

**Conclusion** : winner inchangé, `class_weight='balanced'` conservé sur le pipeline final LGBM+AE. SMOTE est documenté comme finding négatif méthodologique :
- **Sur LGBM seul** (18 features tabulaires) : SMOTE = +14% AP test (gain réel mais à valider en prod).
- **Sur LGBM+AE** (34 features avec embeddings non-linéaires) : SMOTE = −16% AP test (contre-productif).
- **Leçon générale** : l'efficacité de SMOTE dépend fortement de la nature des features. Sur des features riches et non-linéaires (graph, embeddings, manifold), l'interpolation linéaire devient contre-productive.

**Code** : cellule 47 du notebook (`Stage 2.4`) conservée comme preuve documentée. Cellule 56 (étude de sensibilité 9 combinaisons) conservée en amont.

---

## Bugs P2 — testés et résultats

Tous les 4 bugs identifiés ont été testés. Résultat mixte :

### Fix #1 — Baseline encoding fit sur full dataset ✅ APPLIQUÉ (2026-05-20)
**Constat** : cellule `77da55b8` faisait `oe.fit_transform()` sur le dataset complet avant split → mini-leakage dans le baseline historique.
**Fix** : split avant encoding, fit sur `df_train_base` uniquement, `handle_unknown='use_encoded_value', unknown_value=-1`.
**Impact** : recall baseline inchangé (0.405). Pas d'impact perf mais méthodologie propre.

### Fix #2 — `fillna(0)` pour comptes inconnus du test ❌ TESTÉ ET REVERTÉ (2026-05-20)
**Testé** : `fillna(median_train)` + ajout features `is_unknown_sender`, `is_unknown_receiver`.
**Résultat** : Vol@R80 LGBM+AE passé de 3,128 → 3,281 (+4.9%). Les flags `is_unknown_*` sont **constants à 0 sur le test** (SAML-D synthétique = même population train/test, tous les comptes du test sont dans le train).
**Conclusion** : reverté. Sur ce dataset, les features sont inutiles. À retenter sur des données réelles où les comptes test peuvent être totalement nouveaux.

### Fix #3 — Smurfing seuil 5000 non normalisé par devise ❌ TESTÉ ET REVERTÉ (2026-05-20)
**Testé** : seuil P10 par `Payment_currency` au lieu du fixe 5000.
**Résultat** : fan-out score voit sa discriminance s'inverser (top 10% = 0.014% suspect vs moyenne 0.097% = **anti-corrélation**). Vol@R80 LGBM+AE dégradé.
**Cause** : P10 par devise rend la base "petits montants" trop large/floue, dilue le signal layering.
**Conclusion** : reverté. Le seuil fixe 5000 reste meilleur sur SAML-D.

### Fix #4 — `compute_risky_lists()` sans volume minimum ✅ APPLIQUÉ (2026-05-20)
**Fix** : ajout `min_volume=100` excluant les modalités avec trop peu de transactions du train.
**Impact** : neutre sur LGBM+AE winner (3,128/sem inchangé). Méthodologie statistiquement plus robuste.

**Bilan P2** : 2 fixes appliqués (méthodologie), 2 reverts documentés (négatifs). Vol@R80 final = 3,128/sem inchangé.

---

## Hors scope (laissé tel quel)

- **Refacto en module Python `.py`** : ajouterait une couche d'abstraction qui complique la lecture pour un débutant.
- **Optimisations de performance avancées** : le notebook tourne en quelques minutes, pas un sujet.
- **Vraies données bancaires / KYC / historique multi-mois** : nécessaires pour réellement valider la performance en production. À mentionner en revue comme "extension naturelle hors scope projet".
- **CatBoost** : gain marginal vs LGBM, complexité d'encoding catégorielle. Skip.
- **BalancedRandomForest** : redondant avec RF + class_weight balanced déjà testés.
