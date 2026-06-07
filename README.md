# AML Feature Engineering — Détection de blanchiment : ML vs rule-based

> Projet de feature engineering et machine learning appliqué à la détection de transactions de blanchiment d'argent (LCB-FT). Comparaison rigoureuse d'un système ML calibré vs un système rules-based traditionnel sur le dataset SAML-D.

## Contexte métier

Les institutions bancaires et MSB (Money Service Business) sont tenues par les régulations LCB-FT (Tracfin en France, FATF / GAFI à l'international, ACPR pour le contrôle) de détecter et signaler les transactions suspectes. Les systèmes de détection traditionnels reposent sur des **règles métier** (montant > seuil, pays sanctionné, payment type à risque). Ces règles ont des limites bien connues :

- Beaucoup de **faux positifs** : volume d'alertes souvent ingérable par les équipes compliance
- Pas de couverture des **typologies subtiles** (smurfing, layering, fan-out)
- **Seuils statiques** difficiles à calibrer face à l'évolution des typologies

Ce projet quantifie le **gain d'un système ML** vs un rules-based de référence, sur des critères opérationnels compliance (recall, volume d'alertes hebdomadaire, défendabilité métier).

## Dataset

[SAML-D](https://github.com/IBM/SAML-D) — Synthetic Anti-Money Laundering Dataset (IBM), 800 000 transactions, 21 pays, ~0.1 % laundering rate. **Caractère synthétique mentionné explicitement** dans toutes les communications du projet (caveat méthodologique).

Split temporel 80 / 20 : train sur 255 jours (oct. 2022 - juin 2023), test sur 65 jours (juin - août 2023). Train sous-divisé en train_inner (204 jours) + val held-out (51 jours) pour calibrer le seuil compliance sans toucher au test. Anti-leakage strict.

## Architecture

### Feature engineering (18 features de base)

- **Fréquence** : `sender_tx_count`, `receiver_tx_count` (all-time sur train)
- **Listes pays à risque** : UNION 4 sources officielles (41 pays uniques, mai 2026)
  - ONU Security Council sanctions (14 pays)
  - OFAC sanctions programs incluant Ethiopia EO 14046 (8 ajouts)
  - GAFI / FATF black + grey list (plénière 13 février 2026 : 3 + 22)
  - UE Règlement délégué 2016/1675 amendé par 2025/1184, 2026/46, 2026/83
  - + liste interne enrichie sur historique SAML-D
- **Payment types à risque** : Cash Deposit / Cash Withdrawal / Cross-border + observations internes
- **Smurfing score** : nb de senders distincts en petits montants par receiver
- **Fan-out score** : nb de receivers distincts en petits montants par sender
- **Graph features (NetworkX)** : PageRank pondéré, in / out degree (anti-leakage : graphe construit sur train uniquement)

### Modèle ML final

- **LightGBM tuné** (`max_depth=5, learning_rate=0.1, min_child_samples=10`) + calibration sigmoid (Platt scaling, cv=3)
- **Account Autoencoder (AE)** : MLP PyTorch per-compte (425k comptes) → 8-dim latent → 16 features (8 sender + 8 receiver)
- Total : **34 features**
- Méthodologie tuning : TimeSeriesSplit + optimisation Average Precision
- **Split 3 segments** (anti-bias méthodologique) : train_inner (64 %) / val held-out (16 %) / test (20 %). Le seuil compliance est calibré sur val held-out puis appliqué tel quel sur test — pas de seuil choisi sur le test set (défendable face ACPR / Tracfin).
- Sélection winner sur **CV mean_ap** (pas sur le test → anti-peeking strict)
- Explicabilité : SHAP KernelExplainer sur le modèle calibré final

### Baseline rules-based (7 règles)

Tous les seuils sont des constantes business — indépendants du train, **audit-friendly** devant ACPR / Tracfin :

| Règle | Condition | Justification |
|---|---|---|
| **R1** | `Amount > 10 000` | Seuil type Tracfin (vigilance renforcée) |
| **R2** | `Amount > 1 000` ET pays ∈ liste UNION (41 pays) | Sanctions internationales + liste interne |
| **R3** | `Amount > 1 000` ET payment_type ∈ liste UNION | Doctrine AML + observations internes |
| **R4** | `sender_tx_count > 30` (all-time) | Hyperactivité chronique sender |
| **R5** | `receiver_tx_count > 30` (all-time) | Symétrique R4 |
| **R6** | `receiver_smurfing_score > 5` | 5+ senders distincts en petits montants = typologie smurfing |
| **R7** | `Amount > 1 000` ET `sender_tx_count <= 3` | Cold start : compte sender peu connu → vigilance KYC renforcée |

Combinaison OR : alerte si une condition est vérifiée.

> Note : deux règles velocity glissante (burst 24h, velocity 28j) ont été testées puis retirées car SAML-D ne simule pas ces patterns (0 vrai positif). À réactiver en production sur données MSB réelles. Voir [`docs/backlog-after-simplification.md`](docs/backlog-after-simplification.md).

## Résultats

Test set : 160 000 transactions, 210 laundering, 9.3 semaines.

> **Lecture en une phrase** : le rule-based **plafonne à 60 %** de recall — quoi qu'on fasse avec les 7 règles métier, on rate 40 % des cas suspects. Le ML calibré (seuil figé sur validation held-out) atteint **86.7 % de recall sur test** avec **2.2 × moins d'alertes** que le rule-based — c'est l'apport central du projet.

| Système | Alertes / sem | Recall | Précision |
|---|---|---|---|
| Rule-based (7 règles) | 9 682 | 60.0 % | 0.14 % |
| **ML à volume égal** | 9 682 | **86.7 %** | 0.20 % |
| ML à recall égal (60.0 %) | 18 | 60.0 % | — |
| **ML cible compliance (seuil figé val)** | **4 465** | **86.7 %** | — |

**Lecture compliance** :

1. À volume d'alertes équivalent (~9 680 / sem), le ML détecte **1.4 × plus de cas suspects** que le rule-based (86.7 % vs 60.0 %).
2. Pour atteindre le recall plafond du rule-based (60.0 %), le ML n'a besoin que de **18 alertes / semaine** au lieu de 9 682 — soit **538 × moins** de volume.
3. Le rule-based **ne peut pas dépasser** 60.0 % de recall. Le ML calibré atteint 86.7 % de recall (seuil compliance figé sur 51 jours de validation held-out, recall mesuré sur test) avec 4 465 alertes / semaine — soit **+27 points** de recall à volume **2.2 × moindre**. La cible était 80 % en val : le test dépasse légèrement (86.7 %) car distribution proche mais pas identique, sans biais de seuil choisi sur test.

## Reproductibilité

**Option 1 : environnement local**

```bash
# Pré-requis : Python 3.11+, dataset SAML-D dans ../SAML-D_sample_800k.csv
pip install -r requirements.txt
cp .env.example .env  # puis renseigner les variables
jupyter nbconvert --to notebook --execute --inplace feature_engineering_aml.ipynb
```

**Option 2 : sandbox Docker (recommandé pour production)**

```bash
# Build de l'image (aucune donnée embarquée)
docker build -t aml-feature-engineering .

# Run avec volume mount sur le dataset local (jamais dans l'image)
docker run --rm \
    -v /local/path/SAML-D_sample_800k.csv:/data/SAML-D_sample_800k.csv:ro \
    -v $(pwd)/logs:/app/logs \
    --env-file .env \
    aml-feature-engineering
```

Temps d'exécution complet : ~15-30 min sur 800k transactions (CPU, sans GPU requis).

**Anonymisation préalable** (recommandé avant tout partage / collaboration) :

```bash
export ANONYMIZATION_SALT=$(python -c "import secrets; print(secrets.token_hex(32))")
python scripts/anonymize.py SAML-D_sample_800k.csv SAML-D_anonymized.csv
```

## Limitations explicites

- **Dataset synthétique** : les ratios ML / rule-based sont probablement gonflés vs un environnement bancaire réel. SAML-D ne simule pas tous les patterns réels (notamment pas les bursts velocity 24h ni les velocity mensuelles).
- **Pas de features KYC** : âge du compte, profil client, données déclaratives — un système production réel les inclurait.
- **Pas d'historique multi-année** : 8.5 mois de train, fenêtre limitée pour calibrer la velocity long terme.

## Roadmap & implémentation

### A. Passage à l'échelle — ⏳ À venir
- Validation croisée sur un **nouvel échantillon** (800k lignes distinctes, période différente) pour tester la généralisation
- **Entraînement complet sur 9M lignes** avec un hold-out temporel
- *Nécessite le dataset complet SAML-D 9M (à télécharger).*

### B. Résolution du Cold Start — ✅ Implémenté
- Features `is_new_account_sender` / `is_new_account_receiver` (seuil ≤ 3 tx en historique train) — cellule 7 du notebook
- **Règle compliance R7** : Amount > 1 000 ET sender peu connu → alerte vigilance KYC renforcée
- Impact mesuré : recall rule-based passe de 33.3 % à **60.0 %** (+27 points)

### C. Interprétabilité & conformité (XAI) — ✅ Implémenté
- **SHAP par alerte** (cellule 53) : pour chaque alerte ML, top 5 features explicatives avec valeur, contribution SHAP et direction (pousse vers ALERTE / NORMAL)
- 3 cas types analysés : vrai positif, faux positif, borderline
- Répond à l'**article 22 RGPD** (droit à l'explication d'une décision automatisée) et aux exigences **ACPR / Tracfin** d'auditabilité des modèles

### D. Segmentation par typologies — ✅ Implémenté
- Stage 4.3 (cellule 63) : recall mesuré **par typologie LCB-FT** (17 catégories SAML-D) au seuil compliance (figé sur val held-out)
- Identification des **zones aveugles** du ML (Behavioural_Change : 0 % recall — nécessite KYC dynamique) et des **forces** (Smurfing, Cash_Withdrawal, Layered_Fan_Out : 100 % recall)
- Permet à l'analyste compliance de prioriser l'investigation

### E. Compliance by design (sécurité des données) — ✅ Implémenté
- **Anonymisation / pseudonymisation** : `scripts/anonymize.py` (SHA-256 + salt secret) sur les colonnes `Sender_account` / `Receiver_account` — conforme RGPD art. 4(5) et 32
- **Environnement sandbox Docker** : `Dockerfile` avec utilisateur non-root, dépendances pinnées, **aucune donnée dans l'image** (volume mount au runtime)
- **Audit dépendances Python** : GitHub Actions (`.github/workflows/audit.yml`) lance `safety` (CVE) + `bandit` (code) sur push / PR / hebdomadaire
- **Gestion stricte des secrets** : `.env` exclu du git via `.gitignore`, template `.env.example` fourni
- **Traçabilité** : variables `LOG_LEVEL` / `LOG_DIR` pour exportation structurée des journaux d'inférence

### F. Tests de robustesse — rééchantillonnage (finding méthodologique) — ✅ Documenté

Question testée : `SMOTE` ou `RandomOversampler` peuvent-ils battre `class_weight='balanced'` sur le pipeline final ?

**Méthodologie anti-leakage stricte** : `imblearn.Pipeline(SMOTE, LGBM)` wrappé dans `CalibratedClassifierCV(method='sigmoid', cv=3)` — SMOTE appliqué uniquement sur le train de chaque split interne, calibration apprise sur le held-out non resamplé, seuils interprétables comme probabilité calibrée sur la vraie distribution 0.1%.

**Résultat à deux niveaux** :

| Niveau | Métrique | baseline `class_weight` | SMOTE | Δ |
|---|---|---|---|---|
| LGBM seul (18 features) | AP test | 0.290 | 0.330 | **+14 %** |
| **LGBM + AE (34 features)** | AP test | **0.6055** | 0.5097 | **−16 %** |

**Lecture** : SMOTE apporte un gain réel sur LGBM seul mais devient contre-productif une fois les 16 embeddings autoencoder ajoutés. Hypothèse : l'interpolation linéaire de SMOTE est incompatible avec l'embedding non-linéaire de l'autoencoder (positifs synthétiques produits hors-manifold dans l'espace 34-dim). Le seuil calibré SMOTE chute à 0.002 vs 0.063 baseline, confirmant l'écrasement des probabilités.

**Décision** : `class_weight='balanced'` conservé sur le pipeline final LGBM+AE. SMOTE documenté comme finding négatif méthodologique — voir [`docs/backlog-after-simplification.md`](docs/backlog-after-simplification.md#stage-24--smote-sur-lgbmae-testé-contre-productif-sur-pipeline-final) et cellules 47 + 56 du notebook.

**Leçon générale** : l'efficacité de SMOTE dépend fortement de la nature des features. Sur des features riches et non-linéaires (embeddings, graphe), l'interpolation linéaire devient contre-productive. Évidence à mentionner si interrogé sur la gestion du déséquilibre.

## Stack technique

Python · pandas · scikit-learn · LightGBM · XGBoost · imbalanced-learn · PyTorch · NetworkX · SHAP · matplotlib · seaborn

## Structure du dépôt

```
.
├── feature_engineering_aml.ipynb       # Notebook principal (67 cellules)
├── requirements.txt                    # Dépendances Python pinnées
├── Dockerfile                          # Sandbox reproductible (non-root, sans données)
├── .dockerignore                       # Exclut données et secrets de l'image
├── .env.example                        # Template variables d'environnement (à copier en .env)
├── .gitignore                          # Exclut *.csv, .env, logs, etc.
├── scripts/
│   └── anonymize.py                    # Pseudonymisation SHA-256 + salt (RGPD)
├── .github/
│   └── workflows/
│       └── audit.yml                   # CI safety (CVE) + bandit (code) hebdomadaire
├── docs/
│   ├── backlog-after-simplification.md # Suivi des décisions techniques
│   ├── specs/                          # Specs de design historiques
│   └── plans/                          # Plans d'exécution historiques
└── README.md
```
