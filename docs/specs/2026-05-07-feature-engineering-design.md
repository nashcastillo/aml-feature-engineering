# Feature Engineering Comportemental — AML Detection

**Date :** 2026-05-07
**Projet :** aml-feature-engineering (amelioration du projet final Jedha)
**Equipe :** Nashely Castillo

---

## 1. Contexte

Le projet final Jedha utilise un Decision Tree (max_depth=10) avec 6 features pour detecter les transactions suspectes dans le dataset SAML-D (800k lignes, 0.1% suspectes). Le recall actuel est de 43%.

L'objectif est d'ameliorer le recall en ajoutant des features comportementales basees sur les patterns des comptes.

## 2. Question

> Les features comportementales (frequence, reseau, montant, profil de risque) permettent-elles d'ameliorer le recall du modele de detection AML ?

## 3. Features a creer

### 3.1 Frequence (confirme par EDA existant)

| Feature | Formule | Justification metier |
|---|---|---|
| `sender_tx_count` | Nb de transactions par expediteur | Activite du compte (deja existant) |
| `receiver_tx_count` | Nb de transactions par destinataire | Activite du compte (deja existant) |

### 3.2 Reseau — RETIRE apres verification

> Pattern One-to-Many et Many-to-One verifie sur le dataset :
> - sender_unique_receivers : normales=15, suspectes=8.3 (inverse de l'hypothese)
> - receiver_unique_senders : normales=3.9, suspectes=4.0 (aucune difference)
> Conclusion : non discriminant dans ce dataset synthetique. Features retirees.

### 3.3 Montant — RETIRE apres verification

> Le seuil > 50K ne concerne que 2.3% des transactions suspectes (19 sur 833).
> Le seuil < 5K cree un biais (18.7% des transactions normales aussi).
> Le montant est deja capture par la feature `Amount` du modele original.

### 3.4 Temporel — RETIRE apres verification

> **Jour de la semaine :** distribution quasi identique tous les jours, ecarts minuscules.
>
> **Heure :** l'ecart existe mais il est trop faible en valeur absolue pour etre discriminant.
>
> Features retirees.

### 3.5 Geographique (confirme par EDA — pays a risque)

| Feature | Formule | Justification metier |
|---|---|---|
| `is_sender_risky_country` | 1 si Sender_bank_location dans pays a risque, 0 sinon | Pays d'emission a risque |
| `is_receiver_risky_country` | 1 si Receiver_bank_location dans pays a risque, 0 sinon | Pays de destination a risque |

**Pays a risque (identifies dans l'EDA, seuil base sur le taux global de 0.104%) :**
- Emission (seuil 3x = > 0.31%) : Albania, Italy, Netherlands
- Reception (seuil 5x = > 0.52%) : Nigeria, Albania, Morocco, Mexico

### 3.6 Pattern combine — Smurfing (confirme par analyse croisee)

| Feature | Formule | Justification metier |
|---|---|---|
| `receiver_smurfing_score` | Nb de transactions recues par le destinataire avec montant < 5K ET provenant de multiples expediteurs (unique_senders > mediane) | Smurfing : petit montant + beaucoup d'expediteurs = 0.644% suspect (6x la moyenne). Pattern non visible separement, mais confirme en combinant montant + reseau |

> Note : le pattern reseau seul et le petit montant seul ne sont pas discriminants.
> Mais la combinaison (petit montant + beaucoup d'expediteurs) produit un taux suspect
> de 0.644% vs 0.042% pour les petits montants avec peu d'expediteurs.

### 3.7 Profil de risque (proxy comportemental — pas de data leakage)

| Feature | Formule | Justification metier |
|---|---|---|
| `sender_risky_payment_count` | Nb de transactions a risque du compte expediteur | Nb de paiements a risque cumules |
| `receiver_risky_payment_count` | Nb de transactions a risque du compte destinataire | Nb de paiements a risque cumules |

**Definition "transaction a risque" (proxy, sans utiliser Is_laundering) :**
- Payment_type = Cash Deposit, Cash Withdrawal ou Cross-border (taux suspect > 2x la moyenne, confirme par EDA : 0.63%, 0.47%, 0.28% vs moyenne 0.104%)

Le score = nombre de ces criteres remplis par les transactions passees du compte.

## 4. Methodologie

### Etape 1 — EDA exploratoire temporelle
- Extraire `hour` et `day_of_week`
- Comparer les distributions entre transactions normales et suspectes
- Decider si ces features sont pertinentes

### Etape 2 — Construction des features
- Split train/test AVANT le calcul des features
- Calculer les agregats (`groupby`) sur le train set uniquement
- Appliquer les valeurs au test set par lookup (`map`/`merge`)
- Les comptes absents du train recevront la valeur mediane (ou 0)
- Pas de StandardScaler pour les arbres de decision

### Etape 3 — Re-entrainement du modele
- Meme pipeline : train/test 80/20, stratify=y, class_weight='balanced'
- Decision Tree (max_depth=10) avec les nouvelles features
- Comparer recall avant/apres

### Etape 4 — Analyse des resultats
- Matrice de confusion
- Importance des features (les nouvelles sont-elles utiles ?)
- Comparaison train vs test (sur-apprentissage)

## 5. Contraintes

- **Simplicite** : chaque feature doit etre explicable en 1 phrase lors de la soutenance Jedha
- **Pas de data leakage** : aucune feature n'utilise Is_laundering comme input
- **Performance** : doit tourner sur 800k lignes sur un Mac en temps raisonnable
- **Reproductibilite** : random_state=42, meme split que le projet original

## 6. Regard metier — Limites du dataset

### Pattern One-to-Many / Many-to-One

En tant que Compliance Officer LCB-FT, les patterns de reseau sont des indicateurs fondamentaux en AML :
- **One-to-Many** (fan-out) : un compte qui distribue des fonds vers de nombreux destinataires est un signal classique de blanchiment
- **Many-to-One** (smurfing) : un compte collecteur qui recoit de nombreux expediteurs differents est un signal de structuration

Ces patterns ont ete verifies sur le dataset SAML-D :
- `sender_unique_receivers` : normales = 15 destinataires en moyenne, suspectes = 8.3 (inverse de l'hypothese)
- `receiver_unique_senders` : normales = 3.9, suspectes = 4.0 (aucune difference significative)

**Conclusion :** le dataset synthetique SAML-D ne reproduit pas ces patterns pourtant essentiels en AML. C'est une limite du dataset. Dans un environnement reel avec des donnees bancaires, ces features seraient probablement parmi les plus discriminantes. En l'absence de signal dans les donnees, ces features n'ont pas ete incluses dans le modele — ajouter des variables non discriminantes risquerait d'introduire du bruit et de degrader les performances.

### Pattern Smurfing — Signal confirme par analyse croisee

Les patterns de reseau (one-to-many, many-to-one) et les petits montants, pris separement, ne sont pas discriminants dans ce dataset. Cependant, en croisant les deux criteres, le signal apparait clairement :

| Combinaison (cote destinataire) | Taux suspect |
|---|---|
| Petit montant (< 5K) + Peu d'expediteurs | 0.042% |
| **Petit montant (< 5K) + Beaucoup d'expediteurs** | **0.644%** (6x la moyenne) |
| Gros montant (>= 5K) + Beaucoup d'expediteurs | 0.155% |

C'est exactement la definition du smurfing : plusieurs personnes deposent de petites sommes sur un meme compte collecteur pour eviter les seuils de declaration. Ce pattern n'est visible qu'en combinant montant et reseau — c'est pourquoi la feature `receiver_smurfing_score` a ete creee.

Cette approche illustre l'importance du regard metier : un data scientist sans connaissance AML aurait retire les variables reseau apres l'analyse univariee. L'expertise Compliance permet de chercher les bons croisements.

### Paire unique sender-receiver — Feature ecartee par expertise metier

L'analyse data a revele que les transactions entre deux comptes sans historique commun (paire unique) ont un taux suspect de 0.241% (2.3x la moyenne), concentre sur la tranche 2K-5K. Cependant, les transactions ponctuelles de 2K-5K sont un comportement habituel et legitime dans les transferts internationaux. Les clients occasionnels (lignes directrices ACPR/TRACFIN, art. L. 561-10-2) effectuent regulierement des transmissions de fonds ponctuelles — par exemple un particulier qui envoie de l'argent a sa famille a l'etranger. Inclure cette feature genererait un volume massif de faux positifs sur des operations legitimes.

### Seuil de montant — Pourquoi un seuil fixe n'est pas adapte

L'analyse du dataset montre que le seuil > 100K ne concerne que 2.3% des transactions suspectes (19 sur 833). Un seuil fixe n'est donc pas adapte pour ce dataset.

Ce constat est coherent avec les lignes directrices ACPR/TRACFIN : la classification des risques doit etre adaptee aux activites et aux clienteles de l'etablissement. Un virement de 100K peut etre normal pour une entreprise d'import-export, mais suspect pour un particulier salarie. En production, un seuil dynamique et individualise par profil client serait plus pertinent.

## 7. Critere de succes

- Recall test > 43% (amelioration par rapport au modele actuel)
- Les nouvelles features apparaissent dans le top d'importance
- Chaque feature a une justification metier claire
