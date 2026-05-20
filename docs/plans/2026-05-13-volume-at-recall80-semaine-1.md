# Volume@Recall80 — Semaine 1 : Refocus méthodologique

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrer la méthodologie de tuning vers TimeSeriesSplit + optimisation par AP, et ajouter la métrique opérationnelle `volume_at_recall_80` pour mesurer le tradeoff recall/volume.

**Architecture:** Modifier la fonction `manual_cv_tuning` (cellule 29, id `aje4j41cr8u`) pour utiliser `TimeSeriesSplit` au lieu de `StratifiedKFold` et optimiser `average_precision_score` au lieu de `recall_score`. Re-exécuter les 3 cellules de tuning existantes. Ajouter une nouvelle cellule de mesure `volume_at_recall_80` après le tableau seuils.

**Tech Stack:** Python, pandas, scikit-learn (`TimeSeriesSplit`, `average_precision_score`), Jupyter Notebook.

**Spec source :** `docs/specs/2026-05-13-volume-at-recall80-design.md`

---

## File Structure

**Fichier unique modifié** : `feature_engineering_aml.ipynb`

| Cellule | ID | Modification |
|---|---|---|
| Nouvelle cellule (avant cellule 29) | _nouvelle_ | Diagnostic : compter positifs par fold TimeSeriesSplit |
| Cellule 29 | `aje4j41cr8u` | Modifier `manual_cv_tuning` : TimeSeriesSplit + AP |
| Cellule 30 | `qffz1n3bnb8` | Re-exécuter tuning DT (pas de changement code) |
| Cellule 31 | `moylrnegqwa` | Re-exécuter tuning RF (pas de changement code) |
| Cellule 32 | `vwi3mom3ld` | Re-exécuter tuning XGB (pas de changement code) |
| Cellule 35 | `py4hmqgit1l` | Re-exécuter réentraînement avec nouveaux best_params |
| Nouvelle cellule (après cellule 41) | _nouvelle_ | Métrique `volume_at_recall_80` + récap S1 |

**Note importante** : ce projet n'est pas en git. Au lieu de `git commit`, chaque tâche se termine par "sauvegarder le notebook dans Jupyter (Ctrl+S)".

**Note pour le test set** : il couvre du **2023-06-19 au 2023-08-23** ≈ **66 jours ≈ 9.43 semaines**. Cette valeur sert au calcul de `volume_at_recall_80` en alertes/semaine.

---

## Task 1 : Diagnostic préalable TimeSeriesSplit

**Objectif** : Avant de remplacer `StratifiedKFold` par `TimeSeriesSplit`, vérifier que chaque fold de validation contient suffisamment de positifs (cible ≥ 30) pour que le tuning soit stable.

**Files:**
- Modify: `feature_engineering_aml.ipynb` (ajouter une cellule juste avant cellule 29, id `aje4j41cr8u`)

- [ ] **Step 1.1 : Insérer une nouvelle cellule code juste avant la définition de `manual_cv_tuning`**

Code à coller dans la nouvelle cellule :

```python
# --- Diagnostic prealable : positifs par fold TimeSeriesSplit ---
# On verifie que chaque fold de validation a assez de positifs (cible >= 30)
# avant de basculer manual_cv_tuning sur TimeSeriesSplit.
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=3)
y_train_check = df_train['Is_laundering']

print("=== Diagnostic TimeSeriesSplit (n_splits=3) sur df_train ===")
print(f"Total positifs train : {y_train_check.sum()}")
print()
print(f"{'Fold':>5} {'Train size':>12} {'Val size':>10} {'Train pos':>10} {'Val pos':>9}")
print("-" * 50)

min_val_pos = float('inf')
for i, (train_idx, val_idx) in enumerate(tscv.split(df_train), 1):
    n_train = len(train_idx)
    n_val = len(val_idx)
    train_pos = y_train_check.iloc[train_idx].sum()
    val_pos = y_train_check.iloc[val_idx].sum()
    min_val_pos = min(min_val_pos, val_pos)
    print(f"{i:>5d} {n_train:>12,} {n_val:>10,} {train_pos:>10d} {val_pos:>9d}")

print()
if min_val_pos >= 30:
    print(f"OK : min positifs/val = {min_val_pos} >= 30. On peut migrer vers TimeSeriesSplit.")
else:
    print(f"ATTENTION : min positifs/val = {min_val_pos} < 30. Risque d'instabilite.")
    print("Decision : soit on conserve StratifiedKFold, soit on accepte le risque (a documenter).")
```

- [ ] **Step 1.2 : Exécuter la cellule**

Dans Jupyter : Shift+Enter sur la cellule. Vérifier que la sortie affiche un tableau de 3 lignes avec les comptes par fold.

- [ ] **Step 1.3 : Décision de gating**

Lire la dernière ligne de la sortie :
- Si **"OK : min positifs/val >= 30"** → continuer à Task 2.
- Si **"ATTENTION : min positifs/val < 30"** → STOP, en parler à l'utilisatrice. Options possibles :
  - Réduire `n_splits` à 2
  - Conserver `StratifiedKFold` et documenter le compromis dans le notebook
  - Accepter le risque et documenter

- [ ] **Step 1.4 : Sauvegarde**

Dans Jupyter : Ctrl+S (sauvegarder le notebook).

---

## Task 2 : Migrer `manual_cv_tuning` vers TimeSeriesSplit + AP

**Objectif** : Modifier la fonction `manual_cv_tuning` pour utiliser `TimeSeriesSplit` et optimiser `average_precision_score` au lieu de `recall_score`.

**Files:**
- Modify: `feature_engineering_aml.ipynb` cellule 29 (id `aje4j41cr8u`)

- [ ] **Step 2.1 : Vérifier l'import de `average_precision_score`**

Chercher en haut du notebook (cellule 1, id `c42a7001`) si l'import existe déjà. La ligne attendue :

```python
from sklearn.metrics import recall_score, precision_score, average_precision_score, ...
```

Si `average_precision_score` est absent, l'ajouter à l'import existant. Sinon ne rien faire.

- [ ] **Step 2.2 : Remplacer le contenu complet de la cellule 29 (id `aje4j41cr8u`)**

Ancien contenu (à remplacer entièrement) :

```python
# --- Fonction de tuning manuel avec features recalculees par fold ---
from itertools import product

def manual_cv_tuning(df_source, model_class, param_grid, fixed_params, n_splits=3):
    """Teste toutes les combinaisons de param_grid en CV. Optimise le recall.
    Retourne (best_params, best_score, df_results)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_source = df_source['Is_laundering']
    ...
```

Nouveau contenu complet de la cellule :

```python
# --- Fonction de tuning manuel avec features recalculees par fold ---
# Methodologie alignee avec le split train/test temporel :
# - TimeSeriesSplit (forward chaining) au lieu de StratifiedKFold random
# - Optimisation par average_precision_score (AP) au lieu de recall@0.5
#   -> AP capture toute la courbe PR, donc le meilleur tradeoff recall/volume
from itertools import product
from sklearn.model_selection import TimeSeriesSplit

def manual_cv_tuning(df_source, model_class, param_grid, fixed_params, n_splits=3):
    """Teste toutes les combinaisons de param_grid en CV temporelle. Optimise l'AP.
    Retourne (best_params, best_score_ap, df_results).
    df_source doit etre deja trie par date (chrono croissant)."""
    cv = TimeSeriesSplit(n_splits=n_splits)

    combinations = list(product(*param_grid.values()))
    print(f"Testing {len(combinations)} combinations x {n_splits} folds = {len(combinations) * n_splits} fits\n")

    all_results = []
    best_score = -1
    best_params = None

    for i, combo in enumerate(combinations):
        params = dict(zip(param_grid.keys(), combo))

        fold_ap, fold_recall = [], []
        for train_idx, test_idx in cv.split(df_source):
            df_t = df_source.iloc[train_idx].copy()
            df_te = df_source.iloc[test_idx].copy()
            df_t, df_te = build_features_on_fold(df_t, df_te)

            X_tr, X_te = df_t[all_features], df_te[all_features]
            y_tr, y_te = df_t['Is_laundering'], df_te['Is_laundering']

            model = model_class(**fixed_params, **params)
            model.fit(X_tr, y_tr)
            y_proba = model.predict_proba(X_te)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            fold_ap.append(average_precision_score(y_te, y_proba))
            fold_recall.append(recall_score(y_te, y_pred))

        mean_ap = np.mean(fold_ap)
        mean_recall = np.mean(fold_recall)
        all_results.append({**params, 'mean_ap': mean_ap, 'std_ap': np.std(fold_ap),
                            'mean_recall_at_0.5': mean_recall})

        if mean_ap > best_score:
            best_score = mean_ap
            best_params = params

        print(f"  [{i+1}/{len(combinations)}] {params} — AP={mean_ap:.4f}  recall@0.5={mean_recall:.3f}")

    print(f"\n  >>> Best: AP={best_score:.4f} | {best_params}")
    return best_params, best_score, pd.DataFrame(all_results)


print("Fonction manual_cv_tuning prete (TimeSeriesSplit + optimisation AP).")
```

- [ ] **Step 2.3 : Exécuter la cellule modifiée**

Shift+Enter sur la cellule 29. Sortie attendue : `Fonction manual_cv_tuning prete (TimeSeriesSplit + optimisation AP).`

- [ ] **Step 2.4 : Sauvegarder le notebook (Ctrl+S)**

---

## Task 3 : Mettre à jour les print labels + re-exécuter les 3 cellules de tuning

**Objectif** : Relancer les 3 tunings (DT, RF, XGB) avec TimeSeriesSplit + AP. Les cellules contiennent un print `"Meilleur recall CV : ..."` à corriger en `"Meilleur AP CV : ..."` puisque la métrique a changé.

**Files:**
- Modify (texte du print uniquement) + re-exécuter : `feature_engineering_aml.ipynb` cellules 30 (DT), 31 (RF), 32 (XGB)

- [ ] **Step 3.0 : Corriger le print label dans les 3 cellules de tuning**

Dans chacune des 3 cellules (30, 31, 32), remplacer :

```python
print(f"\nMeilleurs parametres : {dt_best_params}")
print(f"Meilleur recall CV   : {dt_best_score:.3f}")
```

par :

```python
print(f"\nMeilleurs parametres : {dt_best_params}")
print(f"Meilleur AP CV       : {dt_best_score:.4f}")
```

Adapter le préfixe pour chaque modèle :
- Cellule 30 (DT) : `dt_best_params`, `dt_best_score`
- Cellule 31 (RF) : `rf_best_params`, `rf_best_score`
- Cellule 32 (XGB) : `xgb_best_params`, `xgb_best_score`

- [ ] **Step 3.1 : Re-exécuter la cellule 30 (Tuning DT)**

Shift+Enter. La sortie doit maintenant montrer des lignes du type `[i/9] {...} — AP=0.0xxx  recall@0.5=0.xxx`. Noter le `Best: AP=` et les `Best params`.

**Vérification de cohérence** : l'AP doit être ≥ 0.02 environ. Si AP < 0.01, c'est un signal d'erreur (probablement la fonction n'a pas été ré-importée).

- [ ] **Step 3.2 : Re-exécuter la cellule 31 (Tuning RF)**

Shift+Enter. Même format de sortie. Noter le `Best`.

**Note** : cette cellule est la plus longue (12 combos × 3 folds × n_estimators=200 = ~10-15 min). Patience.

- [ ] **Step 3.3 : Re-exécuter la cellule 32 (Tuning XGB)**

Shift+Enter. Même format. Noter le `Best`.

**Attente** : XGB peut récupérer ou améliorer son AP par rapport à 0.025 (la valeur précédente était dégradée par le tuning recall@0.5 qui choisissait `min_child_weight=5` au lieu de 1).

- [ ] **Step 3.4 : Noter les 3 résultats**

Dans un coin du notebook (cellule markdown ou commentaire), reporter :

```
Tuning S1 (TimeSeriesSplit + AP):
- DT  TUNE : AP=0.0xxx, best_params={...}
- RF  TUNE : AP=0.0xxx, best_params={...}
- XGB TUNE : AP=0.0xxx, best_params={...}
```

- [ ] **Step 3.5 : Sauvegarder le notebook (Ctrl+S)**

---

## Task 4 : Re-exécuter le réentraînement final avec les nouveaux best_params

**Objectif** : La cellule 35 (id `py4hmqgit1l`) ré-entraîne les 3 modèles finaux sur l'ensemble du `df_train` avec les `best_params` du tuning. Puisque les `best_params` ont changé après Task 3, il faut tout ré-exécuter.

**Files:**
- Modify (re-exécution uniquement) : `feature_engineering_aml.ipynb` cellule 35 (id `py4hmqgit1l`), puis toutes les cellules en aval qui dépendent des modèles ré-entraînés (cellules 36 à 41).

- [ ] **Step 4.1 : Re-exécuter la cellule 35 (Réentraînement)**

Shift+Enter.

- [ ] **Step 4.2 : Re-exécuter les cellules 36 à 41 dans l'ordre**

Dans Jupyter : sélectionner cellule 36, puis Shift+Enter répété jusqu'à la cellule 41 (Tableau seuils XGB TUNE).

Alternativement : menu **Cell → Run All Below** depuis la cellule 35.

- [ ] **Step 4.3 : Vérifier la cohérence du tableau seuils XGB TUNE (cellule 41)**

La sortie doit montrer un tableau seuils complet. Vérifier que le recall à seuil 0.5 est cohérent avec ce qui sort du `classification_report`.

- [ ] **Step 4.4 : Sauvegarder le notebook (Ctrl+S)**

---

## Task 5 : Ajouter la métrique `volume_at_recall_80`

**Objectif** : Ajouter une nouvelle cellule (juste après cellule 41) qui calcule **pour chaque modèle tuné** le volume d'alertes nécessaires pour atteindre recall = 80% sur test set, exprimé en alertes/semaine.

**Files:**
- Modify: `feature_engineering_aml.ipynb` (ajouter une nouvelle cellule juste après cellule 41, id `7eda54b5`)

- [ ] **Step 5.1 : Insérer une nouvelle cellule code juste après la cellule "Tableau seuils XGB TUNE"**

Code à coller :

```python
# --- Metrique operationnelle : volume_at_recall_80 par modele ---
# Question metier : pour atteindre recall=80% sur test, combien d'alertes/sem mon equipe doit-elle traiter ?
# Test set couvre du 2023-06-19 au 2023-08-23 = 66 jours ~= 9.43 semaines
N_SEMAINES_TEST = 9.43

def volume_at_recall(y_true, y_proba, recall_cible=0.80):
    """Retourne (volume_total, seuil_utilise, recall_reel, precision_reelle).
    Trouve le seuil minimal qui atteint recall >= recall_cible.
    Si impossible, retourne le seuil 0 (alerte sur tout)."""
    n_pos = y_true.sum()
    tp_cible = int(np.ceil(recall_cible * n_pos))
    # On trie les scores par ordre decroissant, on prend les k plus eleves
    # jusqu'a avoir tp_cible vrais positifs.
    order = np.argsort(-y_proba)
    y_sorted = y_true.iloc[order].values if hasattr(y_true, 'iloc') else y_true[order]
    cumulative_tp = np.cumsum(y_sorted)
    idx_atteint = np.searchsorted(cumulative_tp, tp_cible)
    if idx_atteint >= len(y_sorted):
        # On a pas atteint le recall meme en alertant sur tout
        seuil = 0.0
        volume = len(y_sorted)
    else:
        seuil = y_proba[order[idx_atteint]]
        volume = idx_atteint + 1
    tp = cumulative_tp[min(idx_atteint, len(y_sorted)-1)]
    recall_reel = tp / n_pos
    precision_reelle = tp / volume if volume > 0 else 0
    return volume, seuil, recall_reel, precision_reelle


# Calcul pour les 3 modeles tunes
print("=== volume_at_recall_80 par modele (recall cible = 80% sur test) ===\n")
print(f"{'Modele':<10s} {'Volume':>10s} {'Volume/sem':>12s} {'Seuil':>8s} {'Recall reel':>12s} {'Precision':>11s}")
print("-" * 70)

for nom, modele in [("DT TUNE", dt_tuned), ("RF TUNE", rf_tuned), ("XGB TUNE", xgb_tuned)]:
    y_proba = modele.predict_proba(X_test_new)[:, 1]
    vol, seuil, recall_r, prec_r = volume_at_recall(y_test_new, y_proba, recall_cible=0.80)
    vol_sem = vol / N_SEMAINES_TEST
    print(f"{nom:<10s} {vol:>10,d} {vol_sem:>12,.0f} {seuil:>8.3f} {recall_r:>12.1%} {prec_r:>11.2%}")

print()
print("Lecture : a recall 80% sur test, RF TUNE genere X alertes au total = Y alertes/sem.")
print("C'est la metrique operationnelle principale du projet.")
```

- [ ] **Step 5.2 : Exécuter la cellule**

Shift+Enter. Sortie attendue : tableau avec 3 lignes (DT TUNE, RF TUNE, XGB TUNE), chacune avec son volume et volume/sem à recall 80%.

**Sanity check** : 
- Pour le RF TUNE, le recall réel doit être ≥ 0.80 (par construction).
- Volume/sem doit être réaliste : entre 100 et 10 000.
- Si volume = total transactions test (~160K), c'est que le modèle ne peut pas atteindre recall 80% même en alertant sur tout → signaler à l'utilisatrice.

- [ ] **Step 5.3 : Ajouter une cellule markdown récapitulative juste après**

Cellule markdown à insérer :

```markdown
## Bilan Semaine 1 — Refocus méthodologique

Changements apportés :
- **CV interne** : `StratifiedKFold` random → `TimeSeriesSplit` (forward chaining, cohérent avec split test temporel)
- **Métrique de tuning** : `recall@0.5` → `average_precision_score` (capture toute la courbe PR)
- **Nouvelle métrique opérationnelle** : `volume_at_recall_80` (alertes/sem pour atteindre 80% de recall)

Cette base méthodologique permet maintenant de comparer les modèles sur **la métrique métier** (volume d'alertes à recall constant) plutôt que sur une métrique statistique abstraite.
```

- [ ] **Step 5.4 : Sauvegarder le notebook (Ctrl+S)**

---

## Task 6 : Vérification de bout en bout

**Objectif** : S'assurer que le notebook s'exécute proprement de haut en bas avec les nouvelles modifications, et que les chiffres sont cohérents.

**Files:**
- Test: `feature_engineering_aml.ipynb` (exécution complète)

- [ ] **Step 6.1 : Restart kernel + Run All**

Dans Jupyter : **Kernel → Restart & Run All**.

Attendre la fin (~20-30 min selon le tuning). Vérifier qu'aucune cellule ne casse.

- [ ] **Step 6.2 : Vérifier les critères de cohérence**

Vérifier dans la sortie :
- [ ] Cellule diagnostic TimeSeriesSplit affiche "OK"
- [ ] 3 cellules tuning affichent `Best: AP=...` (pas `recall=...`)
- [ ] Cellule `volume_at_recall_80` affiche un tableau 3 lignes
- [ ] Au moins UN modèle a un recall réel ≥ 80% dans le tableau volume_at_recall_80

- [ ] **Step 6.3 : Documenter les chiffres clés**

Reporter dans la conversation avec l'utilisatrice les valeurs finales :
- AP par modèle (DT/RF/XGB)
- volume_at_recall_80 par modèle (en alertes/sem)
- Meilleur modèle selon AP
- Meilleur modèle selon volume_at_recall_80 (peut être différent !)

- [ ] **Step 6.4 : Mettre à jour le backlog**

Modifier `docs/backlog-after-simplification.md` :
- Marquer P1.4 (tuner par AP) comme **DONE**
- Ajouter une entrée "S1 méthodo terminée" avec les chiffres ci-dessus
- Mettre à jour P4.13 avec le nouveau modèle gagnant + AP + volume@recall80

- [ ] **Step 6.5 : Sauvegarder le notebook (Ctrl+S)**

---

## Definition of Done — Semaine 1

À l'issue de ce plan :

1. ✓ Fonction `manual_cv_tuning` utilise `TimeSeriesSplit` et optimise par AP
2. ✓ 3 modèles (DT, RF, XGB) re-tunés avec la nouvelle méthodologie
3. ✓ Nouvelle métrique `volume_at_recall_80` calculée pour les 3 modèles
4. ✓ Notebook s'exécute de bout en bout sans erreur (Restart & Run All)
5. ✓ Backlog `docs/backlog-after-simplification.md` mis à jour
6. ✓ Chiffres clés reportés à l'utilisatrice

**Sortie attendue de la semaine** :
> "Mon meilleur modèle (X TUNE) atteint recall 80% à Y alertes/semaine sur test set, avec un AP de Z."

Ces 3 chiffres deviennent la **base de référence S1** contre laquelle on mesurera l'amélioration apportée par LightGBM (S2) et le baseline rule-based (S3).
