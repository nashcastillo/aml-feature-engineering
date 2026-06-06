# Pass de simplification beginner-friendly — Plan d'implémentation

> **Pour l'exécution :** plan interactif. Pour chaque cellule complexe, je propose la version simplifiée dans le chat, l'utilisatrice valide ou pose des questions, j'applique via NotebookEdit, on vérifie l'output dans Jupyter.

**Goal :** Simplifier toutes les cellules "complexes" du notebook (comprehensions, lambdas, chaînes pandas, `**kwargs`, cellules longues) en versions accessibles à un débutant Python, **sans alourdir le code** et idéalement en le raccourcissant.

**Architecture :** parcours du notebook du haut vers le bas. Chaque tâche cible une cellule (ou un groupe de cellules connexes). Les tâches 5 et 12 intègrent en passage les modifs restantes du plan correction-leakage (tasks 3 et 5 de `2026-05-09-fix-risky-lists-leakage.md`).

**Tech Stack :** Jupyter Notebook, pandas, scikit-learn, XGBoost. Notebook : `feature_engineering_aml.ipynb`. Spec : `docs/specs/2026-05-10-simplification-beginner-friendly-design.md`.

**Cellules considérées comme déjà simples (non touchées) :** imports, baseline, splits, encoding, calls sklearn standards, markdowns. ~25 cellules.

**Cellules à simplifier (12 tâches) :** listées ci-dessous dans l'ordre du notebook.

---

## Table des cellules à traiter

| # | Cell ID | Section | Complexité principale |
|---|---|---|---|
| 1 | `51b3a838` | 2 — Smurfing | `groupby().agg(named=tuple)`, multi-statement |
| 2 | `49469cdc` | 2 — Risky Payment Count | + intégration task 3 leakage |
| 3 | `1ajaeytwbf5` | 4 — Train vs Test | `for x, y, z in [tuples]:` × 2 imbriqués |
| 4 | `j88mnm99vd` | 4 — Importance | f-strings imbriquées avec ternaire |
| 5 | `2tsxsbhcv6u` | 4 — Matrices confusion | `for ax, y_t, y_p, title in [(axes[0], ...), ...]` × 4 |
| 6 | `3rzupajshvp` | 4 — Importance plot | comprehension `colors = [... for ... if ... else ...]` |
| 7 | `7xk75du1kzk` | 4 — Tableau recap | f-strings d'alignement répétitifs |
| 8 | `jnb2sya219s` | 5 — CV 5-fold | TRÈS LONG (~80 lignes), `build_features_on_fold` + boucle CV + intégration task 5 leakage |
| 9 | `aje4j41cr8u` | 6 — manual_cv_tuning | `**fixed_params, **params`, `product(*param_values)`, dict comprehension |
| 10 | `lokvziyrhzm` | 6 — Recap tuning | dict comprehension avec filter |
| 11 | `py4hmqgit1l` | 7 — Eval finale | TRÈS LONG (~100 lignes), `lambda` dans `max(key=...)`, multiples boucles |
| 12 | `r2j6u9e1dg` | 8 — PR curve + threshold | `for ax, model, name, color in [tuples]:` + `lambda` dans max |

Note : tâche 8 et 12 du plan original leakage (`2026-05-09-fix-risky-lists-leakage.md`) sont absorbées ici.

---

### Task 1 : Cellule `51b3a838` — Smurfing score

**Complexité :** `groupby('Receiver_account').agg(n_senders=('Sender_account', 'nunique'), n_tx=('Amount', 'count'))` — syntaxe NamedAgg avancée. Variables temporaires recouvrant le même `score_col`.

- [ ] **Step 1 :** Lire la cellule actuelle pour identifier les blocs réellement utilisés vs morts (le calcul `p90`, `conditions`, `masks`, `for label, mask in zip(conditions, masks)` peuvent être conservés avec une syntaxe simplifiée).
- [ ] **Step 2 :** Proposer la version simplifiée à l'utilisatrice :
   - Remplacer `groupby().agg(named=tuple)` par 2 `groupby` séparés (1 ligne par stat, total : 2 lignes au lieu d'une `.agg()` complexe).
   - Remplacer `for label, mask in zip(conditions, masks)` par 3 prints séparés explicites (3 lignes courtes, plus lisible que la boucle).
   - Cible : ~30 lignes → ~25 lignes.
- [ ] **Step 3 :** Appliquer via NotebookEdit après validation.
- [ ] **Step 4 :** Utilisatrice exécute la cellule, vérifie que les stats sont identiques (mean/median/max/P90).

---

### Task 2 : Cellule `49469cdc` — Risky Payment Count (+ leakage fix)

**Complexité :** ligne `RISKY_PAYMENT_TYPES = {'Cash Deposit', 'Cash Withdrawal', 'Cross-border'}` à supprimer (déjà calculé en cellule `0e8b08a1`). Reste de la cellule : simple mais redondance possible.

- [ ] **Step 1 :** Supprimer la ligne `RISKY_PAYMENT_TYPES = {...}` (intègre la task 3 du plan leakage).
- [ ] **Step 2 :** Vérifier que la cellule reste cohérente sans la définition supprimée.
- [ ] **Step 3 :** Appliquer via NotebookEdit, utilisatrice exécute, vérifie output (mean/taux suspect inchangés).

---

### Task 3 : Cellule `1ajaeytwbf5` — Train vs Test (overfitting check)

**Complexité :** double boucle `for model_name, y_tr_pred, y_te_pred in [tuples]: for name, train_val, test_val in [tuples]:` avec construction inline des tuples.

- [ ] **Step 1 :** Proposer la version simplifiée à l'utilisatrice :
   - Remplacer la boucle externe par 3 blocs séquentiels (un par modèle), si la longueur reste raisonnable.
   - OU garder la boucle externe mais sortir les tuples dans une liste nommée explicite (`models_to_check = [...]` sur 4 lignes).
   - Garder la boucle interne sur les 3 métriques (recall/precision/f1) qui est déjà claire.
   - Cible : ~25 lignes → ~20 lignes.
- [ ] **Step 2 :** Appliquer après validation.
- [ ] **Step 3 :** Utilisatrice exécute, vérifie output identique.

---

### Task 4 : Cellule `j88mnm99vd` — Importance des features

**Complexité :** f-string imbriquée avec ternaire `marker = " <-- NEW" if feat in new_features else ""`. Boucle dense.

- [ ] **Step 1 :** Proposer :
   - Sortir le ternaire en `if/else` 2 lignes (plus visible).
   - Garder la structure `for name, importance in [(label, series), ...]:`.
   - Cible : ~12 lignes → ~12 lignes (pas de gain de taille, gain de clarté seulement).
- [ ] **Step 2 :** Appliquer après validation, utilisatrice exécute.

---

### Task 5 : Cellule `2tsxsbhcv6u` — Matrices de confusion

**Complexité :** `for ax, y_true, y_pred, title in [(axes[0], y_test, y_pred_base, "..."), (axes[1], ...), ...]` — 4-uplet par itération, peu lisible.

- [ ] **Step 1 :** Proposer :
   - Remplacer la boucle par **4 blocs explicites** (un par modèle), chacun de ~8 lignes.
   - OU plus court : sortir la liste des 4 modèles dans une variable nommée puis `for i in range(4): ax = axes[i]; y_true, y_pred, title = models_data[i]`.
   - Cible : ~25 lignes → ~25 lignes (équilibre lisibilité / longueur).
- [ ] **Step 2 :** Appliquer après validation. Utilisatrice exécute, vérifie le graphique.

---

### Task 6 : Cellule `3rzupajshvp` — Plot importance des features

**Complexité :** `colors = ['#E74C3C' if f in new_features else '#2E86C1' for f in imp_sorted.index]` — comprehension avec ternaire.

- [ ] **Step 1 :** Proposer :
   - Remplacer la list comprehension par une boucle `for` de 4 lignes (équivalent direct, pas de gain de taille mais plus simple à lire).
   - Garder la structure de la boucle externe `for ax, importance, title in [tuples]:` ou la décomposer en 3 blocs.
   - Cible : ~20 lignes → ~20-22 lignes.
- [ ] **Step 2 :** Appliquer après validation, vérifier le graphique.

---

### Task 7 : Cellule `7xk75du1kzk` — Tableau récap

**Complexité :** 4 prints quasi identiques avec f-string d'alignement complexe `f"{name:<28} {recall:>8.3f} {precision_score(...):>10.4f} {f1_score(...):>8.4f}"`.

- [ ] **Step 1 :** Proposer :
   - Stocker les résultats dans une liste de tuples, faire UN seul `for` qui imprime chaque ligne.
   - Réduit 4 lignes copy-paste à 1 boucle de 3 lignes (ou 4 lignes max).
   - Cible : ~10 lignes → ~8 lignes.
- [ ] **Step 2 :** Appliquer, exécuter, vérifier output identique.

---

### Task 8 : Cellule `jnb2sya219s` — CV 5-fold + `build_features_on_fold` (+ leakage fix)

**Complexité :** TRÈS COMPLEXE — ~80 lignes mélangeant définition de `build_features_on_fold` (~40 lignes), boucle CV (~25 lignes), récapitulatif (~10 lignes). Trois responsabilités dans une cellule.

- [ ] **Step 1 :** Proposer le découpage en **3 cellules** :
   - Cellule 8a : définition de `build_features_on_fold` (avec **intégration de la task 5 du plan leakage** : ajout de `compute_risky_lists(df_fold_train)` au début, utilisation des variables locales `sender_risky` / `receiver_risky` / `risky_payments`).
   - Cellule 8b : boucle CV qui appelle la fonction et collecte les résultats. Inclure le print des listes par fold (vérification 4.3 du spec leakage).
   - Cellule 8c : récapitulatif final.
- [ ] **Step 2 :** Pour la cellule 8a : simplifier la fonction interne en **réutilisant les blocs déjà simplifiés** des tasks 1, 2 (smurfing, risky payment count) — cohérence du code.
- [ ] **Step 3 :** Appliquer via NotebookEdit (1 modify pour 8a, 2 inserts pour 8b/8c). Utilisatrice exécute, vérifie listes stables par fold + recall CV.
- [ ] **Step 4 :** Cible globale : ~80 lignes en une cellule → ~70 lignes en 3 cellules courtes.

---

### Task 9 : Cellule `aje4j41cr8u` — `manual_cv_tuning`

**Complexité :** TRÈS COMPLEXE — `**fixed_params, **params`, `product(*param_values)`, `dict({k: best_row[k] for k in results_df.columns if k not in [...]})`. Beaucoup d'idiomes Python avancés.

- [ ] **Step 1 :** Proposer :
   - Remplacer `param_names = list(param_grid.keys()); param_values = list(param_grid.values()); combinations = list(product(*param_values))` par une approche plus directe : `combinations = list(product(*param_grid.values()))` puis `for combo in combinations:` avec `params = dict(zip(param_grid.keys(), combo))` (3 lignes au lieu de 4, plus claire).
   - Garder `**fixed_params, **params` (idiome Python important, expliquer en commentaire ce que ça fait).
   - Sortir la boucle interne `for train_idx, test_idx in cv.split(...)` dans une fonction helper `evaluate_one_combo(model_class, full_params)` ? **NON** — ajouterait une couche d'indirection. Garder inline.
   - Cible : ~60 lignes → ~50 lignes (suppression des `param_names`/`param_values` séparés, tighten les prints).
- [ ] **Step 2 :** Appliquer après validation, exécuter pour vérifier que la fonction tourne (juste le print de définition).

---

### Task 10 : Cellule `lokvziyrhzm` — Récap tuning

**Complexité :** `dict({k: best_row[k] for k in results_df.columns if k not in ['mean_recall','std_recall','mean_precision']})` — dict comprehension imbriquée dans un dict cast.

- [ ] **Step 1 :** Proposer :
   - Remplacer la dict comprehension par : sélectionner les colonnes pertinentes une fois (`param_cols = [c for c in results_df.columns if c not in ['mean_recall','std_recall','mean_precision']]`) puis utiliser `best_row[param_cols].to_dict()` (2 lignes claires).
   - Cible : ~12 lignes → ~10 lignes.
- [ ] **Step 2 :** Appliquer, exécuter (ou skip si tuning pas exécuté).

---

### Task 11 : Cellule `py4hmqgit1l` — Évaluation finale (modèles tunés)

**Complexité :** TRÈS COMPLEXE — ~100 lignes, 6+ blocs séquentiels avec `print()` longs, `max([(name, score) for ...], key=lambda x: x[1])`, double boucle "avant tuning" / "après tuning" avec tuples inline.

- [ ] **Step 1 :** Proposer le découpage en **3 cellules** :
   - Cellule 11a : entraînement des 3 modèles tunés et prédictions (6-7 blocs courts).
   - Cellule 11b : tableau comparatif (1 boucle simple sur une liste de résultats).
   - Cellule 11c : analyse du surapprentissage (table train vs test, 2 blocs simples).
- [ ] **Step 2 :** Remplacer `max([(name, score) for ...], key=lambda x: x[1])` par : trouver l'index du max via `np.argmax(scores)` puis indexer dans `names` (3 lignes simples au lieu d'une lambda).
- [ ] **Step 3 :** Appliquer (1 modify, 2 inserts). Exécuter, vérifier output.
- [ ] **Step 4 :** Cible : ~100 lignes en une cellule → ~80 lignes en 3 cellules.

---

### Task 12 : Cellule `r2j6u9e1dg` — Courbes PR + tableau seuils

**Complexité :** `for ax, model, name, color in [tuples]:` × 3, puis bloc de tableau `for seuil in [...]` avec calculs imbriqués, `max([(model, name) for ...], key=lambda m: ...)`.

- [ ] **Step 1 :** Proposer :
   - Découper en **2 cellules** : 12a = courbes PR (boucle sur 3 modèles), 12b = tableau seuils du meilleur modèle.
   - Pour 12a : remplacer le tuple à 4 éléments par une liste de dicts ou 3 blocs séparés.
   - Pour 12b : remplacer `max([(m, n) for ...], key=lambda x: ...)` par une boucle explicite qui garde le meilleur recall.
- [ ] **Step 2 :** Appliquer (1 modify + 1 insert). Exécuter, vérifier graphique + tableau.
- [ ] **Step 3 :** Cible : ~70 lignes en une cellule → ~60 lignes en 2 cellules.

---

## Comptage final attendu

| Métrique | Avant | Après (estimation) |
|---|---|---|
| Total lignes de code (cellules code) | ~700 | ~600 |
| Cellules avec >40 lignes | 4 | 0 |
| Comprehensions non triviales | 6+ | 0 |
| Lambdas dans max/sort | 3 | 0 |
| Cellules code | ~25 | ~30 (découpages) |

## Hors scope (non touché)

- Cellule baseline `77da55b8` (bug d'encoding sur full dataset signalé dans la critique) — sera traité dans une autre pass.
- Optimisation perf au-delà de ce que la simplification donne au passage.
- Refacto des features en module `.py` séparé.
- Ajout de nouvelles features ou changement de modèles.

## Références

- Spec : `docs/specs/2026-05-10-simplification-beginner-friendly-design.md`
- Plan leakage en cours (absorbé ici) : `docs/plans/2026-05-09-fix-risky-lists-leakage.md`
- Notebook : `feature_engineering_aml.ipynb`
