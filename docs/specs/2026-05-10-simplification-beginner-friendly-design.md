# Pass de simplification beginner-friendly du notebook AML

**Date :** 2026-05-10
**Projet :** aml-feature-engineering
**Auteur :** Nashely Castillo

---

## 1. Contexte et motivation

Le notebook `feature_engineering_aml.ipynb` contient des cellules dont certains idiomes Python sont avancés (comprehensions, lambdas, `**kwargs`, chaînes pandas longues, fonctions volumineuses). L'utilisatrice est débutante en Python et doit pouvoir **lire, comprendre et reproduire** chaque cellule sans assistance.

Une pass de simplification pédagogique est nécessaire.

## 2. Objectif

Réécrire les cellules complexes du notebook dans un style accessible à un débutant Python, **sans alourdir** le code (idéalement le raccourcir).

## 3. Workflow

Travail interactif **cellule par cellule**, du début du notebook à la fin :

1. Pour chaque cellule code, classer : **simple** ou **complexe**.
2. Si simple → on saute, on passe à la suivante.
3. Si complexe → proposer une version simplifiée avec :
   - Le diff (ce qui change)
   - L'explication pédagogique (quel idiome remplace, par quoi, pourquoi)
   - Le concept à retenir
4. L'utilisatrice valide ou pose des questions.
5. Modification appliquée via NotebookEdit, exécutée dans Jupyter, validée.

## 4. Critères de classification

### Cellule "complexe" (à simplifier)

- Comprehensions de liste / dict / set (`[f(x) for x in y if z]`)
- Lambdas (`lambda x: ...`, `key=lambda x: x[1]`)
- Chaînes pandas longues (`df.X.groupby(Y).agg(Z).reset_index().sort_values()...`)
- Unpacking multiple ou compliqué (`for a, b, c in zip(*X)`, `**fixed_params, **params`)
- `*args`, `**kwargs` non triviaux
- Plusieurs opérations distinctes sur une même ligne
- Cellule de >30-40 lignes qui fait plusieurs choses indépendantes

### Cellule "simple" (à garder)

- Imports, constantes, assignations directes
- Boucles `for` explicites avec `range()` ou itération directe sur une liste
- `if/else` simples
- Calls de fonctions sklearn / xgboost standards (peu importe le nombre de paramètres, c'est de l'API publique)
- Cellules courtes (< 30 lignes) et focalisées

## 5. Règles de simplification

### Substitutions courantes

| Avant (complexe) | Après (simple) |
|---|---|
| `[f(x) for x in xs]` | boucle `for` avec `.append()` (max 3 lignes) |
| `lambda x: x[1]` passé à `max(key=...)` | extraire en fonction nommée OU utiliser `operator.itemgetter` (mais préférer fonction nommée) |
| `for ax, model, name, color in zip(axes, models, names, colors)` | une boucle `for i in range(len(...))` avec accès indexé |
| `**fixed_params, **params` | dictionnaire fusionné en 2 lignes, ou défaire la généralisation |
| `df.X.Y.Z.W()` | 2-3 variables intermédiaires nommées (uniquement si elles clarifient) |

### Règle de taille

**Cible : code de taille égale ou inférieure après simplification.**

- 1 ligne dense → max 2-3 lignes claires (jamais 5+).
- Une cellule longue est **découpée** en plusieurs cellules courtes (le total ne grossit pas).
- Code mort supprimé au passage : prints redondants, variables inutilisées, commentaires QUOI évidents, blocs de "vérification" pléonastiques.

## 6. Intégration avec le plan correction du leakage

Le plan `2026-05-09-fix-risky-lists-leakage.md` (tasks 3-6 restantes) **se dissout dans cette pass** :
- Quand on arrive à la cellule `49469cdc` (Risky Payment Count) : on applique aussi la suppression de la ligne `RISKY_PAYMENT_TYPES = {...}` (task 3 du plan).
- Quand on arrive à la cellule `jnb2sya219s` (CV) : on applique aussi le recalcul dynamique dans `build_features_on_fold()` (task 5 du plan).
- Pas de plan séparé à exécuter, pas de double passage.

## 7. Critère de succès

Pour chaque cellule simplifiée :
- L'utilisatrice peut **lire chaque ligne et l'expliquer à voix haute** (= comprendre ce que ça fait).
- Le nombre de lignes est **égal ou inférieur** à la version d'origine.
- Le comportement (output) est strictement le même (même valeurs, mêmes prints, mêmes graphes).

À la fin de la pass complète :
- Aucune comprehension de liste / dict / set non triviale dans le notebook.
- Aucune lambda non triviale.
- Aucune cellule de >40 lignes.
- Total de lignes du notebook : ≤ version actuelle.

## 8. Hors scope

- Refacto en helpers Python `.py` séparés (cassant pour débutant, complique l'import).
- Changement de méthodologie ou de modèles.
- Ajout de nouvelles features.
- Optimisation performance (sauf si la simplification donne un gain trivial au passage).
