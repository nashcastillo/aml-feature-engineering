# --- AML feature engineering - sandbox reproductible ---
# Environnement isole pour reproduire le pipeline sans contamination locale.
# Aucune donnee reelle ne doit etre presente dans l'image : on mount un volume au runtime.

FROM python:3.11-slim AS base

# Securite : pas d'utilisateur root pour l'execution
RUN groupadd --gid 1000 mlops && useradd --uid 1000 --gid mlops --shell /bin/bash --create-home mlops

# Dependances systeme minimales pour les libs ML
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer Python deps d'abord pour profiter du cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code (notebook, scripts, docs - sans donnees grace au .dockerignore)
COPY --chown=mlops:mlops . .

# Bascule sur l'utilisateur non-root
USER mlops

# Le dataset n'est PAS dans l'image - il faut le mount au runtime :
#   docker run -v /local/path/SAML-D_sample_800k.csv:/data/SAML-D_sample_800k.csv aml-feature-engineering
ENV DATA_PATH=/data/SAML-D_sample_800k.csv

# Par defaut : lancer le notebook end-to-end
CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "feature_engineering_aml.ipynb"]
