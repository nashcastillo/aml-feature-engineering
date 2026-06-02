"""Anonymisation tokenisation des comptes SAML-D (demonstration RGPD).

Hash SHA-256 + salt sur les colonnes Sender_account et Receiver_account.
Conservation des autres colonnes (Amount, Date, Time, Payment_type, ...)
pour les besoins du ML. Operation irreversible sans le salt.

Usage :
    export ANONYMIZATION_SALT=$(python -c "import secrets; print(secrets.token_hex(32))")
    python scripts/anonymize.py input.csv output_anonymized.csv

Note compliance :
- L'anonymisation est conforme RGPD article 4(5) si le salt est conserve secret
- En production : salt en HSM (Hardware Security Module) ou vault, pas en variable d'env
- Article 32 RGPD : pseudonymisation = mesure de securite appropriee
"""
import hashlib
import os
import sys

import pandas as pd


COLUMNS_TO_HASH = ['Sender_account', 'Receiver_account']


def hash_account(account_id, salt):
    """SHA-256 du compte avec salt. Tronque a 16 chars hexa pour lisibilite."""
    h = hashlib.sha256()
    h.update(salt.encode('utf-8'))
    h.update(str(account_id).encode('utf-8'))
    return h.hexdigest()[:16]


def anonymize_csv(input_path, output_path, salt):
    """Charge un CSV, hash les colonnes sensibles, sauvegarde le resultat."""
    print(f"[anonymize] lecture : {input_path}")
    df = pd.read_csv(input_path)
    print(f"[anonymize] {len(df):,} lignes, colonnes : {list(df.columns)}")

    for col in COLUMNS_TO_HASH:
        if col not in df.columns:
            print(f"[anonymize] WARNING : colonne '{col}' absente du CSV, ignoree")
            continue
        # Map sur les uniques pour gagner du temps sur 800k+ lignes
        unique_values = df[col].drop_duplicates()
        mapping = {v: hash_account(v, salt) for v in unique_values}
        df[col] = df[col].map(mapping)
        print(f"[anonymize] {col} : {len(mapping):,} comptes uniques anonymises")

    df.to_csv(output_path, index=False)
    print(f"[anonymize] ecriture : {output_path}")
    print(f"[anonymize] fait. Le salt utilise est SECRET, ne le perdre PAS.")


def main():
    if len(sys.argv) != 3:
        print("Usage : python scripts/anonymize.py input.csv output_anonymized.csv")
        print("Necessite la variable d'environnement ANONYMIZATION_SALT.")
        sys.exit(1)

    salt = os.environ.get('ANONYMIZATION_SALT')
    if not salt or len(salt) < 32:
        print("ERREUR : la variable ANONYMIZATION_SALT doit etre definie (32+ caracteres).")
        print("Generer un salt : python -c \"import secrets; print(secrets.token_hex(32))\"")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(input_path):
        print(f"ERREUR : fichier introuvable : {input_path}")
        sys.exit(1)

    anonymize_csv(input_path, output_path, salt)


if __name__ == '__main__':
    main()
