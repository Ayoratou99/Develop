# Spécification fonctionnelle — Commandes

## Règles métier

1. Une commande est créée au statut `draft`.
2. Seule une commande `draft` peut être confirmée (`POST /api/orders/{id}/confirm`).
3. À la confirmation, une notification est envoyée au client via le service notifier.
4. Le montant total est la somme des lignes (quantité × prix unitaire), en centimes.
5. Une commande confirmée ne peut plus être modifiée, seulement expédiée ou annulée.

## Écrans

- Liste des commandes : référence, statut, montant formaté, bouton Confirmer.
- Recherche client : par nom, 20 résultats maximum.
