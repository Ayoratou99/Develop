# Réunion du 2026-09-01 — statuts de commande

Décision : on n'ajoute PAS de statut `paid` pour l'instant. Le paiement sera géré par un champ `paid_at`
sur `orders` dans une migration ultérieure. Le frontend ne doit pas afficher de filtre « payée ».

Participants : Awa (back), Idris (front), Nadia (produit).
