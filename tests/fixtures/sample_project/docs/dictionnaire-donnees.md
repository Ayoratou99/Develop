# Dictionnaire de données

## Table `orders` (Commandes)

| Colonne        | Type              | Description                                              |
|----------------|-------------------|----------------------------------------------------------|
| id             | bigint            | Identifiant technique                                    |
| reference      | varchar(20)       | Référence métier unique, format CMD-AAAA-NNNNN           |
| customer_id    | bigint (FK)       | Client propriétaire (customers.id)                       |
| status         | enum              | draft, confirmed, shipped, cancelled                     |
| total_amount   | unsigned bigint   | Montant total en centimes d'euro                         |
| confirmed_at   | timestamp (null)  | Date de confirmation, renseignée par OrderService::confirm |

## Table `order_lines` (Lignes de commande)

| Colonne     | Type            | Description                         |
|-------------|-----------------|-------------------------------------|
| order_id    | bigint (FK)     | Commande parente                    |
| product_sku | varchar(40)     | Référence produit                   |
| quantity    | unsigned int    | Quantité commandée                  |
| unit_price  | unsigned bigint | Prix unitaire en centimes           |

## Table `customers` (Clients)

| Colonne   | Type         | Description            |
|-----------|--------------|------------------------|
| email     | varchar(255) | Email unique           |
| full_name | varchar(255) | Nom complet            |
| phone     | varchar(30)  | Téléphone (optionnel)  |
