-- Schéma de référence (généré depuis les migrations)
CREATE TABLE customers (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  email VARCHAR(255) NOT NULL UNIQUE,
  full_name VARCHAR(255) NOT NULL,
  phone VARCHAR(30) NULL
);

CREATE TABLE orders (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  reference VARCHAR(20) NOT NULL UNIQUE,
  customer_id BIGINT NOT NULL REFERENCES customers(id),
  status ENUM('draft','confirmed','shipped','cancelled') NOT NULL DEFAULT 'draft',
  total_amount BIGINT UNSIGNED NOT NULL DEFAULT 0,
  confirmed_at TIMESTAMP NULL
);

CREATE TABLE order_lines (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  order_id BIGINT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
  product_sku VARCHAR(40) NOT NULL,
  quantity INT UNSIGNED NOT NULL,
  unit_price BIGINT UNSIGNED NOT NULL
);
