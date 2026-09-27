"""
generate_data.py
-----------------
Run this FIRST.

This creates a supermarket with 30 products across 6 categories, and
generates 3 years of monthly demand history for each one - showing how
demand changes with PRICE and MONTH (season). This is what our ML model
will learn from.

We deliberately do NOT use "past units sold" as an input feature for
predicting future demand (that would be cheating / data leakage in a real
project). The model only ever sees: product, price, month.

This also sets up the SQLite database and fills the `products` and
`demand_history` tables.
"""

import numpy as np
import pandas as pd
from database import get_connection

np.random.seed(42)

# ---------- 1. DEFINE THE SUPERMARKET'S PRODUCT CATALOG ----------
# category -> list of (product_name, base_price, base_monthly_demand, seasonal_type)
# seasonal_type controls which months get a demand boost for that product
CATALOG = {
    "Grocery": [
        ("Basmati Rice 5kg", 450, 300, "festive"),
        ("Wheat Flour 5kg", 220, 350, "flat"),
        ("Sugar 1kg", 45, 400, "festive"),
        ("Cooking Oil 1L", 160, 280, "flat"),
        ("Toor Dal 1kg", 130, 260, "flat"),
    ],
    "Dairy": [
        ("Milk 1L", 60, 600, "flat"),
        ("Paneer 200g", 90, 220, "festive"),
        ("Butter 500g", 250, 150, "winter"),
        ("Curd 400g", 45, 320, "summer"),
        ("Cheese Slices", 120, 180, "flat"),
    ],
    "Bakery": [
        ("White Bread", 40, 500, "flat"),
        ("Brown Bread", 55, 300, "flat"),
        ("Rusk Pack", 35, 200, "flat"),
        ("Birthday Cake (small)", 300, 60, "festive"),
        ("Cookies Pack", 80, 250, "festive"),
    ],
    "Beverages": [
        ("Cola 1.25L", 60, 350, "summer"),
        ("Mango Juice 1L", 110, 280, "summer"),
        ("Mineral Water 1L", 20, 500, "summer"),
        ("Tea Powder 250g", 140, 300, "winter"),
        ("Instant Coffee 100g", 220, 150, "winter"),
    ],
    "Personal Care": [
        ("Shampoo 340ml", 210, 180, "flat"),
        ("Toothpaste 150g", 90, 300, "flat"),
        ("Soap Bar (pack of 4)", 120, 260, "flat"),
        ("Umbrella", 350, 40, "monsoon"),
        ("Sunscreen 100ml", 280, 60, "summer"),
    ],
    "Household": [
        ("Detergent Powder 1kg", 130, 240, "flat"),
        ("Dishwash Liquid 500ml", 95, 220, "flat"),
        ("Floor Cleaner 1L", 110, 150, "flat"),
        ("Garbage Bags (pack)", 70, 200, "flat"),
        ("Mosquito Repellent", 85, 100, "monsoon"),
    ],
}

SEASON_MONTHS = {
    "festive": [10, 11, 12],      # Oct-Dec: Diwali/Christmas/New Year season
    "winter": [11, 12, 1, 2],     # cold months
    "summer": [3, 4, 5, 6],       # hot months
    "monsoon": [6, 7, 8, 9],      # rainy months
    "flat": [],                   # no strong seasonality
}


def seasonal_multiplier(season_type, month):
    if month in SEASON_MONTHS.get(season_type, []):
        return np.random.uniform(1.4, 1.9)   # 40-90% boost in season
    return np.random.uniform(0.85, 1.1)      # normal fluctuation


def main():
    conn = get_connection()
    cur = conn.cursor()

    # Drop and recreate all tables from scratch every time this script runs.
    # This keeps product IDs always starting from 1, so the trained model
    # and the database never fall out of sync with each other.
    cur.execute("DROP TABLE IF EXISTS sales_log")
    cur.execute("DROP TABLE IF EXISTS demand_history")
    cur.execute("DROP TABLE IF EXISTS products")

    cur.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            base_price REAL NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE demand_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            year INTEGER NOT NULL,
            month INTEGER NOT NULL,
            price REAL NOT NULL,
            units_sold INTEGER NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        )
    """)
    cur.execute("""
        CREATE TABLE sales_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            sale_date TEXT NOT NULL,
            price REAL NOT NULL,
            quantity INTEGER NOT NULL,
            note TEXT,
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        )
    """)
    conn.commit()

    all_rows = []  # for the CSV export too

    for category, products in CATALOG.items():
        for name, base_price, base_demand, season_type in products:
            cur.execute(
                "INSERT INTO products (name, category, base_price) VALUES (?, ?, ?)",
                (name, category, base_price)
            )
            product_id = cur.lastrowid

            # 3 years of monthly data
            for year in [2023, 2024, 2025]:
                for month in range(1, 13):
                    # Generate 3-4 different pricing scenarios per month,
                    # so the model can actually learn "if price goes up/down,
                    # what happens to demand" (price elasticity)
                    num_scenarios = np.random.randint(3, 5)
                    for _ in range(num_scenarios):
                        # price varies +/- 20% around base price (simulating discounts/hikes)
                        price_factor = np.random.uniform(0.8, 1.2)
                        price = round(base_price * price_factor, 2)

                        # core demand formula:
                        # start from base demand, apply seasonal boost,
                        # apply price elasticity (higher price -> lower demand),
                        # add small random noise (real life isn't perfectly predictable)
                        season_mult = seasonal_multiplier(season_type, month)
                        price_effect = 1 - (price_factor - 1) * 0.9  # elasticity factor
                        demand = base_demand * season_mult * price_effect
                        demand *= np.random.uniform(0.95, 1.05)  # small noise
                        demand = max(0, round(demand))

                        all_rows.append([product_id, name, category, year, month, price, demand])

                        cur.execute(
                            "INSERT INTO demand_history (product_id, year, month, price, units_sold) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (product_id, year, month, price, demand)
                        )

    conn.commit()
    conn.close()

    # also save as CSV, useful for showing your mentor the raw dataset
    df = pd.DataFrame(all_rows, columns=[
        "product_id", "product_name", "category", "year", "month", "price", "units_sold"
    ])
    df.to_csv("demand_history.csv", index=False)

    print(f"Done. Inserted {len(CATALOG) * 5} products across {len(CATALOG)} categories.")
    print(f"Generated {len(df)} historical demand records (demand_history.csv + supermarket.db).")


if __name__ == "__main__":
    main()
