"""
database.py
------------
All database related code lives here. We use SQLite because it doesn't need
a separate server installed (like MySQL would) - the whole database is just
one file (supermarket.db). Perfect for a college project.

Tables:
1. products        -> master list of all supermarket products
2. demand_history   -> past monthly demand data (used to train the ML model)
3. sales_log        -> live sales entries recorded by the user through the website
"""

import sqlite3

DB_NAME = "supermarket.db"


def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    return conn


def init_db():
    """Creates the tables if they don't already exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            base_price REAL NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS demand_history (
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
        CREATE TABLE IF NOT EXISTS sales_log (
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
    conn.close()


def get_all_products():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM products ORDER BY category, name").fetchall()
    conn.close()
    return rows


def get_product_by_id(product_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM products WHERE product_id = ?", (product_id,)).fetchone()
    conn.close()
    return row


def add_sale(product_id, sale_date, price, quantity, note=""):
    conn = get_connection()
    conn.execute(
        "INSERT INTO sales_log (product_id, sale_date, price, quantity, note) VALUES (?, ?, ?, ?, ?)",
        (product_id, sale_date, price, quantity, note)
    )
    conn.commit()
    conn.close()


def get_sales_log(limit=100):
    conn = get_connection()
    rows = conn.execute("""
        SELECT sales_log.id, products.name, products.category, sales_log.sale_date,
               sales_log.price, sales_log.quantity, sales_log.note
        FROM sales_log
        JOIN products ON sales_log.product_id = products.product_id
        ORDER BY sales_log.id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return rows


def delete_sale(sale_id):
    conn = get_connection()
    conn.execute("DELETE FROM sales_log WHERE id = ?", (sale_id,))
    conn.commit()
    conn.close()


def get_monthly_demand_summary():
    """Used on the dashboard chart - total units sold across all products, per month."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT year, month, SUM(units_sold) as total_units
        FROM demand_history
        GROUP BY year, month
        ORDER BY year, month
    """).fetchall()
    conn.close()
    return rows


def get_top_products_by_demand(limit=5):
    conn = get_connection()
    rows = conn.execute("""
        SELECT products.name, SUM(demand_history.units_sold) as total_units
        FROM demand_history
        JOIN products ON demand_history.product_id = products.product_id
        GROUP BY products.name
        ORDER BY total_units DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return rows


def get_stats():
    conn = get_connection()
    total_products = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    total_categories = conn.execute("SELECT COUNT(DISTINCT category) FROM products").fetchone()[0]
    total_history_rows = conn.execute("SELECT COUNT(*) FROM demand_history").fetchone()[0]
    total_sales_logged = conn.execute("SELECT COUNT(*) FROM sales_log").fetchone()[0]
    conn.close()
    return {
        "total_products": total_products,
        "total_categories": total_categories,
        "total_history_rows": total_history_rows,
        "total_sales_logged": total_sales_logged
    }
