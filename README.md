# SuperMart Demand Forecasting System — Minor Project

A full website (not just a script) that predicts how much of a supermarket
product will be demanded, based on **price** and **month**, backed by a
real database (SQLite) that also logs live sales.

## 1. What this project does
"This is a demand forecasting system for a supermarket with 30 products
across 6 categories (Grocery, Dairy, Bakery, Beverages, Personal Care,
Household). A machine learning model, trained on 3 years of historical
data, predicts how many units of a product will sell based on its price
and the month — capturing both price elasticity (cheaper = more demand)
and seasonality (e.g. umbrellas sell more in monsoon, cakes sell more
around festivals). The system also has a live sales-entry feature backed
by a SQL database, simulating a real billing counter."

## 2. Tech stack
- **Backend:** Python + Flask (a website framework, not just a script)
- **Database:** SQLite (a real relational database — this is your DBMS component)
- **Machine Learning:** scikit-learn (Random Forest Regressor)
- **Frontend:** HTML + Bootstrap 5 + Chart.js (via CDN, no separate install needed)

## 3. Project structure
```
supermarket_project/
├── database.py          -> all database (SQL) code: create tables, insert, query
├── generate_data.py      -> STEP 1: builds the product catalog + 3 years of demand history
├── train_model.py        -> STEP 2: trains the ML model from the database
├── app.py                 -> STEP 3: the Flask website itself
├── requirements.txt
├── templates/              -> HTML pages (dashboard, predict, record sale, history, products)
├── static/                  -> CSS + generated graphs
├── supermarket.db           -> the SQLite database (created after step 1)
└── demand_model.pkl         -> the trained model (created after step 2)
```

## 4. How to run it (in order — do not skip steps)

```bash
# 1. Install the required libraries
pip install -r requirements.txt

# 2. Build the database + generate historical data
python generate_data.py

# 3. Train the machine learning model
python train_model.py

# 4. Start the website
python app.py
```

Then open your browser and go to: **http://127.0.0.1:5000**

## 5. Pages in the website
| Page | URL | What it does |
|---|---|---|
| Dashboard | `/` | Overview stats, demand trend chart, top products |
| Predict Demand | `/predict` | Pick a product, price, month → get predicted demand |
| Record Sale | `/record-sale` | Log a real sale into the database (billing counter simulation) |
| Sales History | `/history` | View / delete all logged sales (pulled live from the database) |
| Products | `/products` | Browse the full 30-product catalog |

## 6. How the ML model works (explain this to your mentor)

**Step 1 — Data Generation**
Simulated 3 years (2023-2025) of monthly demand for 30 products, with
multiple price scenarios per month per product (~3,800 records total).
Demand is generated using a realistic formula:
`demand = base_demand × seasonal_boost × price_effect × small_random_noise`
- Seasonal boost: products get higher demand in their relevant season
  (umbrellas in monsoon, cakes around festivals, cold drinks in summer).
- Price effect: as price goes up, demand goes down (price elasticity),
  and vice versa.

**Step 2 — Why only price + month as features?**
The model is trained ONLY on `product`, `price`, and `month` — not on
past sales figures. This is a deliberate design choice: the goal is to
answer "what happens to demand if I change the price, or if the month
changes?" rather than just memorizing old sales numbers (which would be
a form of data leakage and wouldn't generalize to new pricing decisions).

**Step 3 — Model comparison**
Two models were trained and compared:
- Linear Regression (R² ≈ 0.36 — too simple, can't capture seasonality)
- Random Forest Regressor (R² ≈ 0.96 — captures non-linear seasonal +
  price patterns much better)

**Step 4 — The database's role**
Unlike a typical "just run a script" ML project, this system stores
everything in a real SQL database:
- `products` table — the master catalog
- `demand_history` table — the historical training data
- `sales_log` table — live sales entered through the website (a genuine
  CRUD feature: Create via the form, Read via History page, Delete via
  the delete button)

## 7. Likely mentor questions — how to answer

- **"Why SQLite and not MySQL?"**
  → "SQLite needs no separate server install, the whole database is one
  file, and it supports full SQL (joins, foreign keys) — ideal for a
  project like this. The same code would work with MySQL/PostgreSQL with
  minimal changes since I used standard SQL."

- **"Why exclude past sales as a feature?"**
  → "Because that would let the model 'cheat' by just copying old
  numbers. I wanted it to actually learn the relationship between price,
  season, and demand — which is more useful for decisions like 'what if
  I raise the price by 10%?'"

- **"Why Random Forest over Linear Regression?"**
  → "Demand isn't a straight line against price or month — it has
  seasonal spikes and dips. Random Forest handles these non-linear
  patterns; Linear Regression assumes everything is a straight
  relationship, so it performs much worse here (R² 0.36 vs 0.96)."

- **"How would you scale this to a real supermarket?"**
  → Replace synthetic data with real POS (point-of-sale) transaction
  data, add more features like weather or local events, retrain
  periodically as new sales_log data accumulates, and deploy on a cloud
  server with a production database like PostgreSQL.

- **"Is this connected to real live data?"**
  → "The 'Record Sale' feature simulates a live billing counter — sales
  entered there are immediately stored in the database and visible on the
  History page. In a real deployment, this would be wired to an actual
  POS system instead of a manual form."

## 8. Optional upgrades if you have extra time
- Auto-retrain the model periodically using accumulated `sales_log` data
- Add login/authentication for an "admin" who manages products
- Add a "compare products" chart on the dashboard
- Deploy it online (Render/PythonAnywhere) so it has a real public link
