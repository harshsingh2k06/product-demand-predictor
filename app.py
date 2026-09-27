"""
app.py
------
This is the website's backend (built with Flask).
Run this LAST, after generate_data.py and train_model.py have both run.

Pages:
  /              -> Dashboard (overview stats + demand trend chart)
  /predict       -> Predict demand for a product at a given price & month
  /record-sale   -> Log a real/live sale into the database
  /history       -> View all logged sales
  /products      -> Browse the product catalog
"""

from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import pandas as pd
from datetime import date

import database as db

app = Flask(__name__)
app.secret_key = "college-project-secret-key"  # needed for flash messages

# Load the trained model + encoder once, when the server starts
model = joblib.load("demand_model.pkl")
encoder = joblib.load("product_encoder.pkl")
features = joblib.load("model_features.pkl")


@app.route("/")
def index():
    stats = db.get_stats()
    monthly = db.get_monthly_demand_summary()
    top_products = db.get_top_products_by_demand()

    chart_labels = [f"{row['month']}/{str(row['year'])[2:]}" for row in monthly]
    chart_values = [row["total_units"] for row in monthly]

    return render_template(
        "index.html",
        stats=stats,
        chart_labels=chart_labels,
        chart_values=chart_values,
        top_products=top_products
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    products = db.get_all_products()
    prediction = None
    selected_product_id = None
    selected_price = None
    selected_month = None

    if request.method == "POST":
        selected_product_id = int(request.form["product_id"])
        selected_price = float(request.form["price"])
        selected_month = int(request.form["month"])

        product_encoded = encoder.transform([selected_product_id])[0]
        input_df = pd.DataFrame([{
            "product_encoded": product_encoded,
            "price": selected_price,
            "month": selected_month
        }])[features]

        result = model.predict(input_df)[0]
        prediction = max(0, int(round(result)))

    return render_template(
        "predict.html",
        products=products,
        prediction=prediction,
        selected_product_id=selected_product_id,
        selected_price=selected_price,
        selected_month=selected_month
    )


@app.route("/record-sale", methods=["GET", "POST"])
def record_sale():
    products = db.get_all_products()

    if request.method == "POST":
        product_id = int(request.form["product_id"])
        sale_date = request.form["sale_date"]
        price = float(request.form["price"])
        quantity = int(request.form["quantity"])
        note = request.form.get("note", "")

        db.add_sale(product_id, sale_date, price, quantity, note)
        flash("Sale recorded successfully!", "success")
        return redirect(url_for("record_sale"))

    return render_template("record_sale.html", products=products, today=date.today().isoformat())


@app.route("/history")
def history():
    sales = db.get_sales_log()
    return render_template("history.html", sales=sales)


@app.route("/delete-sale/<int:sale_id>", methods=["POST"])
def delete_sale(sale_id):
    db.delete_sale(sale_id)
    flash("Entry deleted.", "success")
    return redirect(url_for("history"))


@app.route("/products")
def products():
    all_products = db.get_all_products()
    return render_template("products.html", products=all_products)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
