from flask import Flask, jsonify
from src.models.forecast_28_days import load_data, load_model, create_future_forecast
from src.business.inventory_recommendation import create_inventory_recommendations

app = Flask(__name__)

df = load_data()
model = load_model()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/forecast/28days", methods=["GET"])
def forecast_28_days():
    forecast_df = create_future_forecast(df, model)

    return jsonify({
        "forecast_horizon": 28,
        "sample_records": forecast_df.head(100).to_dict(orient="records")
    })


@app.route("/inventory/recommendations", methods=["GET"])
def inventory_recommendations():
    forecast_df = create_future_forecast(df, model)
    inventory_df = create_inventory_recommendations(forecast_df)

    return jsonify({
        "total_items": len(inventory_df),
        "sample_recommendations": inventory_df.head(100).to_dict(orient="records")
    })

@app.route("/", methods=["GET"])
def home():
    return {
        "message": "Intelligent Forecasting MLOps API",
        "available_endpoints": [
            "/health",
            "/forecast/28days",
            "/inventory/recommendations"
        ]
    }

if __name__ == "__main__":
    app.run(debug=True)