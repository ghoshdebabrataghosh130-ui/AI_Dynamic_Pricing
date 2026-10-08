from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
import os

# Initialize FastAPI application
app = FastAPI(
    title="AI-Driven Dynamic Pricing API",
    description="Dynamic Pricing and Demand Prediction using trained RandomForestRegressor",
    version="1.0"
)

# Enable CORS for external/local UI clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to trained artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "dynamic_pricing_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")

# Load artifacts
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    print("Model, scaler, and feature columns loaded successfully.")
except Exception as e:
    model = None
    scaler = None
    feature_columns = None
    print(f"Warning: Could not load artifacts: {e}")


# Input Schema
class PricingRequest(BaseModel):
    base_price: float = Field(default=1000.0, example=1000.0, description="Actual / List Price")
    units_sold: float = Field(..., example=80.0, description="Number of units sold")
    demand: float = Field(..., example=120.0, description="Estimated customer demand")
    competitor_pricing: float = Field(..., example=950.0, description="Competitor price")
    inventory_level: float = Field(..., example=40.0, description="Current inventory level")
    promotion_discount: float = Field(..., example=5.0, description="Discount percentage (e.g. 5 for 5%)")
    category: str = Field(..., example="Electronics", description="Product category: Clothing, Electronics, Furniture, Groceries, Toys")
    region: str = Field(..., example="East", description="Region: North, South, East, West")
    weather_condition: str = Field(..., example="Sunny", description="Weather: Rainy, Snowy, Sunny, Cloudy")
    seasonality: str = Field(..., example="Summer", description="Season: Autumn, Spring, Summer, Winter")


def dynamic_price_engine(predicted_price: float, inventory_level: float, demand: float, min_price: float = 10.0, max_price: float = 2000.0):
    """Adjusts base ML predicted price according to business rules."""
    recommended_price = predicted_price
    reason = "Base ML Prediction"

    # Relative inventory and demand classification
    is_high_demand = demand > 100
    is_low_demand = demand < 50
    is_high_inventory = inventory_level > 100
    is_low_inventory = inventory_level < 30

    if is_high_demand and is_low_inventory:
        recommended_price *= 1.10
        reason = "High Demand, Low Inventory (Surge +10%)"
    elif is_high_demand and not is_high_inventory:
        recommended_price *= 1.05
        reason = "High Demand, Normal Inventory (+5%)"
    elif is_low_demand and is_high_inventory:
        recommended_price *= 0.90
        reason = "Low Demand, High Inventory (Discount -10%)"
    elif is_low_demand:
        recommended_price *= 0.95
        reason = "Low Demand (Discount -5%)"

    # Business boundary constraints
    recommended_price = max(min_price, min(max_price, recommended_price))

    return round(recommended_price, 2), reason


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def home_ui():
    index_file = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h2>AI Dynamic Pricing API is running. index.html not found.</h2>")


@app.get("/api")
def api_info():
    return {
        "message": "AI-Driven Dynamic Pricing API is running",
        "model": "RandomForestRegressor",
        "docs_url": "/docs",
        "status": "success"
    }


@app.get("/health")
def health():
    is_healthy = model is not None and scaler is not None and feature_columns is not None
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": is_healthy
    }


@app.post("/predict")
def predict_price(request: PricingRequest):
    if model is None or scaler is None or feature_columns is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")

    # 1. Feature Engineering
    inventory_level_log = np.log1p(request.inventory_level)
    inventory_demand_ratio = request.inventory_level / (request.demand + 1e-6)
    sales_demand_ratio = request.units_sold / (request.demand + 1e-6)
    demand_inventory_interaction = request.demand * request.inventory_level
    promotion_discount_interaction = request.demand * request.promotion_discount

    # 2. One-Hot Encodings
    category_features = {f"Category_{c}": 0 for c in ["Clothing", "Electronics", "Furniture", "Groceries", "Toys"]}
    cat_key = f"Category_{request.category}"
    if cat_key in category_features:
        category_features[cat_key] = 1

    region_features = {f"Region_{r}": 0 for r in ["East", "North", "South", "West"]}
    reg_key = f"Region_{request.region}"
    if reg_key in region_features:
        region_features[reg_key] = 1

    weather_features = {f"Weather Condition_{w}": 0 for w in ["Cloudy", "Rainy", "Snowy", "Sunny"]}
    weath_key = f"Weather Condition_{request.weather_condition}"
    if weath_key in weather_features:
        weather_features[weath_key] = 1

    seasonality_features = {f"Seasonality_{s}": 0 for s in ["Autumn", "Spring", "Summer", "Winter"]}
    season_key = f"Seasonality_{request.seasonality}"
    if season_key in seasonality_features:
        seasonality_features[season_key] = 1

    # 3. Assemble input DataFrame
    input_dict = {
        "Base Price": request.base_price,
        "Units Sold": request.units_sold,
        "Demand": request.demand,
        "Competitor Pricing": request.competitor_pricing,
        "Inventory Level_Log": inventory_level_log,
        "Inventory-to-Demand Ratio": inventory_demand_ratio,
        "Sales-to-Demand Ratio": sales_demand_ratio,
        "Demand Inventory Interaction": demand_inventory_interaction,
        "Promotion Discount Interaction": promotion_discount_interaction,
        **category_features,
        **region_features,
        **weather_features,
        **seasonality_features
    }

    input_df = pd.DataFrame([input_dict])

    # Ensure matching column order
    input_df = input_df[feature_columns]

    # 4. Scale inputs
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_columns)

    # 5. Predict base price
    predicted_price = float(model.predict(input_scaled_df)[0])
    recommended_price, reason = dynamic_price_engine(
        predicted_price=predicted_price,
        inventory_level=request.inventory_level,
        demand=request.demand
    )

    return {
        "base_price": request.base_price,
        "predicted_price": round(predicted_price, 2),
        "recommended_price": recommended_price,
        "pricing_strategy": reason,
        "competitor_price": request.competitor_pricing,
        "inventory_level": request.inventory_level,
        "demand": request.demand
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
