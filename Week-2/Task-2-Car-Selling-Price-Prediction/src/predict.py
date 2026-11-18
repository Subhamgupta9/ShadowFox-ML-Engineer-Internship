"""
Production Prediction Engine and Input Validation Module.
ShadowFox ML Engineer Internship — Task 2: Car Selling Price Prediction
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "car_price_model.joblib"
REF_YEAR = 2018

VALID_FUEL_TYPES = ['Petrol', 'Diesel', 'CNG']
VALID_SELLER_TYPES = ['Dealer', 'Individual']
VALID_TRANSMISSIONS = ['Manual', 'Automatic']
VALID_OWNERS = [0, 1, 2, 3]

FEATURE_COLUMNS = ['Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner', 'Vehicle_Age']

def format_inr(amount_in_lakhs: float) -> str:
    """Formats amount in Lakhs into Indian Rupee denomination (e.g., ₹5,42,000)."""
    rupees = int(round(amount_in_lakhs * 100000))
    s = str(rupees)
    if len(s) <= 3:
        return f"₹{s}"
    last_three = s[-3:]
    remaining = s[:-3]
    # Indian comma grouping (2 digits grouping before last 3 digits)
    groups = []
    while len(remaining) > 2:
        groups.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        groups.insert(0, remaining)
    formatted = ",".join(groups) + "," + last_three
    return f"₹{formatted}"

class CarPricePredictor:
    """
    Production-grade inference engine for car resale price prediction with strict input validation.
    """
    def __init__(self, model_path: str or Path = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Please execute train.py first."
            )
        self.pipeline = joblib.load(self.model_path)

    def validate_inputs(self, raw_input: dict) -> pd.DataFrame:
        """
        Validates raw input fields, performs type checks, and calculates Vehicle_Age.
        Supports both PascalCase and snake_case / lowercase keys.
        """
        # Key normalization mapping
        key_map = {
            'present_price': 'Present_Price',
            'kms_driven': 'Kms_Driven',
            'fuel_type': 'Fuel_Type',
            'seller_type': 'Seller_Type',
            'transmission': 'Transmission',
            'owner': 'Owner',
            'year': 'Year',
            'vehicle_age': 'Vehicle_Age'
        }
        normalized = {}
        for k, v in raw_input.items():
            norm_k = key_map.get(k.lower(), k)
            normalized[norm_k] = v

        required_keys = ['Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']
        for key in required_keys:
            if key not in normalized:
                raise ValueError(f"Missing required input parameter: '{key}'")
                
        # Validate Present_Price
        try:
            present_price = float(normalized['Present_Price'])
        except (ValueError, TypeError):
            raise ValueError(f"Present_Price must be a numeric value, got {normalized['Present_Price']}")
        if present_price <= 0 or present_price > 150.0:
            raise ValueError(f"Present_Price must be between 0.01 and 150.00 Lakhs, got {present_price}")
            
        # Validate Kms_Driven
        try:
            kms_driven = int(normalized['Kms_Driven'])
        except (ValueError, TypeError):
            raise ValueError(f"Kms_Driven must be an integer, got {normalized['Kms_Driven']}")
        if kms_driven < 0 or kms_driven > 1000000:
            raise ValueError(f"Kms_Driven must be between 0 and 1,000,000 km, got {kms_driven}")
            
        # Validate Categoricals
        fuel = str(normalized['Fuel_Type']).capitalize()
        if fuel not in VALID_FUEL_TYPES:
            raise ValueError(f"Invalid Fuel_Type '{fuel}'. Must be one of {VALID_FUEL_TYPES}")
            
        seller = str(normalized['Seller_Type']).capitalize()
        if seller not in VALID_SELLER_TYPES:
            raise ValueError(f"Invalid Seller_Type '{seller}'. Must be one of {VALID_SELLER_TYPES}")
            
        trans = str(normalized['Transmission']).capitalize()
        if trans not in VALID_TRANSMISSIONS:
            raise ValueError(f"Invalid Transmission '{trans}'. Must be one of {VALID_TRANSMISSIONS}")
            
        # Validate Owner
        try:
            owner = int(normalized['Owner'])
        except (ValueError, TypeError):
            raise ValueError(f"Owner must be an integer, got {normalized['Owner']}")
        if owner not in VALID_OWNERS:
            raise ValueError(f"Owner count must be in {VALID_OWNERS}, got {owner}")
            
        # Validate Year / Vehicle_Age
        MIN_VALID_YEAR = 2003
        MAX_VALID_YEAR = REF_YEAR  # 2018
        if 'Vehicle_Age' in normalized:
            try:
                vehicle_age = int(normalized['Vehicle_Age'])
            except (ValueError, TypeError):
                raise ValueError(f"Vehicle_Age must be an integer, got {normalized['Vehicle_Age']}")
            if vehicle_age < 0 or vehicle_age > (REF_YEAR - MIN_VALID_YEAR):
                raise ValueError(
                    f"Vehicle_Age must be between 0 and {REF_YEAR - MIN_VALID_YEAR} years (got {vehicle_age}). "
                    f"Negative vehicle ages or years past {REF_YEAR} are not supported."
                )
        elif 'Year' in normalized:
            try:
                year = int(normalized['Year'])
            except (ValueError, TypeError):
                raise ValueError(f"Manufacturing Year must be an integer, got {normalized['Year']}")
            if year < MIN_VALID_YEAR or year > MAX_VALID_YEAR:
                raise ValueError(
                    f"Manufacturing Year must be between {MIN_VALID_YEAR} and {MAX_VALID_YEAR} (got {year}). "
                    f"The model is calibrated on historical secondary market transactions with reference horizon {REF_YEAR}; "
                    f"future manufacturing years (2019+) would produce negative vehicle ages and are invalid."
                )
            vehicle_age = MAX_VALID_YEAR - year
        else:
            raise ValueError("Either 'Vehicle_Age' or 'Year' must be provided.")
            
        df_input = pd.DataFrame([{
            'Present_Price': present_price,
            'Kms_Driven': kms_driven,
            'Fuel_Type': fuel,
            'Seller_Type': seller,
            'Transmission': trans,
            'Owner': owner,
            'Vehicle_Age': vehicle_age
        }])[FEATURE_COLUMNS]
        
        return df_input

    def predict(self, raw_input: dict = None, **kwargs) -> dict:
        """
        Predicts resale selling price from input dictionary or keyword arguments.
        
        Args:
            raw_input: Dictionary with vehicle attributes (or pass as kwargs).
            **kwargs: Alternative keyword argument specification.
            
        Returns:
            Dictionary containing predicted price in Lakhs, formatted INR string, and valuation analytics.
        """
        if raw_input is None:
            raw_input = kwargs
        elif kwargs:
            raw_input = {**raw_input, **kwargs}
            
        df_input = self.validate_inputs(raw_input)
        
        raw_pred = self.pipeline.predict(df_input)[0]
        # Economic salvage price floor: vehicle cannot sell for negative value
        pred_lakhs = max(0.10, float(raw_pred))
        
        present_price = df_input['Present_Price'].iloc[0]
        deprec_lakhs = max(0.0, present_price - pred_lakhs)
        depreciation_pct = min(100.0, max(0.0, (deprec_lakhs / present_price) * 100.0))
        retention_pct = max(0.0, 100.0 - depreciation_pct)
        formatted_price = format_inr(pred_lakhs)
        
        return {
            "predicted_price_lakhs": round(pred_lakhs, 2),
            "selling_price_lakhs": round(pred_lakhs, 2),
            "predicted_price_inr": formatted_price,
            "formatted_inr": formatted_price,
            "depreciation_lakhs": round(deprec_lakhs, 2),
            "depreciation_percentage": round(depreciation_pct, 1),
            "retention_percentage": round(retention_pct, 1),
            "input_features": df_input.iloc[0].to_dict()
        }

if __name__ == "__main__":
    predictor = CarPricePredictor()
    test_car = {
        'Present_Price': 9.85,
        'Kms_Driven': 18000,
        'Fuel_Type': 'Diesel',
        'Seller_Type': 'Dealer',
        'Transmission': 'Manual',
        'Owner': 0,
        'Year': 2016
    }
    result = predictor.predict(test_car)
    print("Inference Test Result:")
    print(f"  Predicted Price: {result['predicted_price_lakhs']} Lakhs ({result['predicted_price_inr']})")
    print(f"  Depreciation:    {result['depreciation_percentage']}%")
