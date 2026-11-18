"""
Unit tests for model loading, prediction functionality, and input validation.
"""

import sys
import pytest
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.predict import CarPricePredictor, format_inr

def test_model_loading():
    model_path = ROOT_DIR / "models" / "car_price_model.joblib"
    assert model_path.exists(), f"Saved model pipeline does not exist at {model_path}"
    
    predictor = CarPricePredictor(model_path)
    assert predictor.pipeline is not None

def test_valid_prediction():
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
    res = predictor.predict(test_car)
    
    assert "predicted_price_lakhs" in res
    assert "predicted_price_inr" in res
    assert res['predicted_price_lakhs'] > 0
    assert res['predicted_price_inr'].startswith("₹")
    assert 0 <= res['depreciation_percentage'] <= 100

def test_boundary_inputs():
    predictor = CarPricePredictor()
    # Brand new expensive luxury SUV
    expensive_car = {
        'Present_Price': 50.0,
        'Kms_Driven': 500,
        'Fuel_Type': 'Diesel',
        'Seller_Type': 'Dealer',
        'Transmission': 'Automatic',
        'Owner': 0,
        'Year': 2018
    }
    res = predictor.predict(expensive_car)
    assert res['predicted_price_lakhs'] > 20.0, "Luxury SUV should retain high value"

    # Very old high-mileage hatchback
    old_car = {
        'Present_Price': 3.5,
        'Kms_Driven': 150000,
        'Fuel_Type': 'Petrol',
        'Seller_Type': 'Individual',
        'Transmission': 'Manual',
        'Owner': 3,
        'Year': 2004
    }
    res_old = predictor.predict(old_car)
    assert res_old['predicted_price_lakhs'] < 2.0, "Old high mileage car should have heavy depreciation"

def test_invalid_inputs_raise_errors():
    predictor = CarPricePredictor()
    
    # Negative Present Price
    with pytest.raises(ValueError, match="Present_Price"):
        predictor.predict({
            'Present_Price': -5.0,
            'Kms_Driven': 10000,
            'Fuel_Type': 'Petrol',
            'Seller_Type': 'Dealer',
            'Transmission': 'Manual',
            'Owner': 0,
            'Year': 2015
        })
        
    # Invalid Fuel Type
    with pytest.raises(ValueError, match="Fuel_Type"):
        predictor.predict({
            'Present_Price': 5.0,
            'Kms_Driven': 10000,
            'Fuel_Type': 'Hydrogen',
            'Seller_Type': 'Dealer',
            'Transmission': 'Manual',
            'Owner': 0,
            'Year': 2015
        })
        
    # Negative Mileage
    with pytest.raises(ValueError, match="Kms_Driven"):
        predictor.predict({
            'Present_Price': 5.0,
            'Kms_Driven': -500,
            'Fuel_Type': 'Petrol',
            'Seller_Type': 'Dealer',
            'Transmission': 'Manual',
            'Owner': 0,
            'Year': 2015
        })

    # Future Manufacturing Year (e.g., 2019 or 2026 beyond REFERENCE_YEAR 2018)
    with pytest.raises(ValueError, match="Manufacturing Year must be between"):
        predictor.predict({
            'Present_Price': 5.0,
            'Kms_Driven': 10000,
            'Fuel_Type': 'Petrol',
            'Seller_Type': 'Dealer',
            'Transmission': 'Manual',
            'Owner': 0,
            'Year': 2024
        })

    # Historical Out-of-Range Year (< 2003)
    with pytest.raises(ValueError, match="Manufacturing Year must be between"):
        predictor.predict({
            'Present_Price': 5.0,
            'Kms_Driven': 10000,
            'Fuel_Type': 'Petrol',
            'Seller_Type': 'Dealer',
            'Transmission': 'Manual',
            'Owner': 0,
            'Year': 1998
        })

    # Negative Vehicle Age
    with pytest.raises(ValueError, match="Vehicle_Age must be between"):
        predictor.predict({
            'Present_Price': 5.0,
            'Kms_Driven': 10000,
            'Fuel_Type': 'Petrol',
            'Seller_Type': 'Dealer',
            'Transmission': 'Manual',
            'Owner': 0,
            'Vehicle_Age': -1
        })

def test_format_inr():
    assert format_inr(3.35) == "₹3,35,000"
    assert format_inr(23.0) == "₹23,00,000"
    assert format_inr(0.5) == "₹50,000"
