"""
Prediction Module for Resume Screening System.

This module is responsible for:
1. Loading the trained machine learning model (`model.pkl`) using pickle.
2. Predicting the Resume Score (0-100) from a 6-element feature vector.
3. Clamping (restricting) the final score between 0 and 100.
4. Assigning a qualitative recommendation badge based on the score.
"""

import os
import pickle

# Path to the serialized RandomForestRegressor model
MODEL_PATH = "model.pkl"

# Global variable to cache the loaded model in memory
_loaded_model = None


def load_model():
    """
    Loads the trained model from disk into memory.
    Uses caching so the application doesn't reload the file from disk on every prediction.
    """
    global _loaded_model

    if _loaded_model is not None:
        return _loaded_model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. Please run 'train_model.py' first!"
        )

    try:
        with open(MODEL_PATH, "rb") as f:
            _loaded_model = pickle.load(f)
        return _loaded_model
    except Exception as e:
        raise RuntimeError(f"Failed to load machine learning model: {str(e)}")


def clamp_score(score: float) -> float:
    """
    Restricts the predicted score strictly within the 0 to 100 range.

    Parameters:
        score (float): Raw output prediction from the regression model.

    Returns:
        float: Rounded score clamped between 0.0 and 100.0.
    """
    if score < 0.0:
        return 0.0
    elif score > 100.0:
        return 100.0
    return round(score, 1)


def get_recommendation(score: float) -> dict:
    """
    Assigns a category badge and color theme based on the Resume Score.

    ===================================================================
    RECOMMENDATION LOGIC:
    - Score >= 80  -> Excellent Match (Green)
    - Score >= 60  -> Good Match      (Yellow/Info)
    - Score >= 40  -> Average Match   (Orange/Warning)
    - Else         -> Poor Match      (Red/Danger)
    ===================================================================

    Parameters:
        score (float): Clamped Resume Score (0 - 100).

    Returns:
        dict: Contains the recommendation text and Bootstrap color class.
    """
    if score >= 80.0:
        return {
            "label": "Excellent Match",
            "color_class": "success",  # Green
            "hex_color": "#28a745"
        }
    elif score >= 60.0:
        return {
            "label": "Good Match",
            "color_class": "info",     # Light Blue / Yellow-Green
            "hex_color": "#17a2b8"
        }
    elif score >= 40.0:
        return {
            "label": "Average Match",
            "color_class": "warning",  # Orange
            "hex_color": "#ffc107"
        }
    else:
        return {
            "label": "Poor Match",
            "color_class": "danger",   # Red
            "hex_color": "#dc3545"
        }


def predict_resume_score(feature_vector: list) -> dict:
    """
    Master prediction function that takes a feature vector, runs it through the
    trained Random Forest Regressor, and returns the final score and recommendation.

    Parameters:
        feature_vector (list): [Similarity, MatchingSkills, ResumeWordCount,
                                Projects, Education, Experience]

    Returns:
        dict: {
            "score": float (0-100),
            "recommendation": str,
            "color_class": str,
            "hex_color": str
        }
    """
    # Step 1: Load the trained RandomForest model
    model = load_model()

    # Step 2: Ensure feature vector is formatted as a 2D array for scikit-learn
    # Shape expected: [[sim, skills, words, projects, edu, exp]]
    input_features = [feature_vector]

    # Step 3: Make prediction
    raw_prediction = model.predict(input_features)[0]

    # Step 4: Clamp score between 0 and 100
    final_score = clamp_score(float(raw_prediction))

    # Step 5: Get recommendation badge and theme color
    rec_data = get_recommendation(final_score)

    return {
        "score": final_score,
        "recommendation": rec_data["label"],
        "color_class": rec_data["color_class"],
        "hex_color": rec_data["hex_color"]
    }