"""
Main Flask Web Application for Resume Screening and Ranking System.

This script:
1. Initializes the Flask web server.
2. Defines routes for displaying the home page ('/').
3. Handles POST requests to '/analyze' for file upload and processing.
4. Orchestrates text cleaning, TF-IDF feature extraction, and ML scoring.
5. Returns results or meaningful error messages to the frontend template.
"""

import os
from flask import Flask, render_template, request
from utils.helper import save_and_read_file
from utils.preprocess import preprocess_text
from utils.feature_extractor import extract_all_features
from utils.predictor import predict_resume_score

# Initialize Flask application
app = Flask(__name__)

# Configure a temporary directory to store uploaded files safely before reading
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Set maximum allowed payload size to 5MB to prevent overly large file uploads
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


@app.route("/", methods=["GET"])
def index():
    """
    Renders the default home page with file upload buttons and no result card.
    """
    return render_template("index.html", result=None, error=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Handles file upload form submission, executes the ML pipeline, and renders results.

    Pipeline Steps:
    1. Validate that both Resume and Job Description files exist in request.
    2. Read text safely from uploaded .txt files.
    3. Clean and tokenize text using NLTK preprocessing.
    4. Compute TF-IDF Cosine Similarity and extract NLP features.
    5. Pass features to trained Random Forest model to predict Resume Score (0-100).
    6. Render index.html with the analyzed metrics card.
    """
    # Step 1: Check if both file input fields are present in the form request
    if "resume" not in request.files or "jd" not in request.files:
        return render_template(
            "index.html",
            result=None,
            error="Please upload both a Resume and a Job Description file.",
        )

    resume_file = request.files["resume"]
    jd_file = request.files["jd"]

    # Step 2: Validate that neither file input was left blank
    if resume_file.filename == "" or jd_file.filename == "":
        return render_template(
            "index.html",
            result=None,
            error="One or both files were not selected. Please select .txt files for both.",
        )

    try:
        # Step 3: Safely save, read raw content, and clean up temporary files
        raw_resume = save_and_read_file(resume_file, app.config["UPLOAD_FOLDER"])
        raw_jd = save_and_read_file(jd_file, app.config["UPLOAD_FOLDER"])

        # Check for empty files after reading
        if not raw_resume.strip() or not raw_jd.strip():
            return render_template(
                "index.html",
                result=None,
                error="Uploaded files appear to be empty. Please check your .txt files.",
            )

        # Step 4: Text Preprocessing (lowercase, remove punctuation, remove stopwords)
        clean_resume = preprocess_text(raw_resume)
        clean_jd = preprocess_text(raw_jd)

        # Step 5: Feature Extraction (TF-IDF vectorization, cosine similarity, matching skills)
        features_data = extract_all_features(raw_resume, clean_resume, clean_jd)

        # Step 6: Machine Learning Prediction (Random Forest Regressor)
        prediction_data = predict_resume_score(features_data["feature_vector"])

        # Step 7: Bundle all metrics into a clean dictionary for HTML rendering
        result = {
            "score": prediction_data["score"],
            "recommendation": prediction_data["recommendation"],
            "color_class": prediction_data["color_class"],
            "similarity_percentage": features_data["similarity_percentage"],
            "matching_skills_count": features_data["matching_skills_count"],
            "matched_skills_list": features_data["matched_skills_list"],
            "word_count": features_data["word_count"],
            "projects_count": features_data["projects_count"],
            "education_found": features_data["education_found"],
            "experience_found": features_data["experience_found"],
        }

        # Render the template with the computed analysis results
        return render_template("index.html", result=result, error=None)

    except ValueError as val_err:
        # Catch validation errors (e.g., uploading a .pdf or .docx instead of .txt)
        return render_template("index.html", result=None, error=str(val_err))
    except Exception as e:
        # Catch unexpected server or model errors gracefully
        return render_template(
            "index.html",
            result=None,
            error=f"An error occurred while processing: {str(e)}",
        )


if __name__ == "__main__":
    # Run the web server in debug mode for beginner development and testing
    app.run(host="127.0.0.1", port=5000, debug=True)