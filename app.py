import os
import PyPDF2
import docx
from flask import Flask, render_template, request
from utils.preprocess import preprocess_text
from utils.feature_extractor import extract_all_features
from utils.predictor import predict_resume_score

# Initialize Flask application
app = Flask(__name__)

# Configure a temporary directory (Keeping this just in case other parts of your app need it)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Set maximum allowed payload size to 5MB to prevent overly large file uploads
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# --- NEW SMART TEXT EXTRACTOR ---
def extract_text_from_file(uploaded_file):
    """Extracts text from .txt, .pdf, or .docx files straight from memory."""
    filename = uploaded_file.filename.lower()
    
    # 1. Handle PDF files
    if filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

    # 2. Handle Word documents (.docx)
    elif filename.endswith('.docx'):
        doc = docx.Document(uploaded_file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    # 3. Handle plain text (.txt)
    elif filename.endswith('.txt'):
        return uploaded_file.read().decode('utf-8', errors='ignore')
        
    else:
        raise ValueError("Unsupported file format. Please upload .txt, .pdf, or .docx")

@app.route("/", methods=["GET"])
def index():
    """
    Renders the default home page with file upload buttons and no result card.
    """
    return render_template("index.html", result=None, error=None)


@app.route("/analyze", methods=["POST"])
def analyze():
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
            error="One or both files were not selected. Please select .txt, .pdf, or .docx files.",
        )

    try:
        # Step 3: Extract text based on file type
        raw_resume = extract_text_from_file(resume_file)
        raw_jd = extract_text_from_file(jd_file)

        # Check for empty files after reading
        if not raw_resume.strip() or not raw_jd.strip():
            return render_template(
                "index.html",
                result=None,
                error="Uploaded files appear to be empty or unreadable.",
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
        # Catch validation errors 
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
