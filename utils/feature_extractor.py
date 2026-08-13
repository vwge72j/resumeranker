
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Predefined list of technical skills to match between Resume and Job Description
DEFAULT_SKILLS = [
    "python", "java", "c++", "html", "css", "javascript", "sql", "mysql",
    "mongodb", "flask", "django", "react", "node.js", "machine learning",
    "artificial intelligence", "data structures", "algorithms", "git",
    "github", "linux", "networking", "cyber security", "operating system",
    "oop", "dbms"
]

# Keywords to detect Education and Experience qualifications
EDUCATION_KEYWORDS = [
    "b.tech", "b.e.", "mca", "m.tech", "diploma", "bachelor", "master",
    "degree", "university", "college", "btech", "mtech"
]

EXPERIENCE_KEYWORDS = [
    "internship", "experience", "intern", "project", "developed",
    "built", "worked", "engineered", "implemented", "responsible"
]

PROJECT_KEYWORDS = [
    "project", "developed", "built", "created", "designed", "application", "system"
]


def compute_tfidf_and_similarity(resume_text: str, jd_text: str) -> tuple:
   
    # Handle edge case where one or both texts are empty
    if not resume_text or not jd_text:
        return 0.0, 0.0

    # Step 1: Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()

    try:
        # Step 2: Fit vectorizer on both documents and transform them into numerical matrices
        # Shape: (2 documents, number_of_unique_words_in_vocabulary)
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])

        # Step 3: Calculate Cosine Similarity between Resume vector (row 0) and JD vector (row 1)
        # Cosine Similarity measures the angle between two multi-dimensional vectors.
        # A value of 1.0 means identical direction/meaning; 0.0 means completely orthogonal (no match).
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        similarity_score = float(similarity_matrix[0][0])

        # Convert decimal similarity to a percentage rounded to 2 decimal places (e.g., 0.83 -> 83.0)
        similarity_percentage = round(similarity_score * 100, 2)

        return similarity_score, similarity_percentage

    except ValueError:
        # Occurs if text contains no valid vocabulary tokens after cleaning
        return 0.0, 0.0


def extract_matching_skills(resume_text: str, jd_text: str, skills_list: list = None) -> tuple:
    """
    Counts how many technical skills appear in BOTH the resume and job description.

    Parameters:
        resume_text (str): Cleaned resume text.
        jd_text (str): Cleaned job description text.
        skills_list (list, optional): Custom list of skills. Defaults to DEFAULT_SKILLS.

    Returns:
        tuple: (matching_skills_count: int, matched_skills_names: list)
    """
    if skills_list is None:
        skills_list = DEFAULT_SKILLS

    matched_skills = []

    for skill in skills_list:
        # Use regular expression with word boundaries (\b) to match exact skill terms
        # Example: \bjava\b matches "java" but not "javascript"
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'

        in_resume = re.search(pattern, resume_text.lower()) is not None
        in_jd = re.search(pattern, jd_text.lower()) is not None

        # Skill is a match only if it is required by JD AND present in Resume
        if in_resume and in_jd:
            matched_skills.append(skill)

    return len(matched_skills), matched_skills


def check_keywords_presence(text: str, keywords: list) -> int:
    """
    Checks if any keyword from a predefined category exists in the text.

    Returns:
        int: 1 if at least one keyword is found, 0 otherwise (Binary flag for ML model).
    """
    for kw in keywords:
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        if re.search(pattern, text.lower()):
            return 1
    return 0


def count_project_mentions(text: str) -> int:
    """
    Estimates how many projects are mentioned by counting project-related action keywords.
    """
    count = 0
    for kw in PROJECT_KEYWORDS:
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        matches = re.findall(pattern, text.lower())
        count += len(matches)
    # Cap the count at 5 so extreme keyword repetition doesn't distort ML predictions
    return min(count, 5)


def get_word_count(text: str) -> int:
    """Returns the total number of words in the raw or cleaned resume text."""
    if not text:
        return 0
    return len(text.split())


def extract_all_features(raw_resume: str, clean_resume: str, clean_jd: str) -> dict:
    """
    Master function that extracts all beginner-level features and structures them
    both for UI display and as a numerical feature vector for the Machine Learning model.

    Returns:
        dict: Contains UI display metrics and the exact 6-element list for `model.predict()`.
    """
    # 1. Compute TF-IDF Cosine Similarity
    sim_score, sim_percentage = compute_tfidf_and_similarity(clean_resume, clean_jd)

    # 2. Extract Matching Skills
    match_count, matched_skills = extract_matching_skills(clean_resume, clean_jd)

    # 3. Resume Word Count (using raw resume to count actual length)
    word_count = get_word_count(raw_resume)

    # 4. Count Project mentions
    projects_count = count_project_mentions(raw_resume)

    # 5. Education & Experience binary flags (1 = Found, 0 = Not Found)
    edu_flag = check_keywords_presence(raw_resume, EDUCATION_KEYWORDS)
    exp_flag = check_keywords_presence(raw_resume, EXPERIENCE_KEYWORDS)

    # Build the 6-element numerical feature vector matching `training.csv` column order:
    # [Similarity, MatchingSkills, ResumeWordCount, Projects, Education, Experience]
    feature_vector = [
        sim_score,
        match_count,
        word_count,
        projects_count,
        edu_flag,
        exp_flag
    ]

    return {
        "similarity_score": sim_score,
        "similarity_percentage": sim_percentage,
        "matching_skills_count": match_count,
        "matched_skills_list": matched_skills,
        "word_count": word_count,
        "projects_count": projects_count,
        "education_found": bool(edu_flag),
        "experience_found": bool(exp_flag),
        "feature_vector": feature_vector
    }
