

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Automatically download required NLTK datasets (silent download if already present)
# Automatically download required NLTK datasets safely
try:
    nltk.data.find('corpora/stopwords')
except (LookupError, OSError):
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, OSError):
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except (LookupError, OSError):
    nltk.download('punkt_tab', quiet=True)


def preprocess_text(text: str) -> str:
    """
    Cleans and standardizes raw input text for NLP analysis.

    Parameters:
        text (str): The raw string from a .txt file (Resume or Job Description).

    Returns:
        str: A cleaned string containing only lowercase alphabetical words
             without punctuation, numbers, or stopwords.
    """
    # Safety check: if input is empty or invalid, return an empty string
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Convert text to lowercase for uniformity (e.g., "Python" -> "python")
    text = text.lower()

    # Step 2: Remove numbers and punctuation using Regular Expression
    # [^a-zA-Z\s] means "replace anything that is NOT a letter or whitespace with nothing"
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Step 3: Remove extra spaces and newlines by collapsing multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 4: Tokenize text (break sentence into individual word tokens)
    # Using basic split/word_tokenize for beginner simplicity
    try:
        tokens = word_tokenize(text)
    except Exception:
        # Fallback to simple whitespace splitting if NLTK tokenizer fails
        tokens = text.split()

    # Step 5: Remove NLTK English stopwords
    # Using a set for O(1) fast lookup speed
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word not in stop_words]

    # Step 6: Join cleaned word tokens back into a single clean text string
    cleaned_text = " ".join(cleaned_tokens)

    return cleaned_text
