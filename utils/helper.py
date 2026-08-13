"""
Helper Module for Resume Screening System.

This module provides utility functions for:
1. Validating uploaded files (.txt format only)
2. Safely saving and reading text files from disk
3. Cleaning up temporary uploaded files after processing
"""

import os
from werkzeug.utils import secure_filename

# Allow only plain text (.txt) files as specified in project rules
ALLOWED_EXTENSIONS = {'txt'}


def allowed_file(filename: str) -> bool:
    """
    Checks if an uploaded filename has an allowed extension (.txt).

    Parameters:
        filename (str): The name of the uploaded file.

    Returns:
        bool: True if file ends with .txt, False otherwise.
    """
    if '.' not in filename:
        return False
    # Split filename by '.' and check if the extension in lowercase is 'txt'
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def save_and_read_file(file_obj, upload_folder: str) -> str:
    """
    Safely saves an uploaded file to disk, reads its content, and removes the temp file.

    Parameters:
        file_obj: The Flask FileStorage object from request.files.
        upload_folder (str): Directory path where temporary files are stored.

    Returns:
        str: The raw text content of the uploaded file.

    Raises:
        ValueError: If file is missing or invalid format.
        IOError: If file reading fails.
    """
    if not file_obj or file_obj.filename == '':
        raise ValueError("No file selected for upload.")

    if not allowed_file(file_obj.filename):
        raise ValueError("Invalid file format. Only .txt files are accepted.")

    # Secure the filename to prevent directory traversal security issues
    safe_name = secure_filename(file_obj.filename)
    file_path = os.path.join(upload_folder, safe_name)

    # Ensure the upload folder exists
    os.makedirs(upload_folder, exist_ok=True)

    try:
        # Save temporary file to disk
        file_obj.save(file_path)

        # Read content using UTF-8 encoding (fallback to latin-1 if characters fail)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()

        return content

    finally:
        # Clean up: remove temporary file after reading so uploads folder stays clean
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass