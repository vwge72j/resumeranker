
import os
from werkzeug.utils import secure_filename

# Allow only plain text (.txt) files as specified in project rules
ALLOWED_EXTENSIONS = {'txt'}


def allowed_file(filename: str) -> bool:
    
    if '.' not in filename:
        return False
    # Split filename by '.' and check if the extension in lowercase is 'txt'
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def save_and_read_file(file_obj, upload_folder: str) -> str:
  
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
