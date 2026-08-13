

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def generate_synthetic_dataset(filename: str = "training.csv", num_samples: int = 400):
    
    np.random.seed(42) # For reproducible results

    data = []

    for _ in range(num_samples):
        # Generate random feature values within realistic ranges
        similarity = round(np.random.uniform(0.10, 0.95), 2)
        
        # Matching skills scale loosely with similarity
        max_skills = int(similarity * 15)
        matching_skills = np.random.randint(0, max(1, max_skills) + 1)
        
        word_count = np.random.randint(120, 600)
        projects = np.random.randint(0, 6)
        education = np.random.choice([0, 1], p=[0.15, 0.85]) # 85% have education keywords
        experience = np.random.choice([0, 1], p=[0.30, 0.70]) # 70% have experience keywords

        # Calculate a realistic synthetic target score (0-100)
        # We assign logical weights to each feature + a little random Gaussian noise
        base_score = (
            (similarity * 45.0) +
            (matching_skills * 2.2) +
            (min(word_count, 450) / 450.0 * 10.0) +
            (projects * 2.0) +
            (education * 5.0) +
            (experience * 8.0)
        )
        
        # Add random noise (-3 to +3) to simulate real-world human variance
        noise = np.random.normal(0, 2.0)
        final_score = round(np.clip(base_score + noise, 10.0, 98.0), 1)

        data.append({
            "Similarity": similarity,
            "MatchingSkills": matching_skills,
            "ResumeWordCount": word_count,
            "Projects": projects,
            "Education": education,
            "Experience": experience,
            "FinalScore": final_score
        })

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"[+] Successfully generated synthetic dataset '{filename}' with {num_samples} samples.")
    return df


def train_and_evaluate_model():
    """Loads training.csv, trains RandomForestRegressor, evaluates metrics, and saves model.pkl."""
    csv_path = "training.csv"
    model_path = "model.pkl"

    # Step 1: Ensure dataset exists (or generate it)
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 500:
        print("[!] Generating synthetic training data...")
        df = generate_synthetic_dataset(csv_path, num_samples=400)
    else:
        df = pd.read_csv(csv_path)
        print(f"[+] Loaded existing dataset '{csv_path}' ({len(df)} rows).")

    # Step 2: Separate Features (X) and Target Variable (y)
    feature_columns = ["Similarity", "MatchingSkills", "ResumeWordCount", "Projects", "Education", "Experience"]
    X = df[feature_columns]
    y = df["FinalScore"]

    # Step 3: Split into 80% Training and 20% Testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Step 4: Initialize and train RandomForestRegressor
    # Random Forest combines multiple decision trees to make accurate, stable regression predictions
    print("[*] Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 5: Evaluate Model Performance on unseen test data
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n=============================================")
    print("      ML MODEL EVALUATION RESULTS            ")
    print("=============================================")
    print(f"  Mean Absolute Error (MAE):  {mae:.2f}")
    print(f"  Root Mean Squared Error:   {rmse:.2f}")
    print(f"  R² Score (Accuracy Score): {r2:.4f}")
    print("=============================================\n")

    # Step 6: Save trained model using Pickle
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"[+] Successfully saved trained ML model to '{model_path}'.")


if __name__ == "__main__":
    train_and_evaluate_model()
