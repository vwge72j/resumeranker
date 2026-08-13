# Smart Resume Screening & Ranking System

An end-to-end, beginner-friendly **Machine Learning and Natural Language Processing (NLP)** web application built with Python and Flask. The system compares an applicant's resume against a job description, calculates mathematical similarity, extracts key qualifications, and predicts an overall **Resume Score (0–100)** using a trained Regression model.

---

## Table of Contents
- [Project Workflow](#project-workflow)
- [Folder Structure](#folder-structure)
- [Installation & Setup Guide](#installation--setup-guide)
- [How It Works: ML & NLP Concepts Explained](#how-it-works-ml--nlp-concepts-explained)
  - [1. TF-IDF Vectorization](#1-tf-idf-vectorization)
  - [2. Cosine Similarity](#2-cosine-similarity)
  - [3. Feature Extraction](#3-feature-extraction)
  - [4. Random Forest Regressor](#4-random-forest-regressor)
- [Screenshots Placeholder](#screenshots-placeholder)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [References](#references)

---

## Project Workflow

```text
       [ Upload Resume (.txt) & Job Description (.txt) ]
                              │
                              ▼
                [ Text Preprocessing (NLTK) ]
         • Lowercase  • Remove Punctuation/Numbers
         • Remove Stopwords  • Tokenize & Join
                              │
                              ▼
                [ TF-IDF Vectorization (NLP) ]
            Converts cleaned text into numerical vectors
                              │
                              ▼
              [ Cosine Similarity Calculation ]
             Measures angle/similarity between vectors
                              │
                              ▼
               [ Feature Extraction Pipeline ]
         • Cosine Similarity %   • Matching Technical Skills
         • Word Count            • Project Mentions
         • Education Keywords    • Experience Keywords
                              │
                              ▼
           [ Random Forest Regressor (model.pkl) ]
               Predicts base score from 6 features
                              │
                              ▼
            [ Score Clamping (0–100) & UI Display ]
       Assigns Recommendation: Excellent / Good / Average / Poor