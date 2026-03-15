# 📄 AI Resume Screening & Candidate Ranking System

An AI-powered Resume Screening System that automatically analyzes resumes, extracts relevant skills, compares them with a job description, and ranks candidates based on their suitability for the role. 
This project demonstrates how **Natural Language Processing (NLP)** and **Machine Learning techniques** can assist recruiters in efficiently shortlisting candidates.

---

## 🎯 Project Objective

Recruiters often receive **hundreds of resumes** for a single job opening.
Manually reviewing them is:
* Time-consuming
* Inconsistent
* Prone to human bias

This system automates the screening process by:

* Extracting relevant **skills from resumes**
* Comparing resumes with **job descriptions**
* Calculating **match scores**
* Ranking candidates based on **role fit**
* Identifying **missing skills**

---

## 🚀 Key Features

* Resume text cleaning and preprocessing 
* Skill extraction using NLP techniques
* Job description parsing
*  Resume-to-job similarity scoring
* Candidate ranking based on match score
*  Skill gap identification
*  Interactive **Streamlit dashboard**

---

## 🧠 Machine Learning Workflow

1. Resume Dataset Loading
2. Text Cleaning & Preprocessing
3. Stopword Removal
4. Skill Extraction from Resume Text
5. TF-IDF Vectorization
6. Cosine Similarity Calculation
7. Resume Ranking based on Match Score
8. Skill Gap Analysis

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* NLTK
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Streamlit

---

## 📊 System Architecture

```
Resume Dataset
       │
       ▼
Text Cleaning & Preprocessing
       │
       ▼
Skill Extraction
       │
       ▼
TF-IDF Vectorization
       │
       ▼
Cosine Similarity
       │
       ▼
Candidate Ranking
       │
       ▼
Skill Gap Identification
```

---

## 📂 Project Structure

```
resume-screening-system
│
├── app.py
├── Resume_small.csv
├── Resume_Screening_System.ipynb
├── requirements.txt
└── README.md
```

---

## 📊 Example Output

| Candidate Category | Skills                        | Missing Skills | Match Score |
| ------------------ | ----------------------------- | -------------- | ----------- |
| ENGINEERING        | Python, SQL, Machine Learning | Deep Learning  | 15.98%      |
| AGRICULTURE        | Python, Java, SQL             | Deep Learning  | 15.95%      |

The system ranks candidates based on **similarity between resume content and job description**.

---

## 📈 Dashboard Features

The Streamlit application includes:
📄 Resume Screening - Enter a job description and rank resumes automatically.

💡 Example Job Descriptions - Predefined examples to test the system.

📊 Dataset Dashboard - Visualizations showing resume category distribution.

ℹ️ About Project - Project explanation and methodology.

---

## 📌 Example Job Description

```
Looking for a Data Scientist skilled in Python,
Machine Learning, SQL, Pandas, and Deep Learning.
```

The system will analyze resumes and produce:

* Candidate ranking
* Match score
* Missing skills

---

## 🎯 Use Cases

* HR recruitment automation
* Applicant Tracking Systems (ATS)
* Resume shortlisting tools
* HR-tech startup solutions
* Candidate skill gap analysis

---

## ⭐ Future Improvements

* Named Entity Recognition for better skill extraction
* Deep Learning based resume matching
* Resume PDF parsing
* Candidate comparison dashboard
* Integration with real HR platforms

---

💻 [GitHub Code](https://github.com/riyach3150/FUTURE_ML_03)

🚀 [Live Demo](https://futureml03-w4yhxysxttsnjxx3wwrhmf.streamlit.app/)
