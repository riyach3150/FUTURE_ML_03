import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords

# -----------------------------
# Download stopwords safely
# -----------------------------

try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------

st.markdown("""
<style>

.main-title{
font-size:40px;
font-weight:700;
}

.result-box{
padding:20px;
border-radius:12px;
background:#111;
border:1px solid #333;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Dataset (Cached)
# -----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("Resume_small.csv")
    return df

df = load_data()

# -----------------------------
# Text Cleaning Function
# -----------------------------

def clean_text(text):

    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]',' ',text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)

df["Cleaned_Resume"] = df["Resume_str"].apply(clean_text)

# -----------------------------
# Skill Extraction
# -----------------------------

skills_list = [
'python','java','c++','machine learning','deep learning','nlp',
'sql','tensorflow','pytorch','keras','opencv','excel',
'communication','management','data analysis','tableau',
'power bi','pandas','numpy','scikit-learn','data science',
'artificial intelligence','ai','statistics','big data'
]

def extract_skills(text):

    found = []

    for skill in skills_list:
        if skill in text:
            found.append(skill)

    return found

def get_missing_skills(resume_skills, job_description):

    job_skills = []

    for skill in skills_list:
        if skill in job_description:
            job_skills.append(skill)

    missing = []

    for skill in job_skills:
        if skill not in resume_skills:
            missing.append(skill)

    return missing

df["Skills"] = df["Cleaned_Resume"].apply(extract_skills)

# -----------------------------
# Sidebar Navigation
# -----------------------------

st.sidebar.title("📄 Navigation")

page = st.sidebar.radio(
"Go to",
[
"Resume Screening",
"Example Job Descriptions",
"Candidate Dashboard",
"About Project"
]
)

st.sidebar.markdown("---")
st.sidebar.info("Future Interns — ML Resume Screening Project")

# -----------------------------
# PAGE 1 — RESUME SCREENING
# -----------------------------

if page == "Resume Screening":

    st.title("📄 AI Resume Screening System")

    st.write("""
This system ranks resumes based on how well they match a **job description** using **Natural Language Processing and Machine Learning**.
""")

    job_description = st.text_area(
        "Enter Job Description",
        placeholder="Example: Looking for Data Scientist with Python, Machine Learning, SQL"
    )

    if st.button("🔍 Screen Resumes"):

        if job_description.strip() == "":
            st.warning("Please enter a job description.")
            st.stop()

        clean_job = clean_text(job_description)

        vectorizer = TfidfVectorizer()

        resume_vectors = vectorizer.fit_transform(df["Cleaned_Resume"])

        job_vector = vectorizer.transform([clean_job])

        scores = cosine_similarity(job_vector, resume_vectors)[0]

        df["Score"] = scores * 100

        df["Missing_Skills"] = df["Skills"].apply(
        lambda x: get_missing_skills(x, clean_job)
        )

        ranked = df.sort_values(by="Score", ascending=False)

        top5 = ranked.head(5)

        st.subheader("🏆 Top Matching Candidates")

        st.dataframe(
        top5[["Category","Skills","Missing_Skills","Score"]]
        .style.format({"Score":"{:.2f}"})
        )
        # -----------------------------
        # Visualization
        # -----------------------------

        st.subheader("📊 Top Candidate Scores")

        fig, ax = plt.subplots()

        ax.bar(top5["Category"], top5["Score"])

        plt.xticks(rotation=45)

        st.pyplot(fig)

# -----------------------------
# PAGE 2 — EXAMPLE JOBS
# -----------------------------

elif page == "Example Job Descriptions":

    st.title("💡 Example Job Descriptions")

    examples = [
    "Looking for Data Scientist with Python, Machine Learning, SQL, Pandas",
    "Software engineer required with Java, C++, Data Structures",
    "Data Analyst with Excel, SQL, Tableau, Data Visualization",
    "AI Engineer with Deep Learning, Python, TensorFlow",
    "Business analyst with management and communication skills"
    ]

    for e in examples:
        st.write("•", e)

# -----------------------------
# PAGE 3 — DATASET DASHBOARD
# -----------------------------

elif page == "Candidate Dashboard":

    st.title("📊 Resume Dataset Dashboard")

    col1, col2 = st.columns(2)

    col1.metric("Total Resumes", len(df))
    col2.metric("Job Categories", df["Category"].nunique())

    st.divider()

    st.subheader("Resume Category Distribution")

    fig, ax = plt.subplots()

    sns.countplot(y="Category", data=df, ax=ax)

    st.pyplot(fig)

# -----------------------------
# PAGE 4 — ABOUT PROJECT
# -----------------------------

elif page == "About Project":

    st.title("ℹ️ About This Project")

    st.markdown("""

### AI Resume Screening System

This project builds a **machine learning system that automatically screens and ranks resumes based on job descriptions**.

### Machine Learning Workflow

1. Resume Text Cleaning  
2. NLP Preprocessing  
3. Skill Extraction  
4. TF-IDF Vectorization  
5. Cosine Similarity Matching  
6. Resume Ranking  
7. Skill Gap Analysis  

### Tools Used

- Python  
- Scikit-learn  
- NLTK  
- Streamlit  

### Goal

To help recruiters **automatically shortlist candidates based on job requirements**.

---

Developed for **Future Interns – Machine Learning Internship Task**
""")
