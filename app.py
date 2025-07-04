import spacy
from spacy.pipeline import EntityRuler
from spacy.lang.en import English
from spacy.tokens import Doc
import gensim
from gensim import corpora
from spacy import displacy
import pandas as pd
import numpy as np
import jsonlines
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download(['stopwords','wordnet'])
#warning
import warnings
import requests
warnings.filterwarnings('ignore')
import os
import uuid
import pandas as pd
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="testdb"
)

cur=mydb.cursor()

import joblib
from sentence_transformers import SentenceTransformer

# Load saved models
clf = joblib.load("xgb_resume_classifier.pkl")
le = joblib.load("label_encoder.pkl")

# Load the same embedding model you used for training
model = SentenceTransformer('all-mpnet-base-v2')


resume=[]
job_desc=[]

def getvals(lst,tablename):
    cur.execute(f"select * from {tablename}")
    res=cur.fetchall()

    if tablename=="resume":
        id=1000
    else:
        id=100
    
    for i in res:
        id=id+1
        txt=i[1]
        Cat_res=i[2]
        lst.append({
                "ID": id,
                f"{tablename}_str": txt,
                "Category": Cat_res 
            })

getvals(resume,"resume")
getvals(job_desc,"JD")


df_resume=pd.DataFrame(resume)
df_JD=pd.DataFrame(job_desc)


def load_spacy_model():
    nlp = spacy.load('en_core_web_lg')
    # Add entity ruler with overwrite enabled
    ruler = nlp.add_pipe('entity_ruler', config={"overwrite_ents": True})
    skill_pattern_path = r'/home/malladi/projects/resume_project/jz_skill_patterns.jsonl'
    nlp.get_pipe("entity_ruler").from_disk(skill_pattern_path)
    return nlp


nlp = load_spacy_model()

def get_skills(text, nlp):

    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    return skills

def unique_skills(skills):

    return list(set(skills))

def clean_resume_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def process_dataset(nlp, df):
    clean_resumes = [clean_resume_text(resume) for resume in df["resume_str"]]
    df["Clean_Resume"] = clean_resumes
    df["skills"] = df["Clean_Resume"].str.lower().apply(lambda x: get_skills(x, nlp))
    df["skills"] = df["skills"].apply(unique_skills)

    print("Dataset processing complete.")
    return df

df = process_dataset(nlp, df_resume)



def extract_experience_years(text):
    text = text.lower()

    # Pattern: "3 - 5 years", "3-5 years"
    match_range = re.search(r'(\d+)\s*[-to]+\s*(\d+)\s*(?:years|yrs)', text)
    if match_range:
        return int(match_range.group(2))  # Use upper bound

    # Pattern: "5+ years", "5+ yrs"
    match_plus = re.search(r'(\d+)\+\s*(?:years|yrs)', text)
    if match_plus:
        return int(match_plus.group(1))

    # Pattern: "minimum of 3 years", "at least 4 years"
    match_min = re.search(r'(?:minimum of|at least)\s+(\d+)\s*(?:years|yrs)', text)
    if match_min:
        return int(match_min.group(1))

    # Pattern: "3 years", "2 yrs"
    match_single = re.search(r'(\d+)\s*(?:years|yrs)', text)
    if match_single:
        return int(match_single.group(1))

    # Fallback
    return 0


def calculate_combined_match_score(input_resume,job_description,nlp,model=SentenceTransformer('all-MiniLM-L6-v2'),skill_exact_weight=0.0,skill_semantic_weight=0.7,experience_weight=0.3):
    model.eval()
    input_resume_cleaned = clean_resume_text(input_resume.lower())
    job_description_cleaned = clean_resume_text(job_description.lower())

    resume_skills = [s.lower().strip() for s in unique_skills(get_skills(input_resume_cleaned, nlp))]
    jd_skills = [s.lower().strip() for s in unique_skills(get_skills(job_description_cleaned, nlp))]

    # --- Exact Match ---
    if jd_skills:
        exact_matches = set(resume_skills).intersection(set(jd_skills))
        skill_match_exact = len(exact_matches) / len(jd_skills)
    else:
        skill_match_exact = 0

    # --- Semantic Match ---
    if resume_skills and jd_skills:
        resume_vec = model.encode(" ".join(resume_skills))
        jd_vec = model.encode(" ".join(jd_skills))
        sim = cosine_similarity([resume_vec], [jd_vec])[0][0]
        skill_match_semantic = min(sim * 1.5, 1.0)  # Boost and cap at 1.0
    else:
        skill_match_semantic = 0

    # --- Experience Match ---
    req_exp = extract_experience_years(job_description)
    res_exp = extract_experience_years(input_resume)

    if req_exp > 0:
        experience_match = min(res_exp / req_exp, 1.0)
    elif res_exp > 0:
        experience_match = 0.5  # Fallback
    else:
        experience_match = 0

    # --- Combine ---
    total = (
        skill_match_exact * skill_exact_weight +
        skill_match_semantic * skill_semantic_weight +
        experience_match * experience_weight
    ) * 100

    return round(total, 2)






jd_input = input("Select the role: ")



# Reset index to use iloc safely
resumes = df_resume[df_resume["Category"] == jd_input][["ID", "resume_str"]].reset_index(drop=True)
resclas=df_resume["resume_str"]


# def predict_role(resumes):
#     embedder = SentenceTransformer('sentence-t5-large')
#     embed = embedder.encode(resumes, show_progress_bar=False)
#     pred_labels = clf.predict(embed)
#     pred_roles = le.inverse_transform(pred_labels)
#     for r, p in zip(resumes, pred_roles):
#         print(f"Predicted Role: {p}\nResume Snippet: {r[:100]}...\n")

# # === Example usage ===
# new_resumes = df_resume["resume_str"]
# predict_role(new_resumes)


jds = df_JD[df_JD["Category"] == jd_input][["ID", "JD_str"]].reset_index(drop=True)

final_scores = []

for i in range(len(jds)):
    for j in range(len(resclas)):
        jd_text = jds.iloc[i]["JD_str"]
        res_text = resclas.iloc[j]
        score = calculate_combined_match_score(res_text, jd_text, nlp)

        res_id = resumes.iloc[j]["ID"]
        jd_id = jds.iloc[i]["ID"]

        final_scores.append((jd_id, res_id, score))

# Sort by score descending
top_10 = sorted(final_scores, key=lambda x: x[2], reverse=True)[:10]

# Print results
print("\nTop 10 Resume Scores for the job of",jd_input)
for rank, (jd_id, res_id, score) in enumerate(top_10, 1):
    print(f"{rank}. Resume ID: {res_id} — Score: {score:.4f}")
    