import mysql.connector
import PyPDF2
import pandas as pd
import os
import uuid
from sentence_transformers import SentenceTransformer
import joblib
clf = joblib.load("xgb_resume_classifier.pkl")
le = joblib.load("label_encoder.pkl")


mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="testdb"
)

cur=mydb.cursor()





def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        return f"Error reading {pdf_path}: {e}"

# Directory containing PDF resumes
resume_dir = r"/home/malladi/projects/resume_project/for_the_db"  # Put resume pdf's  here
JD_dir=r"/home/malladi/projects/resume_project/JD" #Put JD pdf's here
# Collect extracted resume data
data = []
job=[]

# Loop through all PDFs in the folder
if os.path.exists(resume_dir):
    for filename in os.listdir(resume_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(resume_dir, filename)
            resume_text = extract_text_from_pdf(pdf_path)
            resume_id = str(uuid.uuid4())  # Unique identifier
            data.append({
                "ID": resume_id,
                "resume_str": resume_text,
                "Category": ""
            })

if os.path.exists(JD_dir):
    for filename in os.listdir(JD_dir):
        if filename.endswith(".pdf"):
            
            pdf_path = os.path.join(JD_dir, filename)
            JD_text = extract_text_from_pdf(pdf_path)
            JD_id=str(uuid.uuid4())
            job.append({
                "ID": JD_id,
                "JD_str": JD_text,
                "Category": ""
            }
            )
            

# Convert to DataFrame
df_resumes = pd.DataFrame(data)
df_JD=pd.DataFrame(job)






def predict_role(resume_str):
    embedder = SentenceTransformer('sentence-t5-large')
    embed = embedder.encode([resume_str], show_progress_bar=False)
    pred_label = clf.predict(embed)[0]  # Get single prediction
    pred_role = le.inverse_transform([pred_label])[0]
    return pred_role






def insert_values(mydb,cur,df,table_name):
    
    if table_name=="resume":
        for i,row in df.iterrows():
            id=1020+i
            res_str=row["resume_str"]
            rescat=predict_role(res_str)
            sql="insert into resume(ID,resume_str,role) VALUES (%s,%s,%s)"
            val=(id,res_str,rescat)
            cur.execute(sql, val)
            

        
    else:
        for i,row in df.iterrows():
            id=100+i
            JD_str=row["JD_str"]
            JDcat=predict_role(JD_str)
            print(JDcat,JD_str[:100])
            sql="insert into JD(JD_id,JD_str,role) VALUES (%s,%s,%s)"
            val=(id,JD_str,JDcat)
            cur.execute(sql, val)
            
    mydb.commit()
    
    
    
    print(cur.rowcount,"rows inserted")

insert_values(mydb,cur,df_JD,"JD")

