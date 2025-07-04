import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

from sklearn.metrics import classification_report, confusion_matrix
import joblib

from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load your manually created CSV
df = pd.read_excel("/home/malladi/projects/resume_project/for_class.xlsx")
  # has resume_str, role

# Encode roles
le = LabelEncoder()
df["label"] = le.fit_transform(df["role"])




model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim embeddings
X = model.encode(df["resume_str"].tolist(), show_progress_bar=True)
y = df["label"].values



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



joblib.dump(clf, "resume_classifier_xgb.pkl")
joblib.dump(le, "label_encoder.pkl")
model.save("embedding_model/")


def classify_resume(resume_text):
    embed = model.encode([resume_text])
    label = clf.predict(embed)[0]
    return le.inverse_transform([label])[0]

# Example
new_resume = "Experience in building scalable systems using Node.js, React, and cloud deployment on AWS."
print("Predicted Role:", classify_resume(new_resume))
