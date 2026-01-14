import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("training_data.csv")

X = df["dirty"]
y = df["clean"]

model = Pipeline([
    ("vectorizer", TfidfVectorizer(analyzer="char", ngram_range=(2, 4))),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X, y)

joblib.dump(model, "username_cleaner_ai.pkl")

print("🤖 AI model trained & saved successfully")