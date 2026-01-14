import pandas as pd
import joblib

df = pd.read_csv("input.csv")

model = joblib.load("username_cleaner_ai.pkl")

df.columns = df.columns.str.strip().str.lower()

if "username" in df.columns:
    df["username"] = (
        df["username"]
        .astype(str)
        .apply(lambda x: model.predict([x])[0])
    )


df.to_csv("ai_cleaned_output.csv", index=False)

print("🧠 AI-cleaned file generated: ai_cleaned_output.csv")



