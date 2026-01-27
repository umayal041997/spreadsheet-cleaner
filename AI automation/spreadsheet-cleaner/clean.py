# SAMPLE STEP 1

#Rules:
# Trim spaces
# Convert to lowercase
# Remove numbers


# import pandas as pd

# df = pd.read_csv("input.csv")

# df["username"] = (
#     df["username"]
#     .astype(str)
#     .str.strip()
#     .str.lower()
#     .str.replace(r"\d+", "", regex=True)
# )

# df.to_csv("cleaned.csv", index=False)

# print("Cleanup done. Check cleaned.csv")



import pandas as pd
import json
import joblib

from rapidfuzz import process, fuzz
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

input_file = "HABAAAE_test.csv"
output_file = "output_HABAAAE.csv"



STANDARD_COLUMNS = {
    "username": ["user", "user name", "login", "member", "client", "account holder"],
    "transaction date": ["transaction date", "txn date", "date time", "created at", "time"],
    "credit": ["credit", "amount", "value", "cr", "amt"],
    "account name": ["account", "bank account", "wallet account"]
}


def smart_rename_columns(df):
    new_columns = {}

    for col in df.columns:
        best_match = None
        best_score = 0
        original = col
        col_lower = col.strip().lower()


        if col_lower in STANDARD_COLUMNS.keys():
            new_columns[original] = col_lower
            continue
        if col_lower == "debit":
            new_columns[original] = col_lower
            continue

        for standard, variations in STANDARD_COLUMNS.items():
            match, score, _ = process.extractOne(col_lower, variations, scorer=fuzz.partial_ratio)

            if score > best_score and score > 70:
                best_match = standard
                best_score = score

        # After checking all standards for this column, decide on the final name
        if best_match:
            new_columns[original] = best_match
        else:
            new_columns[original] = col_lower

    # Apply all renames together
    df.rename(columns=new_columns, inplace=True)
    return df



df = pd.read_csv(input_file)
if "Credit" in df.columns:
    raw_credit = df["Credit"].copy()
df = smart_rename_columns(df)

# Deduplicate columns that may have been mapped to same standard name
df = df.loc[:, ~df.columns.duplicated()]

print("Rows:", len(df))
print("Columns:", list(df.columns))
print("\nSample Data:")
print(df.head())
print("\n✅ File loaded successfully, starting cleaning...")


# # ---- AI CLEANING RULES ----
# df.columns = df.columns.str.strip().str.lower()

columns_to_drop = [ "upline ref id", "ref no", "bank details", "receipt","notes" ]

df = df.drop(columns=columns_to_drop, errors="ignore")



# clean username
if "username" in df.columns:
    df["username"] = (
        df["username"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"(\s+\d+)+$", "", regex=True)
    )

# if  "transaction date" in df.columns:

#         position = df.columns.get_loc("transaction date")

#         df[['wallet', "date"]] = df["transaction date"].str.split(
#             r"\s{5,}", n=1, expand=True
#         )

#         df[['date']] = df['date'].str.split(r"\s{2,}", n=1).str[0]

#         df.drop(columns=["transaction date"], inplace=True)

#         cols = list(df.columns)
#         cols.remove("wallet")
#         cols.remove("date")

#         cols.insert(position, "wallet")
#         cols.insert(position + 1, "date")

#         df = df[cols]


if "transaction date" in df.columns:

    # Use the first occurrence index even if there are duplicate column names
    position = list(df.columns).index("transaction date")

    # get only the first one
    txn_col = df["transaction date"]
    if isinstance(txn_col, pd.DataFrame):
        txn_col = txn_col.iloc[:, 0]

    df[["wallet", "date"]] = txn_col.astype(str).str.split(
        r"\s{5,}", n=1, expand=True
    )

    df["date"] = df["date"].astype(str).str.split(r"\s{2,}", n=1).str[0]

    df.drop(columns=["transaction date"], inplace=True)

    cols = list(df.columns)
    cols.remove("wallet")
    cols.remove("date")

    cols.insert(position, "wallet")
    cols.insert(position + 1, "date")

    df = df[cols]

if "account name" in df.columns:

      position = df.columns.get_loc("account name")

      df[["acc_name", "provider"]] = df["account name"].str.split("/", n=1, expand=True)
      df[["provider_name", "acc_no"]] = df["provider"].str.split(r"\s{2,}", n=1, expand=True)

      df.drop(columns=["account name"], inplace=True)

      cols = list(df.columns)
      cols.remove("acc_name")
      cols.remove("provider_name")
      cols.remove("provider")
      cols.remove("acc_no")

      cols.insert(position, "acc_name")
      cols.insert(position + 1, "provider_name")
      cols.insert(position + 2, "acc_no")

      df = df[cols]

if "credit" in df.columns:
    credit_col = raw_credit
    if isinstance(credit_col, pd.DataFrame):
        credit_col = credit_col.iloc[:, 0]

    extracted = credit_col.astype(str).str.findall(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?")
    cleaned_credit = extracted.apply(lambda x: x[0] if len(x) else "0")

    df["credit"] = (
        cleaned_credit
        .str.replace(",", "", regex=False)
        .astype(float)
    )

df = df.dropna(axis=1, how="all")

df.to_csv(output_file, index=False)

print(f"🤖 AI Cleaning completed → {output_file}")


def detect_anomalies(df):
    anomalies = pd.DataFrame()

    required_columns = ["username", "credit", "date"]
    for col in required_columns:
        if col in df.columns:
            series = df[col]
            # If duplicate column names produced a DataFrame, take the first column
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            missing = df[series.isna() | (series.astype(str).str.strip() == "")]
            anomalies = pd.concat([anomalies, missing])

    duplicates = df[df.duplicated(keep=False)]
    anomalies = pd.concat([anomalies, duplicates])

    if "credit" in df.columns:
        mean = df['credit'].mean()
        std = df['credit'].std()
        outliers = df[(df['credit'] > mean + 3*std) | (df['credit'] < mean - 3*std)]
        anomalies = pd.concat([anomalies, outliers])


    anomalies = anomalies.drop_duplicates()


    return anomalies






PATTERN_DB = "pattern_memory.json"

def load_patterns():
    if Path(PATTERN_DB).exists():
        with open(PATTERN_DB,"r") as f:
            return json.load(f)

    return {
        "files_seen": 0,
        "columns": {},
        "credit_stats": {"min": [], "max":[], "mean":[]},
        "row_counts": []
    }

def save_patterns(patterns):
    with open(PATTERN_DB, "w") as f:
        json.dump(patterns,f,indent=4)

def learn_patterns(df):
    patterns = load_patterns()
    patterns["files_seen"] +=1

    #counts how many times each column name appears and stores the frequency.
    for col in df.columns:
        patterns["columns"][col] = patterns["columns"].get(col,0) + 1

    #understand credit behaviour
    if "credit" in df.columns:
        patterns["credit_stats"]["min"].append(float(df["credit"].min()))
        patterns["credit_stats"]["max"].append(float(df["credit"].max()))
        patterns["credit_stats"]["mean"].append(float(df["credit"].mean()))

    patterns["row_counts"].append(len(df))

    save_patterns(patterns)
    return patterns


def detect_pattern_anomalies(df, patterns):
    warnings = []

    #column drift
    known_cols = set(patterns["columns"].keys())
    new_cols = set(df.columns)
    unseen = known_cols - new_cols
    if unseen:
        warnings.append(f"New unseen columns: {list(unseen)}")

    #credit drift
    if "credit" in df.columns and patterns["credit_stats"]["mean"]:
        hist_avg = sum(patterns["credit_stats"]["mean"])/len(patterns["credit_stats"]["mean"])
        curr_avg = df["credit"].mean()
        if curr_avg > hist_avg *3:
            warnings.append("Average credit much higher than historical pattern")

    # Size drift
    if patterns["row_counts"]:
        avg_rows = sum(patterns["row_counts"]) / len(patterns["row_counts"])
        if len(df) > avg_rows * 3:
            warnings.append("Row count unusually large vs past files")

    return warnings


# --- RUN ANOMALY DETECTION ---
anomalies_df = detect_anomalies(df)

anomalies_df.to_csv("anomalies_output.csv", index=False)


patterns = learn_patterns(df)
pattern_warnings = detect_pattern_anomalies(df, patterns)

if pattern_warnings:
    with open("pattern_warnings.txt", "w") as f:
        for w in pattern_warnings:
            f.write(w + "\n")
    print("⚠ Pattern warnings saved to pattern_warnings.txt")
else:
    print("✅ File matches historical patterns")



# STEP 4.3 — Create training dataset from your cleaned CSV
def build_features(df):
    X = pd.DataFrame()

    X["credit"] = df["credit"] if "credit" in df.columns else 0
    X["missing_username"] = df["username"].isna().astype(int) if "username" in df.columns else 0
    X["missing_date"] = df["date"].isna().astype(int) if "date" in df.columns else 0
    X["is_duplicate"] = df.duplicated().astype(int)

    return X


# STEP 4.4 — Auto-label training data (weak supervision)

def auto_label(df):
    labels = []

    # Check if credit column exists and calculate threshold
    has_credit = "credit" in df.columns
    credit_threshold = df["credit"].mean() * 3 if has_credit else None

    for _, row in df.iterrows():
        if has_credit and row.get("credit", 0) > credit_threshold:
            labels.append("high_risk")
        elif pd.isna(row.get("username")) or pd.isna(row.get("date")):
            labels.append("suspicious")
        else:
            labels.append("normal")


    return labels


# Train the model

def train_model(df):
    X = build_features(df)
    y = auto_label(df)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    joblib.dump(model, "risk_model.pkl")
    joblib.dump(encoder,"label_encoder.pkl")


    print("🤖 Risk classification model trained and saved")


train_model(df)


#STEP 4.6 — Predict risk for new file

def predict_risk(df):
    model = joblib.load("risk_model.pkl")
    encoder = joblib.load("label_encoder.pkl")

    X = build_features(df)
    preds = model.predict(X)
    labels = encoder.inverse_transform(preds)

    df["risk_level"] = labels
    df.to_csv("classified_output.csv", index=False)

    print("🚨 Risk classification saved to classified_output.csv")


predict_risk(df)
