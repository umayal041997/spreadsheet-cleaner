from flask import Flask, render_template, request, send_file
import pandas as pd
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def clean_csv(input_path, output_path):
    df = pd.read_csv(input_path)

    df.columns = df.columns.str.strip().str.lower()
    columns_to_drop = ["upline ref id", "ref no", "bank details", "receipt", "notes"]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    if "username" in df.columns:
        df["username"] = (
            df["username"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"(\s+\d+)+$", "", regex=True)
        )

    if "transaction date" in df.columns:
        position = df.columns.get_loc("transaction date")
        df[["wallet", "date"]] = df["transaction date"].str.split(r"\s{5,}", n=1, expand=True)
        df["date"] = df["date"].str.split(r"\s{2,}", n=1).str[0]
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
        df["credit"] = (
            df["credit"]
            .astype(str)
            .str.replace('"', '', regex=False)
            .str.replace(',', '', regex=False)
            .replace('', '0')
            .astype(float)
        )

    df = df.dropna(axis=1, how="all")
    df.to_csv(output_path, index=False)

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        output_path = os.path.join(OUTPUT_FOLDER, "cleaned_" + file.filename)

        file.save(input_path)
        clean_csv(input_path, output_path)

        return send_file(output_path, as_attachment=True)

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
