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

input_file = "HABAAAE_test.csv"
output_file = "output_HABAAAE.csv"

df = pd.read_csv(input_file)

print("Rows:", len(df))
print("Columns:", list(df.columns))
print("\nSample Data:")
print(df.head())

df.to_csv(output_file, index=False)

print("\n✅ File loaded and saved successfully")


# ---- AI CLEANING RULES ----
df.columns = df.columns.str.strip().str.lower()

columns_to_drop = [ "upline ref id", "ref no", "bank details", "receipt","notes" ]

df = df.drop(columns=columns_to_drop, errors="ignore")

# Example: clean username column if exists
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

    position = df.columns.get_loc("transaction date")

    df[["wallet", "date"]] = df["transaction date"].str.split(
        r"\s{5,}", n=1, expand=True
    )

    # FIX HERE
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
            .str.replace('"', '', regex=False)   # remove double quotes
            .str.replace(',', '', regex=False)   # remove thousand separators
            .replace('', '0')                    # empty → 0
            .astype(float)
        )


df = df.dropna(axis=1, how="all")

df.to_csv("cleaned_output.csv", index=False)

print("🤖 AI Cleaning completed → cleaned_output.csv")