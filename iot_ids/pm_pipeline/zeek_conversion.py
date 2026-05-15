import os
import pandas as pd

def parse_zeek_log(file_path):
    columns = []
    data = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # Detect column names
            if line.startswith("#fields"):
                columns = line.split()[1:]  # skip "#fields"
                num_cols = len(columns)
                continue

            # Skip other metadata lines
            if line.startswith("#"):
                continue

            # Parse data line
            fields = line.split("\t")

            # Pad missing fields
            if len(fields) < num_cols:
                fields += ["-"] * (num_cols - len(fields))

            # Trim extra fields
            if len(fields) > num_cols:
                fields = fields[:num_cols]

            data.append(fields)
    return columns, data

def clean_tunnel(df):
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    noisy_cols = [
        c for c in df.columns
        if (
            "tunnel_parent" in c
            or "detailed_label" in c
            or c == "label"
        )
    ]
    if noisy_cols:
        for col in noisy_cols:
            if df[col].notna().any() and (df[col] != "").any() and (df[col] != "-").any():
                df["label"] = df[col]
                break
        else:
            df["label"] = ""

        for col in noisy_cols:
            if col != "label":
                df.drop(columns=col, inplace=True, errors="ignore")
    else:
        df["label"] = ""
    return df

def add_metadata(df, source_file):
    if "ts" in df.columns:
        ts_numeric = pd.to_numeric(df["ts"], errors="coerce")
        df["datetime"] = pd.to_datetime(ts_numeric, unit="s", utc=True)
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df["source_file"] = os.path.basename(source_file)

    df["event_order"] = 1
    if "history" in df.columns:
        df["history_activity"] = df["history"].str[0]
    return df

def proto_filter(df):
    if "proto" in df.columns: 
        df = df[df["proto"] == "tcp"] 
        print(f" Rows after TCP filter: {len(df)}") 
        return df
    else: 
        print(f" Proto column not found")
        return df

def carot_filter(str):
    hist_chars = []
    i = 0
    while i<len(str):
        if str[i] == "^":
            temp_str = str[i] + str[i+1]
            hist_chars.append(temp_str)
            i+=2
        else:
            hist_chars.append(str[i])
            i+=1
    return hist_chars

def history_filter(df):
    if "history" not in df.columns:
        print(f" History column not found")
        return df
    
    df = df.copy()
    df["history_list"] = df["history"].apply(carot_filter)
    exploded_df = df.explode("history_list").reset_index(drop=True)
    exploded_df["event_order"] = exploded_df.groupby("uid").cumcount() + 1
    exploded_df["history_activity"] = exploded_df["history_list"]
    exploded_df.drop(columns=["history_list"], inplace=True)
    print(f" Rows after exploding history: {len(exploded_df)}")
    return exploded_df

def read_zeek_log(file_path):
    columns, data = parse_zeek_log(file_path)
    df = pd.DataFrame(data, columns=columns)
    df = clean_tunnel(df)
    df = add_metadata(df, file_path)
    return df

def extract_zeek_logs(data_dir):
    zeek_files = [f for f in os.listdir(data_dir) if f.endswith(".log.labeled")]
    print("Zeek files detected:", zeek_files)
    dfs = []
    for fname in zeek_files:
        print(f"Reading {fname}...")
        path = os.path.join(data_dir, fname)
        df = read_zeek_log(path)
        print(f"  Loaded rows: {len(df)}")
        df.replace("-", "", inplace=True)
        df = proto_filter(df)
        df = history_filter(df)
        dfs.append(df)

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print("\nCombined head:")
        print(combined_df.head())

        print("\nColumns:")
        print(combined_df.columns.tolist())
        return combined_df
    else:
        print("No Zeek files found.")
        return pd.DataFrame()


