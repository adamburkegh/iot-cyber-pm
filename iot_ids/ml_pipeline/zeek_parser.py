import pandas as pd

NUMERIC_COLUMNS = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "id.orig_p",
    "id.resp_p",
]


def zeek_to_df(input_file: str) -> pd.DataFrame:
    field_names = None
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#fields"):
                parts = line.rstrip("\n\r").split("\t")
                if len(parts) > 1:
                    raw = parts[1:]
                    field_names = []
                    for i, name in enumerate(raw):
                        name = name.strip()
                        if name:
                            if i == len(raw) - 1 and "   " in name:
                                field_names.extend(
                                    s.strip() for s in name.split("   ") if s.strip()
                                )
                            else:
                                field_names.append(name)
                continue
            if line.startswith("#types"):
                continue
            if not line.startswith("#") and line.strip():
                break
    if field_names is None:
        raise ValueError("Could not find #fields line in the file")
    num_cols = len(field_names)
    data_rows = []
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n\r").split("\t")
            if fields:
                last = fields[-1].strip()
                if "   " in last:
                    fields = fields[:-1] + [
                        s.strip() for s in last.split("   ") if s.strip()
                    ]
                else:
                    n_pad = num_cols - (len(fields) - 1)
                    if n_pad > 1:
                        fields = fields[:-1] + [last] + [""] * (n_pad - 1)
                    else:
                        fields = fields[:-1] + [last]
            if len(fields) < num_cols:
                fields.extend([""] * (num_cols - len(fields)))
            elif len(fields) > num_cols:
                fields = fields[:num_cols]
            data_rows.append(fields)
    df = pd.DataFrame(data_rows, columns=field_names)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if col in NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(
                df[col].replace("-", "").replace("(empty)", ""), errors="coerce"
            )
        else:
            df[col] = df[col].replace("(empty)", "")
    return df
