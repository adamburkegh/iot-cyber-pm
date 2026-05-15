
from sklearn.ensemble import RandomForestClassifier

from iot_ids.ml_pipeline import run_ml_pipeline
from iot_ids.pm_pipeline import run_pipeline
import os

features = [
    #"trace_probability",
    #"earth_movers_distance",
    #"chi_square",
    #"entropic_relevance",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "orig_ip_bytes",
    "resp_ip_bytes",
    "missed_bytes",
    "id.orig_p",
    "id.resp_p",
]

MALICIOUS_FOLDER = os.path.join("dataset", "malicious")

def main():
    # 1. Run PM pipeline: benign -> model; malicious -> PM metrics
    pm_metrics_df = run_pipeline(output_parent_folder='var')

    # 2. Run ML pipeline on IoT-23 data, merging PM metrics on uid
    model = RandomForestClassifier(random_state=1, n_estimators=20, max_depth=5, n_jobs=-1)
    result = run_ml_pipeline(
        data_dir=MALICIOUS_FOLDER,
        model=model,
        feature_cols=features,
        strategy="lofo",
        file_col="source_file",
        feature_injectors=[
            {"metrics_df": pm_metrics_df, "merge_on": "uid"},
        ],
    )
    return result


if __name__ == "__main__":
    main()
