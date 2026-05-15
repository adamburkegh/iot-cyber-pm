import pm4py as pm 
import pandas as pd
import ebi
import os
import re
try:
    import zeek_conversion as zc
except ImportError:
    from . import zeek_conversion as zc
from pm4py.objects.log.obj import EventLog
from pm4py.algo.filtering.log.variants import variants_filter
from datetime import datetime as dt

# folders for benign and malicious Zeek logs within this project
BENIGN_FOLDER = os.path.join("dataset", "benign")
MALICIOUS_FOLDER = os.path.join("dataset", "malicious")

def export_pm4py_to_lpn(pm_net, pm_initial_marking, output_folder):
    # 1. Collect places and index them by NAME
    places = list(pm_net.places)
    place_index = {p.name: i for i, p in enumerate(places)}
    # 2. Collect transitions in stable order
    transitions = list(pm_net.transitions)
    # 3. Sanitize labels for EBI
    def sanitize(label):
        if label is None:
            return None
        return re.sub(r"[^A-Za-z0-9_]", "_", label)
    lpn_path = os.path.join(output_folder, f"labelled petri_net.lpn")
    with open(lpn_path, "w", encoding="utf-8") as f:
        # Header
        f.write("labelled Petri net\n")
        # Number of places
        f.write(f"{len(places)}\n")
        # Initial marking
        for p in places:
            tokens = pm_initial_marking[p] if p in pm_initial_marking else 0
            f.write(f"{tokens}\n")
        # Number of transitions
        f.write(f"{len(transitions)}\n")
        # Write each transition
        for t in transitions:
            # Label or silent
            safe_label = sanitize(t.label)
            if safe_label is None:
                f.write("silent\n")
            else:
                f.write(f"label {safe_label}\n")
            # Input places (compare by NAME)
            in_places = [
                arc.source.name
                for arc in pm_net.arcs
                if arc.target.name == t.name and arc.source.name in place_index
            ]
            f.write(f"{len(in_places)}\n")
            for p_name in in_places:
                f.write(f"{place_index[p_name]}\n")

            # Output places (compare by NAME)
            out_places = [
                arc.target.name
                for arc in pm_net.arcs
                if arc.source.name == t.name and arc.target.name in place_index
            ]
            f.write(f"{len(out_places)}\n")
            for p_name in out_places:
                f.write(f"{place_index[p_name]}\n")
        # Final newline required by EBI parser
        f.write("\n")
    return lpn_path
def process_model_metrics(trace, model_obj, model_path, benign_log):
    trace_event_log = EventLog([trace])

    tp = get_trace_probability(trace_event_log, model_obj)

    emd = ebi.conformance_earth_movers(trace_event_log, benign_log)
    cs  = ebi.conformance_chi_square(trace_event_log, model_obj)
    er  = ebi.conformance_entropic_relevance(trace_event_log, model_obj)

    match= re.search(r"[-+]?\d*\.\d+|\d+", str(er))
    er_value = float(match.group()) if match else None    
    metrics = {
        "activity_labels": get_activity_labels(trace),
        "trace_probability": tp,
        "earth_movers_distance": emd[0],
        "chi_square": float(cs[0]),
        "entropic_relevance": er_value,
        }
    return metrics

def write_metrics(test_event_log_obj, benign_event_log_obj, slpn_obj, slpn_path, output_folder):
    print("Grouping by variants.....")
    variants = variants_filter.get_variants(test_event_log_obj)
    
    variant_metrics = {}
    v_count = 0
    for variant, traces in variants.items():
        current_trace = traces[0]
        metrics = process_model_metrics(current_trace, slpn_obj, slpn_path, benign_event_log_obj)
        variant_metrics[variant] = metrics
        v_count += 1
        if v_count % 10 == 0:
            print(f"Calculated {v_count} unique variants...")

    print("Creating metrics DataFrame...")
    metrics_list = []
    for variant, traces in variants.items():
        metrics = variant_metrics[variant]
        for trace in traces:
            row = metrics.copy()
            row["uid"] = trace.attributes["concept:name"]
            metrics_list.append(row)
    metrics_df = pd.DataFrame(metrics_list)
    print(f"Metrics DataFrame with {len(metrics_df)} rows")
    return metrics_df
    
def get_trace_probability(trace_event_log, model_obj):
    result = ebi.probability_log(model_obj, trace_event_log)
    return float(result[0])
        
def get_activity_labels(trace):
    labels = [event["concept:name"] for event in trace]
    #print("Activity labels: ", labels)
    return labels

def get_slpn( event_log_obj, output_folder):
    #pm4py petri net + lpn
    net, initial_marking, final_marking  = pm.discover_petri_net_inductive(event_log_obj)
    #image prep
    image_path = os.path.join(output_folder, f"petri_net.png")
    # try to save visualization, if Graphviz is not installed, skip image generation
    try:
        pm.save_vis_petri_net(net, initial_marking, final_marking, image_path)
    except Exception as e:
        print(f"Could not save Petri net visualization (Graphviz may be missing): {e}")
    
    #lpn to slpn
    lpn_path = export_pm4py_to_lpn(net, initial_marking, output_folder)
    with open(lpn_path, "r") as file:
        lpn_model_string = file.read()

    slpn_obj=ebi.discover_alignments(event_log_obj, lpn_model_string)

    slpn_path = os.path.join(output_folder, f"stochastic_petri_net.slpn")
    with open(slpn_path, "w") as file:
        file.write(slpn_obj)
    
    print(f"Model discovery complete. SLPN saved to: {slpn_path}")
    return slpn_obj, slpn_path

def get_log_object(df):
    dtypes = {
        "uid": str,
        "datetime": str,
        "event_order": int,
        "history_activity": str,
        "label": str
    }
    important_cols = ["uid", "datetime", "event_order", "history_activity", "label"]
    df = df[important_cols].copy()
    df = df.astype(dtypes, errors="ignore")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values(["uid", "datetime", "event_order"], ascending=True, inplace=True)
    df = pm.format_dataframe(df, case_id="uid", activity_key="history_activity", timestamp_key="datetime")
    log = pm.convert_to_event_log(df, case_id_key="uid", activity_key="history_activity", timestamp_key="datetime")
    print(f"Converted DataFrame to EventLog with {len(log)} traces and {sum(len(trace) for trace in log)} events.")
    return log

def create_output_folder(parent_folder):
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    target_folder = os.path.join(parent_folder, timestamp)
    os.makedirs(target_folder, exist_ok=True)
    return target_folder

def run_pipeline(output_parent_folder):
    # scans for log.labeled files in benign folder
    print("Extracting benign logs...")
    benign_folder = BENIGN_FOLDER
    benign_dataframe = zc.extract_zeek_logs(benign_folder)
    benign_event_log_obj = get_log_object(benign_dataframe)

    output_folder = create_output_folder(output_parent_folder)

    print("Discovering model from benign logs...")
    slpn_obj, slpn_path = get_slpn( benign_event_log_obj, output_folder)
    # scans for log.labeled files in malicious folder
    print("Extracting malicious logs...")
    mal_folder = MALICIOUS_FOLDER
    mal_dataframe = zc.extract_zeek_logs(mal_folder)
    mal_event_log_obj = get_log_object(mal_dataframe)

    print("Calculating metrics for malicious logs...")
    metrics_dataframe = write_metrics(mal_event_log_obj, benign_event_log_obj, slpn_obj, slpn_path, output_folder)
    print("Pipeline execution complete.")
    print(f"\nPM Metrics shape: {metrics_dataframe.shape}")
    print(metrics_dataframe.head())

    return metrics_dataframe


