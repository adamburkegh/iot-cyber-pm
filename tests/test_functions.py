import pandas as pd
import ebi
import numpy as np
from pm4py.objects.log.obj import EventLog
import fractions as Fraction
from scipy.stats import ttest_ind as t_test
import matplotlib.pyplot as plt
#from pipeline import get_trace_probability, process_model_metrics


#event log obj is log2 obj
def intersection_test(event_log_obj, benign_slpn_path,log1_path, log2_path):   
    log1 = get_log(log1_path)
    log2 = get_log(log2_path)

    intersection = log1.intersection(log2)          
    print("\nNumber of overlapping traces: ", len(intersection))
    print("Overlapping traces: ", intersection)

    for trace in event_log_obj:
        activity_labels = [event["concept:name"] for event in trace]
        if tuple(activity_labels) in intersection:
            print("\nFound overlapping trace: ", activity_labels)
            get_trace_probability(trace, benign_slpn_path)

def get_sample_log(event_log, sample_size):
    if len(event_log) > sample_size:
        sample_list = event_log[:sample_size]
        sample_log = EventLog(sample_list)
        return sample_log
    else:
        return event_log

def statistical_test(m_type, benign_log_obj, test_log_obj, benign_slpn_obj, benign_slpn_path):
    benign_metrics= []
    print("Getting benign trace metrics")
    for trace in benign_log_obj:
        match m_type:
            case "tp":
                tp = get_trace_probability(trace, benign_slpn_path)
                metric = float(Fraction(tp))
            case "emd":
                emd = ebi.conformance_earth_movers_stochastic_conformance(EventLog([trace]), benign_log_obj)
                metric = emd[0]  
            case "cs":
                metric = ebi.conformance_chi_square_stochastic_conformance(EventLog([trace]), benign_slpn_obj)
            case "er":
                er = ebi.conformance_entropic_relevance(EventLog([trace]), benign_slpn_obj)
                metric = float(er)
            case "hsc":
                hsc = ebi.conformance_hellinger_stochastic_conformance(EventLog([trace]), benign_slpn_obj)
                metric = float(hsc[0])
            case "js":
                js = ebi.conformance_jensen_shannon(EventLog([trace]), benign_slpn_obj)
                metric = float(js)
        benign_metrics.append(metric)
    
    print("BENIGN FILE COMPLETE")

    benign_from_mal=[]
    mal_metrics = []
    print ("Getting malicious trace metrics")
    sample_log = get_sample_log(test_log_obj, 1000)  # Adjust sample size as needed
    for trace in sample_log:
        match m_type:
            case "tp":
                tp = get_trace_probability(trace, benign_slpn_path)
                metric = float(Fraction(tp))
            case "emd":
                emd = ebi.conformance_earth_movers_stochastic_conformance(EventLog([trace]), benign_log_obj)
                metric = emd[0]  
            case "cs":
                metric = ebi.conformance_chi_square_stochastic_conformance(EventLog([trace]), benign_slpn_obj)
            case "er":
                er = ebi.conformance_entropic_relevance(EventLog([trace]), benign_slpn_obj)
                metric = float(er)
            case "hsc":
                hsc = ebi.conformance_hellinger_stochastic_conformance(EventLog([trace]), benign_slpn_obj)
                metric = float(hsc[0])
            case "js":
                js = ebi.conformance_jensen_shannon(EventLog([trace]), benign_slpn_obj)
                metric = float(js)

        if trace.attributes["mal activity"] == "-   Benign   -":
            benign_from_mal.append(metric)
        else:
            mal_metrics.append(metric)
    print("MALICIOUS FILE COMPLETE")

    all_benign = benign_metrics+ benign_from_mal

    t_stat, p_value = t_test(all_benign, mal_metrics)
    print("Malicious traces: ", len(mal_metrics))
    print("Benign traces: ", len(all_benign))

    print(" \n T-STATISTIC:", t_stat, "\n P-VALUE:", p_value)

# Run stats
    describe("Benign ", all_benign)
    describe("Malicious ", mal_metrics)


    plt.figure(figsize=(8, 6))

    plt.boxplot(
        [all_benign, mal_metrics],
        tick_labels=["Benign", "Malicious"],
        patch_artist=True
    )

    plt.title(f"Distribution of {m_type.upper()}", fontsize=14)
    plt.ylabel("Value")
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.show()

def describe(name, values):
    values = np.array(values)
    print(f"\n===== {name} =====")
    print(f"Count: {len(values)}")
    print(f"Min: {values.min():.6f}")
    print(f"Max: {values.max():.6f}")
    print(f"Mean: {values.mean():.6f}")
    print(f"Median: {np.median(values):.6f}")
    print(f"Std Dev: {values.std():.6f}")

    print("\nQuantiles:")
    for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        print(f"  {int(q*100)}th percentile: {np.quantile(values, q):.6f}")

def get_log(path):
    
    dataframe = pd.read_csv(path)
    dataframe = dataframe.sort_values(by=["uid", "datetime"])
    traces = dataframe.groupby("uid")["history_activity"].apply(tuple)
    return set(traces)


def metrics_result_test(benign_event_log_obj, test_event_log_obj, benign_slpn_obj, benign_slpn_path):
    for i in range(1):
        trace = benign_event_log_obj[i]
        print("\n BENIGN RESULT INDEX: ", i)
        process_model_metrics(trace, benign_slpn_obj, benign_slpn_path)

    for i in range(1):
        trace = test_event_log_obj[i]
        print("TEST RESULT INDEX: ", i)
        process_model_metrics(trace, benign_slpn_obj, benign_slpn_path)



