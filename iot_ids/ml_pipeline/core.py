import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def check_df(data, label_col, feature_cols, file_col=None):
    errors = []
    recommendations = []
    if label_col not in data.columns:
        errors.append(f"Label column '{label_col}' not found.")
    else:
        if data[label_col].isna().all():
            errors.append(
                f"Label column '{label_col}' is all NaN. "
                "For IoT-23 use tunnel_parents parsing or check data source."
            )
        elif data[label_col].isna().any():
            n = data[label_col].isna().sum()
            errors.append(f"Label column '{label_col}' has {n} missing value(s).")
    missing = [c for c in feature_cols if c not in data.columns]
    if missing:
        errors.append(f"Missing feature column(s): {', '.join(missing)}")
    if file_col is not None and file_col not in data.columns:
        errors.append(f"File column '{file_col}' not found (required for LOFO split).")
    elif file_col and file_col in data.columns:
        n_files = data[file_col].nunique()
        recommendations.append(f"Found {n_files} unique source file(s) in '{file_col}'.")
    return {"valid": len(errors) == 0, "errors": errors, "recommendations": recommendations}


def merge_pm_metrics(data, metrics_df, merge_on="uid", how="left"):
    if merge_on not in data.columns:
        raise ValueError(f"Merge key '{merge_on}' not found in data.")
    if merge_on not in metrics_df.columns:
        raise ValueError(f"Merge key '{merge_on}' not found in metrics.")
    return data.merge(metrics_df, on=merge_on, how=how)


def split_df(data, feature_cols, label_col, file_col=None, strategy="random", test_size=0.2, random_state=42, test_file=None):
    X = data[feature_cols].copy().fillna(0)
    y = data[label_col]
    split_info = {}
    if strategy == "random":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        split_info["strategy"] = "random"
        split_info["test_size"] = test_size
    else:
        if not file_col or file_col not in data.columns:
            raise ValueError("file_col is required for LOFO and must exist in data.")
        files = data[file_col].astype(str)
        if test_file is None:
            unique = sorted(files.unique().tolist())
            if len(unique) < 2:
                raise ValueError(f"LOFO requires at least 2 source files. Only 1 found: {unique[0]!r}.")
            test_file = unique[-1]
        train_mask = files != test_file
        test_mask = files == test_file
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        split_info["strategy"] = "lofo"
        split_info["test_file"] = test_file
    split_info["train_size"] = len(y_train)
    split_info["test_size"] = len(y_test)
    return X_train, X_test, y_train, y_test, split_info


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def get_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return None
    imp = model.feature_importances_
    if len(imp) != len(feature_names):
        return None
    return {feature_names[i]: float(imp[i]) for i in range(len(feature_names))}


def train_ml_model(model, X_train, y_train):
    try:
        m = clone(model)
    except Exception:
        m = model
    m.fit(X_train, y_train)
    return m


def evaluate_ml_model(model, X_test, y_test, feature_cols):
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    importance = get_feature_importance(model, feature_cols)
    return metrics, importance


def print_header():
    print("\n" + "=" * 80)
    print("    Machine Learning Model Evaluation Framework for IoT Network Traffic")
    print("=" * 80)


def print_dataset_info(log):
    if log:
        print("\n[DATASET PROCESSING]")
        print("-" * 70)
        print("\n".join(log))


def print_dataset_validation(validation):
    print("\n[STEP 1/4] VALIDATING DATASET...")
    print("-" * 70)
    if not validation["valid"]:
        print("VALIDATION FAILED")
        for err in validation["errors"]:
            print(f"  • {err}")
    else:
        print("VALIDATION PASSED")
        if validation.get("recommendations"):
            print("  Recommendations:")
            for rec in validation["recommendations"]:
                print(f"    • {rec}")


def print_split_strategy(strategy):
    if strategy == "lofo":
        print("   Split strategy: Leave-One-File-Out (LOFO)")
        print("    Each source file will be used once as the test set; all remaining files form the training set in that fold.")
    else:
        print("   Split strategy: RANDOM train/test split.")


def print_split_info(strategy, n_folds_or_files):
    print("\n[STEP 2/4] PREPROCESSING AND SPLITTING DATA...")
    print("-" * 70)
    if strategy == "random":
        print(f"  • RANDOM split: {n_folds_or_files} folds")
        print("  • Each fold uses a different random train/test partition.\n")
    else:
        print(f"  • LOFO split: {n_folds_or_files} unique source files found")
        print(f"  • Will iterate over each file as test set once\n")


def print_fold_header(fold_idx, n_folds, strategy, test_file=None, random_state=None):
    if strategy == "random":
        print(f"[RANDOM Fold {fold_idx}/{n_folds}] random_state={random_state}")
    else:
        print(f"[LOFO Fold {fold_idx}/{n_folds}] Testing on file: '{test_file}'")
    print("-" * 70)


def print_split_result(train_size, test_size, strategy, n_files):
    if strategy == "random":
        print(f"  • Split result: {train_size:,} train, {test_size:,} test samples")
    else:
        print(f"  • Training on {n_files - 1} files ({train_size:,} samples)")
        print(f"  • Testing on 1 file ({test_size:,} samples)")


def print_training_info(train_size, model_type):
    print("\n[STEP 3/4] TRAINING MODEL...")
    print("-" * 70)
    print(f"  • Model trained successfully with {train_size:,} samples")
    print(f"  • Model type: {model_type}")


def print_metrics_info(metrics):
    print("\n[STEP 4/4] EVALUATING MODEL...")
    print("-" * 70)
    for name, value in metrics.items():
        print(f"  • {name.replace('_', ' ').title()}: {value:.4f}")


def print_feature_importance(imp, indent="  ", imp_std=None):
    for name, val in sorted(imp.items(), key=lambda x: x[1], reverse=True):
        std_val = (imp_std or {}).get(name)
        if std_val is not None:
            print(f"{indent}- {name}: {val:.4f} ± {std_val:.4f}")
        else:
            print(f"{indent}- {name}: {val:.4f}")


def print_evaluation_summary(cv_metrics_mean, cv_metrics_std, strategy, n_folds):
    summary_title = "RANDOM CROSS-VALIDATION SUMMARY" if strategy == "random" else "LOFO CROSS-VALIDATION SUMMARY"
    print("\n" + "=" * 70)
    print(summary_title)
    print("=" * 70)
    print(f"  • Total folds: {n_folds}")
    print("  • Aggregate metrics (mean ± std across folds):")
    for name in sorted(cv_metrics_mean.keys()):
        print(f"    • {name.replace('_', ' ').title()}: {cv_metrics_mean[name]:.4f} ± {cv_metrics_std[name]:.4f}")
    print("=" * 70)


def run_evaluation(
    data,
    model,
    feature_cols,
    label_col="label",
    file_col="source_file",
    strategy="lofo",
    test_size=0.2,
    random_state=42,
    n_folds=None,
    verbose=True,
):
    val = check_df(data, label_col, feature_cols, file_col)
    if not val["valid"]:
        return {
            "success": False,
            "error": "Validation failed",
            "validation": val,
            "metrics": {},
            "train_size": 0,
            "test_size": 0,
            "split_strategy": strategy,
            "per_file_results": None,
            "per_fold_results": None,
        }
    per_fold_results = []
    all_metrics = {}
    if strategy == "random":
        n_folds = n_folds if n_folds is not None else 12
        if verbose:
            print_split_info(strategy, n_folds)
        fold_specs = [(None, random_state + i) for i in range(n_folds)]
    else:
        if not file_col or file_col not in data.columns:
            return {
                "success": False,
                "error": "file_col is required for LOFO split and must exist in data.",
                "validation": val,
                "metrics": {},
                "train_size": 0,
                "test_size": 0,
                "split_strategy": strategy,
                "per_file_results": None,
                "per_fold_results": None,
            }
        unique_files = sorted(data[file_col].astype(str).unique().tolist())
        if len(unique_files) < 2:
            return {
                "success": False,
                "error": (
                    "LOFO split requires at least 2 source files with valid labels. "
                    f"Only 1 file found: {unique_files[0]}. "
                    "Add more .labeled.csv files or use SplitStrategy.RANDOM."
                ),
                "validation": val,
                "metrics": {},
                "train_size": 0,
                "test_size": 0,
                "split_strategy": strategy,
                "per_file_results": None,
                "per_fold_results": None,
            }
        if verbose:
            print_split_info(strategy, len(unique_files))
        fold_specs = [(tf, None) for tf in unique_files]
    n_folds_done = len(fold_specs)
    for fold_idx, (test_file, rs) in enumerate(fold_specs, 1):
        try:
            if strategy == "random":
                X_train, X_test, y_train, y_test, info = split_df(
                    data, feature_cols, label_col, file_col,
                    strategy="random", test_size=test_size, random_state=rs,
                )
            else:
                X_train, X_test, y_train, y_test, info = split_df(
                    data, feature_cols, label_col, file_col,
                    strategy="lofo", test_file=test_file,
                )
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "validation": val,
                "metrics": {},
                "train_size": 0,
                "test_size": 0,
                "split_strategy": strategy,
                "per_file_results": per_fold_results if strategy == "lofo" else None,
                "per_fold_results": per_fold_results,
                "metrics_std": {},
            }
        train_size, test_size_fold = info["train_size"], info["test_size"]
        if verbose:
            print_fold_header(fold_idx, n_folds_done, strategy, test_file, rs)
            print_split_result(train_size, test_size_fold, strategy, n_folds_done)
        fitted = train_ml_model(model, X_train, y_train)
        if verbose:
            print_training_info(train_size, type(fitted).__name__)
        metrics, importance = evaluate_ml_model(fitted, X_test, y_test, feature_cols)
        if verbose:
            print_metrics_info(metrics)
        if importance is not None and verbose:
            print("  • Feature importance (this fold):")
            print_feature_importance(importance, indent="      ")
        rec = {"train_size": train_size, "test_size": test_size_fold, "metrics": metrics}
        if strategy == "random":
            rec["fold"] = fold_idx
        else:
            rec["test_file"] = test_file
        if importance is not None:
            rec["feature_importance"] = importance
        per_fold_results.append(rec)
        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(v)
        if verbose:
            print()
    metrics_mean = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    metrics_std = {k: float(np.std(v)) for k, v in all_metrics.items()}
    total_train = sum(r["train_size"] for r in per_fold_results)
    total_test = sum(r["test_size"] for r in per_fold_results)
    out = {
        "success": True,
        "metrics": metrics_mean,
        "metrics_std": metrics_std,
        "train_size": total_train,
        "test_size": total_test,
        "split_strategy": strategy,
        "per_file_results": per_fold_results if strategy == "lofo" else None,
        "per_fold_results": per_fold_results,
    }
    importance_list = [r["feature_importance"] for r in per_fold_results if "feature_importance" in r]
    if importance_list:
        names = list(importance_list[0].keys())
        out["feature_importance"] = {n: float(np.mean([d[n] for d in importance_list])) for n in names}
        out["feature_importance_std"] = {n: float(np.std([d[n] for d in importance_list])) for n in names}
    if verbose:
        print_evaluation_summary(metrics_mean, metrics_std, strategy, n_folds_done)
    return out


def run_ml_pipeline(
    data_dir,
    model,
    feature_cols,
    label_col="label",
    file_col="source_file",
    strategy="lofo",
    test_size=0.2,
    random_state=42,
    n_folds=None,
    feature_injectors=None,
):
    from .data_loader import load_iot23_dataset

    print_header()
    df, meta = load_iot23_dataset(data_dir=data_dir)
    print(f"\nIoT-23 Dataset shape: {df.shape}")
    print(df.head())
    if feature_injectors:
        for inj in feature_injectors:
            if callable(inj):
                df = inj(df)
            else:
                df = merge_pm_metrics(
                    df,
                    metrics_df=inj.get("metrics_df"),
                    merge_on=inj.get("merge_on", "uid"),
                )
        print(f"\nMerged Dataset with PM metrics shape: {df.shape}")
        print(df.head())
    log = meta.get("processing_log", [])
    print_dataset_info(log)
    if df.empty:
        return {
            "success": False,
            "error": "No data loaded (no files found or all empty).",
            "validation": None,
            "metrics": {},
            "train_size": 0,
            "test_size": 0,
            "split_strategy": strategy,
        }
    validation = check_df(df, label_col=label_col, feature_cols=feature_cols, file_col=file_col)
    print_dataset_validation(validation)
    if not validation["valid"]:
        return {
            "success": False,
            "error": "Validation failed",
            "validation": validation,
            "metrics": {},
            "train_size": 0,
            "test_size": 0,
            "split_strategy": strategy,
        }
    print_split_strategy(strategy)
    result = run_evaluation(
        data=df,
        model=model,
        feature_cols=feature_cols,
        label_col=label_col,
        file_col=file_col,
        strategy=strategy,
        test_size=test_size,
        random_state=random_state,
        n_folds=n_folds,
        verbose=True,
    )
    print_result_summary(result)
    return result


def print_result_summary(result):
    if result.get("success"):
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print("\nDataset Information:")
        print(f"  - Strategy: {result['split_strategy']}")
        print(f"  - Training samples: {result['train_size']}")
        print(f"  - Test samples: {result['test_size']}")
        print("\nModel Performance:")
        std = result.get("metrics_std") or {}
        for name, val in result["metrics"].items():
            s = std.get(name)
            if s is not None:
                print(f"  - {name.replace('_', ' ').title()}: {val:.4f} ± {s:.4f}")
            else:
                print(f"  - {name.replace('_', ' ').title()}: {val:.4f}")
        if result.get("feature_importance"):
            print("\nFeature Importance (aggregated across folds):")
            print_feature_importance(
                result["feature_importance"],
                indent="  ",
                imp_std=result.get("feature_importance_std"),
            )
        print("=" * 70)
    else:
        print(f"\nEvaluation failed: {result.get('error', 'Unknown error')}")
