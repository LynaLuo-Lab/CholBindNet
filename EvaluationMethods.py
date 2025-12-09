import pandas as pd
import numpy as np

def apply_labeling_from_spies(df, mean_score, percentile_50, percentile_25, percentile_75, min_score, max_score):
    def get_min_max_label(score):
        if score < min_score:
            return "StrongNegative"
        elif min_score <= score < mean_score:
            return "PseudoNegative"
        elif mean_score <= score < max_score:
            return "PseudoPositive"
        else:
            return "StrongPositive"

    df["min_max_label"] = df["average_score"].apply(get_min_max_label)

    def get_percentile_label(score):
        if score < percentile_25:
            return "StrongNegative"
        elif percentile_25 <= score < percentile_50:
            return "PseudoNegative"
        elif percentile_50 <= score < percentile_75:
            return "PseudoPositive"
        else:
            return "StrongPositive"

    df["percentile_label"] = df["average_score"].apply(get_percentile_label)
    df["threshold_label"] = df["average_score"].apply(lambda x: "Negative" if x < 0.5 else "Positive")

    min_max_results = []
    min_max_results.append((df["average_score"] < min_score).sum()) # min_max_strong_negative_count
    min_max_results.append(((df["average_score"] > min_score) & (df["average_score"] < mean_score)).sum()) # min_max_pseudo_negative_count
    min_max_results.append(((df["average_score"] > mean_score) & (df["average_score"] < max_score)).sum()) # min_max_pseudo_positive_count
    min_max_results.append((df["average_score"] > max_score).sum()) # min_max_strong_positive_count

    percentile_results = []
    percentile_results.append((df["average_score"] < percentile_25).sum()) # percentile_strong_negative_count
    percentile_results.append(((df["average_score"] > percentile_25) & (df["average_score"] < percentile_50)).sum()) # percentile_pseudo_negative_count
    percentile_results.append(((df["average_score"] > percentile_50) & (df["average_score"] < percentile_75)).sum()) # percentile_pseudo_positive_count
    percentile_results.append((df["average_score"] > percentile_75).sum()) # percentile_strong_positive_count

    threshold_results = []
    threshold_results.append((df["average_score"] < 0.5).sum()) # threshold_negative_count
    threshold_results.append((df["average_score"] > 0.5).sum()) # threshold_positive_count

    # Table 1: Min/Max and Percentile
    labels_4 = ["StrongNegative", "PseudoNegative", "PseudoPositive", "StrongPositive"]
    minmax_percentile_df = pd.DataFrame({
        "Label": labels_4,
        "Min_Max (Using Mean)": min_max_results,
        "Percentile (75-50-25)": percentile_results
    })

    # Table 2: Threshold 0.5
    labels_2 = ["Negative", "Positive"]
    threshold_df = pd.DataFrame({
        "Label": labels_2,
        "Threshold (0.5)": threshold_results
    })

    print(minmax_percentile_df,"\n", threshold_df)

    return df, min_max_results, percentile_results, threshold_results

def get_spy_info(csv):
    df = pd.read_csv(csv)  # Replace with your actual filename if needed

    # Extract the 'average_score' column
    scores = df["average_score"]

    # Compute statistics
    mean_score = scores.mean()
    percentile_50 = scores.quantile(0.5)
    percentile_25 = scores.quantile(0.25)
    percentile_75 = scores.quantile(0.75)
    min_score = scores.min()
    max_score = scores.max()

    # Print results
    # print(f"Mean: {mean_score:.6f}")
    # print(f"50th Percentile (Median): {percentile_50:.6f}")
    # print(f"Bottom 25 Percentile: {percentile_25:.6f}")
    # print(f"Top 25 Percentile: {percentile_75:.6f}")
    # print(f"Min: {min_score:.6f}")
    # print(f"Max: {max_score:.6f}\n")

    return mean_score, percentile_50, percentile_25, percentile_75, min_score, max_score

def show_evaluation_results(min_max_all, percentile_all, threshold_all):
    # Compute mean and std for each label
    min_max_mean = np.mean(min_max_all, axis=0)
    min_max_std = np.std(min_max_all, axis=0)

    percentile_mean = np.mean(percentile_all, axis=0)
    percentile_std = np.std(percentile_all, axis=0)

    threshold_mean = np.mean(threshold_all, axis=0)
    threshold_std = np.std(threshold_all, axis=0)

    # Create DataFrames to display
    labels_4 = ["StrongNegative", "PseudoNegative", "PseudoPositive", "StrongPositive"]
    labels_2 = ["Negative", "Positive"]

    min_max_df = pd.DataFrame({
        "Label": labels_4,
        "Min-Max Mean": min_max_mean,
        "Min-Max Std": min_max_std
    })

    percentile_df = pd.DataFrame({
        "Label": labels_4,
        "Percentile Mean": percentile_mean,
        "Percentile Std": percentile_std
    })

    threshold_df = pd.DataFrame({
        "Label": labels_2,
        "Threshold Mean": threshold_mean,
        "Threshold Std": threshold_std
    })

    # Display
    print("=== Min-Max Results ===")
    print(min_max_df)
    print("\n=== Percentile Results ===")
    print(percentile_df)
    print("\n=== Threshold Results ===")
    print(threshold_df)