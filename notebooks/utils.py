import os
import re
import ast

import numpy as np
import pandas as pd
from collections import defaultdict, Counter

from typing import List, Dict, Any
from scipy.stats import sem
from datetime import datetime

"""
This is a collection of utility functions to parse logs and extract statistics.
Might seemm arbitrary, but it was made just for the log structure we had at the time of submission.
"""

BENCHMARK_MAP = {
    "game24" : "Game of 24",
    "hle": "Humanity's Last Exam",
    "hotpotqa": "HotpotQA",
    "humaneval": "HumanEval",
    "scibench": "SciBench",
    "sonnetwriting": "Creative Writing",
    "all": "All"
}

def get_subfolders(root: str):
    return [
        os.path.join(root, name)
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]

def get_files(root: str):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files

def eval_logs(log):
    parsed_log = {}
    for k, v in log.items():
        if isinstance(v, str):
            parsed_log[k] = ast.literal_eval(f"{v}")
        elif isinstance(v, dict):
            parsed_log[k] = eval_logs(v)
        else:
            raise ValueError(f"Unexpected type {type(v)} for key {k}")
    if parsed_log == {}:
        print("problem here")
    return parsed_log

def parse_log(path: str) -> dict:
    """
    Parse a tab-indented text block into a nested dictionary.
    Lines ending with ":" are treated as new dictionary keys.
    Lines with "key: value" become key-value pairs.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    result = {}
    stack = [(0, result)]  # (indent_level, current_dict)

    for line in text.splitlines():
        if not line.strip():
            continue

        # Count indentation (tabs only)
        indent = len(line) - len(line.lstrip("\t"))
        key_value = line.strip().split(":", 1)

        if len(key_value) == 2:
            key, value = key_value
            key, value = key.strip(), value.strip()
        else:
            key, value = key_value[0].strip(), {}

        # Find parent dictionary for this indentation
        while stack and stack[-1][0] >= indent + 1:
            stack.pop()

        current_dict = stack[-1][1]

        if value == "":
            current_dict[key] = {}
            stack.append((indent + 1, current_dict[key]))
        else:
            current_dict[key] = value
    result["Log path"] = f"'{path}'"
    try:
        log = eval_logs(result)
    except:
        print(f"Error with log: {path}")
        log = {}
    return log

def get_logs(
        logs_path: str,
        experiment: str,
        model: str,
        benchmarks: List[str],
        methods: List[str],
        split: str = "test",
        verbose: bool = True,
        max_repeats: int = None,
        run: int = None
    ) -> List[dict]:
    
    root_path = os.path.join(logs_path, f"{experiment}/{model}")
    benchmarks = benchmarks or [path.split("/")[-1] for path in get_subfolders(root_path)]
    all_logs = []
    for benchmark in benchmarks:
        # Get all log paths for the current benchmark
        path = root_path + f"/{benchmark}"
        log_paths = get_files(path)

        # Remove logs from other splits
        logs_paths =[path for path in log_paths if split in path]

        # Limit number of logs
        if max_repeats:
            logs_paths = [log_path for log_path in logs_paths if int(log_path.split('.log')[0].split('_')[-1]) < max_repeats]

        if run:
            logs_paths = [log_path for log_path in logs_paths if int(log_path.split('.log')[0].split('_')[-1]) == run]

        # Parse current logs
        logs = [parse_log(path) for path in logs_paths]

        # Remove logs of a different method
        if methods:
            logs = [log for log in logs if log["General information"]["Method"] in methods]
        all_logs.extend(logs)

        methods_current = [log["General information"]["Method"] for log in logs]
        count = Counter(methods_current)
        
        if verbose:
            print(f"{benchmark.capitalize()} logs retrieved ({len(logs)}):")
            for method in sorted(count):
                print(f"\t{method}: {count[method]} runs")
    
    if verbose:
        print(f"Total logs retrieved: {len(all_logs)}")
    return all_logs


def get_cis(samples: np.array) -> List[float]:

    if len(samples) <= 1:
        return None, None
    
    standard_error = sem(samples)

    low_ci = np.mean(samples) - 2 * standard_error
    high_ci = np.mean(samples) + 2 * standard_error
    return low_ci, high_ci

def get_quality_stats(logs: List[dict], verbose=True,) -> dict:
        
    problematic_logs = [log for log in logs if "Correct" not in log.get("Quality", {})]
    logs = [log for log in logs if "Correct" in log.get("Quality", {})]


    
    if verbose and problematic_logs:
        print("'Correct' not found in the following logs:")
        for log in problematic_logs:
            print(f"\t{log['Log path']}")

    samples = np.ravel([log["Quality"]["Correct"] for log in logs])



    low_ci, high_ci = get_cis(samples)
    mean = samples.mean()
    p5 = np.percentile([np.mean(log["Quality"]["Correct"]) for log in logs], 5)

    return {
        "mean": mean,
        "low_ci": low_ci,
        "high_ci": high_ci,
        "margin": (high_ci - low_ci)/2,
        "p5": p5,
        "samples": samples
    }

def get_cost_stats(logs: List[dict], summed: bool) -> dict:
    #costs = [get_cost(log) for log in logs]
    costs = []
    for log in logs:
        try:
            costs.append(get_cost(log))
        except Exception as e:
            continue
    stats = {}

    if summed:
        costs = [
            {
                "tokens_in": np.sum(cost["tokens_in"]),
                "tokens_out": np.sum(cost["tokens_out"]),
                "cost": np.sum(cost["cost"])
                }
                for cost in costs
            ]
    for key in ["tokens_in", "tokens_out", "cost"]:
        samples = np.ravel([cost[key] for cost in costs])
        low_ci, high_ci = get_cis(samples)
        mean = samples.mean()
        p95 = np.percentile(samples, 95)

        stats[key] = {
            "mean": mean,
            "low_ci": low_ci,
            "high_ci": high_ci,
            "margin": (high_ci - low_ci)/2 if low_ci is not None else None,
            "p95": p95,
            "samples": samples
        }
    return stats

def save_dataframe(
        df: pd.DataFrame,
        data_path: str,
        experiment: str,
        model: str,
    ):
    
    now =  datetime.now()
    
    # Path to output
    if experiment:
        out_path = os.path.join(data_path, f"{experiment}/{model}/{now.strftime('%d')}/{now.strftime('%H:%M')}.parquet")
    else:
        out_path = os.path.join(data_path, f"{now.strftime('%d')}/{now.strftime('%H:%M')}.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save dataframe to parquet
    df.to_parquet(out_path)
    if experiment:
        df.to_parquet(os.path.join(data_path, f"{experiment}/{model}/latest.parquet"))
    else:
        df.to_parquet(os.path.join(data_path, f"latest.parquet"))

def get_cost(log):
    cost = defaultdict(list)
    for tab, info in log.get("API Detailed Information (per tab)", {}).items():
        cost["tokens_in"].append(info["Tokens (total)"]["in"])
        cost["tokens_out"].append(info["Tokens (total)"]["out"])
        cost["cost"].append(info["Cost (total)"]["total"])
    #print(log["Log path"])

    assert np.isclose(np.sum(cost["cost"]), log["All Tabs"]["Cost (total)"]["total"], atol=1e-08), f"Cost mismatch at {log['Log path']}"
    assert np.isclose(np.sum(cost["tokens_in"]), log["All Tabs"]["Tokens (total)"]["in"], atol=1e-08), f"Tokens in mismatch at {log['Log path']}"
    assert np.isclose(np.sum(cost["tokens_out"]), log["All Tabs"]["Tokens (total)"]["out"], atol=1e-08), f"Tokens out mismatch at {log['Log path']}"
    
    return cost

def get_calls(
        logs_path: str,
        experiment: str,
        model: str,
        split: str = "test",
        benchmarks: List[str] = [],
        methods: List[str] = [],
        verbose: bool = True,
        max_repeats: int = None,
        run: int = None
    ) -> List[dict]:

    # Set the root path for all raw call logs saved for this experiment and model
    root_path = os.path.join(logs_path, "raw_calls" , f"{experiment}/{model}")

    # If no benchmarks are provided, get all benchmarks based on the subfolders in the root path
    benchmarks = benchmarks or [path.split("/")[-1] for path in get_subfolders(root_path)]

    all_calls = []
    for benchmark in benchmarks:
        # Get all log paths for the current benchmark
        path = root_path + f"/{benchmark}"
        log_paths = get_files(path)

        # Remove logs from other splits
        logs_paths =[path for path in log_paths if split in path]

        # Filter number of logs by max_repeats if provided
        if max_repeats:
            logs_paths = [log_path for log_path in logs_paths if int(log_path.split('.log')[0].split('_')[-1]) < max_repeats]

        # Filter by specific run if provided
        if run:
            logs_paths = [log_path for log_path in logs_paths if int(log_path.split('.log')[0].split('_')[-1]) == run]

        # Filter by methods if provided
        if methods:
            log_paths = [log_path for log_path in log_paths if log_path.split("/")[-1].split["_"][0] in methods]

        # Parse current logs
        calls = [
            {
                "benchmark":benchmark, 
                "calls":load_call(path), 
                "path":path
            } 
            for path in log_paths
        ]

        # Extend all calls
        all_calls.extend(calls)

        if verbose:
            print(f"{benchmark.capitalize()} calls retrieved ({len(calls)} logs).")

    if verbose:
        print(f"Total calls retrieved: {len(all_calls)} logs.")
    return all_calls


def load_call(path: str) -> List[Dict[str, Any]]:
    """
    Given a log file path, it reads the file, splits the calls and parses them.
    Log format must be as expected (see logs).
    Input
    -----
        path: str
            Path to the log file.
    Output
    ------
        List of parsed calls (dictionaries).
    """

    # Separates different calls in the log file
    DELIMITER = ("="*100 + "\n")*2 + "="*100 

    # Read the log file
    with open(path, "r") as f:
        text = f.read().strip()
    
    # Split the text into individual calls
    calls = [s for s in text.split(DELIMITER) if s]
    
    # Parse each call
    parsed_calls = [parse_call(call) for call in calls]

    return parsed_calls

def parse_call(text: str) -> Dict[str, Any]:
    """
    Regex-based parser for the log format used in the API calls.
    Parses a single call to the api and its responses.
    God took the wheel in regex expressions but they seem to work.
    
    Input
    -----
    text: str
        The text of a single API call log. in the expected format (see logs)

    Output
    ------
    Dictionary with keys:
        "user_message": str
            The user message sent to the API.
        "N": int
            The number of responses requested/expected.
        "responses": List[str]
            The list of responses received from the API.
    """

    STARLINE = r"^\*+\s*$"  # a line that's all stars (any length)

    USER_BANNER_RE = re.compile(
        rf"{STARLINE}\n^\*\s*USER\s*\*\s*$\n{STARLINE}\n",
        re.MULTILINE
    )

    N_BANNER_RE = re.compile(
        rf"{STARLINE}\n^\*\s*N:\s*(\d+)\s*\*\s*$\n{STARLINE}\n",
        re.MULTILINE
    )

    RESP_BANNER_RE = re.compile(
        rf"{STARLINE}\n^\*\s*RESPONSE\s+(\d+)\s*\*\s*$\n{STARLINE}\n",
        re.MULTILINE
    )
    user_m = USER_BANNER_RE.search(text)
    if not user_m:
        raise ValueError("USER banner not found")

    n_m = N_BANNER_RE.search(text, pos=user_m.end())
    if not n_m:
        raise ValueError("N banner not found")

    user_message = text[user_m.end():n_m.start()].strip("\n")
    n_value = int(n_m.group(1))

    resp_matches = list(RESP_BANNER_RE.finditer(text, pos=n_m.end()))
    responses: List[Dict[str, Any]] = []

    for i, m in enumerate(resp_matches):
        idx = int(m.group(1))
        start = m.end()
        end = resp_matches[i + 1].start() if i + 1 < len(resp_matches) else len(text)
        body = text[start:end].strip("\n")
        responses.append(body)

    return {"user_message": user_message, "N": n_value, "responses": responses}


# For a given benchmark-model combo, extract sample-level token vs score relationships
def extract_sample_data(df, benchmark, model):
    """Extract per-sample token and score data for a benchmark-model combination."""
    combo_data = df[(df['Benchmark'] == benchmark) & (df['Model'] == model)]
    
    if len(combo_data) == 0:
        return None
    
    # Combine data across all experiments
    sample_data = []
    
    for _, row in combo_data.iterrows():
        tokens = row['tokens_out']
        scores = row['scores']
        
        # Each sample has a token count and a score
        if isinstance(tokens, (list, np.ndarray)) and isinstance(scores, (list, np.ndarray)):
            for sample_idx, (token, score) in enumerate(zip(tokens, scores)):
                sample_data.append({
                    'sample_idx': sample_idx,
                    'tokens': token,
                    'score': score,
                    'benchmark': benchmark,
                    'model': model
                })
    
    return pd.DataFrame(sample_data)

def categorize_samples(df, benchmark, model):
    """Categorize all samples in a benchmark-model combination by token sensitivity."""
    sample_data = extract_sample_data(df, benchmark, model)
    
    if sample_data is None:
        return None
    
    categories = {
        'ALWAYS_SUCCEEDS': [],
        'ALWAYS_FAILS': [],
        'TOKEN_SENSITIVE_POS': [],
        'TOKEN_AVERSE': [],
        'TOKEN_AGNOSTIC': [],
        'WEAK_DEPENDENCY': []
    }
    
    correlations = []
    success_rates = []
    
    # Analyze each sample
    for sample_idx in sorted(sample_data['sample_idx'].unique()):
        sample = sample_data[sample_data['sample_idx'] == sample_idx]
        
        success_rate = sample['score'].mean()
        token_corr = sample['tokens'].corr(sample['score']) if len(sample) > 1 else 0
        
        success_rates.append(success_rate)
        correlations.append(token_corr)
        
        # Categorize
        if success_rate == 1.0:
            categories['ALWAYS_SUCCEEDS'].append(sample_idx)
        elif success_rate == 0.0:
            categories['ALWAYS_FAILS'].append(sample_idx)
        elif token_corr > 0.3:
            categories['TOKEN_SENSITIVE_POS'].append(sample_idx)
        elif token_corr < -0.3:
            categories['TOKEN_AVERSE'].append(sample_idx)
        elif abs(token_corr) < 0.1:
            categories['TOKEN_AGNOSTIC'].append(sample_idx)
        else:
            categories['WEAK_DEPENDENCY'].append(sample_idx)
    
    return {
        'categories': categories,
        'correlations': correlations,
        'success_rates': success_rates,
        'num_samples': len(sample_data['sample_idx'].unique())
    }

#=============== Lars Functions ===============#
    
def load_data(path: str, mode: str, scores: bool=False, columns: list=[]) -> pd.DataFrame:
    """Load and preprocess data."""
    
    assert mode in ["Method", "Model", "Strategy"], "Only meant to be used with the Method/Model columns"
    df = pd.read_parquet(path)
    
    # Aggregation: Mean score per run (if individual questions exist)
    if "scores" in df.columns and isinstance(df["scores"].iloc[0], (list, np.ndarray)):
         df["Score"] = df["scores"].apply(lambda x: np.mean(x))

    # Aggregation: Sum cost per run (if individual questions exist)
    if "costs" in df.columns and isinstance(df["costs"].iloc[0], (list, np.ndarray)):
         df["Cost"] = df["costs"].apply(lambda x: np.sum(x))
    
    if "tokens_in" in df.columns and isinstance(df["tokens_in"].iloc[0], (list, np.ndarray)):
         df["TokensIn"] = df["tokens_in"].apply(lambda x: np.sum(x))

    if "tokens_out" in df.columns and isinstance(df["tokens_out"].iloc[0], (list, np.ndarray)):
         df["TokensOut"] = df["tokens_out"].apply(lambda x: np.sum(x))

    # Formatting
    df = df.rename(columns={"Method": "Strategy"})
    df[mode] = df[mode].astype("str")
    df["Benchmark"] = df["Benchmark"].astype("str")
    
    return df[[mode, "Benchmark", "Score", "Cost", "TokensIn", "TokensOut"]+columns]
    if scores:
        return df[[mode, "Benchmark", "Score", "Cost", "scores"]+columns]
    else:
        return df[[mode, "Benchmark", "Score", "Cost"]+columns]

def calculate_stability_metrics(df: pd.DataFrame, mode: str, col: str = "Score"):
    """
    Metric 2: Stability (The Noise).
    Calculate Variance of Z-Scores to measure inconsistency.
    """
    print(f"Calculating Z-Score Variance (Stability) for '{col}'...")

    # 1. Standardize (Z-Score) within each Benchmark
    # This removes "Difficulty" and "Scale" (Heteroscedasticity)
    def standardize(x):
        std = x.std()
        if pd.isna(std) or std == 0:
            return np.zeros_like(x) # Invariant benchmark -> Neutral signal
        return (x - x.mean()) / std

    # Use a temporary column name to avoid overwriting if analyzing multiple cols
    z_col_name = f"Z_{col}"
    df[z_col_name] = df.groupby("Benchmark")[col].transform(standardize)
    
    # 2a. Global Noise: Variance of ALL Z-scores for a strategy
    global_z_var = df.groupby(mode)[z_col_name].var()
    
    # 2b. Avg Run-to-Run Noise: Variance within each cell, averaged
    cell_vars = df.groupby([mode, "Benchmark"])[z_col_name].var()
    avg_run_noise = cell_vars.groupby(mode).mean()

    # Combine
    z_stats = pd.DataFrame({
        "Global_Noise": global_z_var,
        "Run_Noise": avg_run_noise
    })
    
    return z_stats

def bootstrap_confidence_intervals(df: pd.DataFrame, mode: str, col: str = "Score", n_bootstrap: int = 1000, 
                                 confidence: float = 0.95, seed: int = 42) -> pd.DataFrame:
    """
    Metric 3: Confidence Intervals (Cluster Bootstrap).
    Resamples BENCHMARKS (not runs) to account for task sampling uncertainty.
    """
    print(f"Bootstrapping Confidence Intervals for '{col}' (n={n_bootstrap})...")
    rng = np.random.default_rng(seed)
    strategies = sorted(df[mode].unique())
    benchmarks = df["Benchmark"].unique()
    
    boot_means = {s: [] for s in strategies}
    
    for _ in range(n_bootstrap):
        # Cluster Bootstrap: Resample benchmarks with replacement
        boot_benchmarks = rng.choice(benchmarks, size=len(benchmarks), replace=True)
        
        boot_dfs = []
        for i, b in enumerate(boot_benchmarks):
            subset = df[df["Benchmark"] == b].copy()
            subset["Benchmark"] = f"{b}_{i}" 
            boot_dfs.append(subset)
            
        boot_df = pd.concat(boot_dfs)
        means = boot_df.groupby(mode)[col].mean()
        
        for s in strategies:
            boot_means[s].append(means.get(s, np.nan))
            
    # Compile CIs
    alpha = 1 - confidence
    results = []
    for s in strategies:
        samples = np.array(boot_means[s])
        samples = samples[~np.isnan(samples)]
        
        if len(samples) > 0:
            lower = np.percentile(samples, 100 * alpha / 2)
            upper = np.percentile(samples, 100 * (1 - alpha / 2))
            results.append({
                mode: s,
                "CI_Lower": lower,
                "CI_Upper": upper,
                "CI_Formatted": f"[{lower:.2f}, {upper:.2f}]"
            })
            
    return pd.DataFrame(results).set_index(mode)

def stratified_bootstrap_ci(df: pd.DataFrame, mode: str, col: str = "Score", n_bootstrap: int = 1000, 
                            confidence: float = 0.95, seed: int = 42, fn = np.mean) -> pd.DataFrame:
    """
    Stratified Bootstrap: Fixed Benchmarks, Random Runs.
    Simulates variability if we re-ran the exact same benchmark suite.
    """
    print(f"Bootstrapping Stratified CIs for '{col}' (n={n_bootstrap})...")
    rng = np.random.default_rng(seed)
    strategies = df[mode].unique()
    benchmarks = df["Benchmark"].unique()
    
    # Pre-organize runs for speed: {Strategy: {Bench: [scores...]}}
    data_map = {}
    for s in strategies:
        s_df = df[df[mode] == s]
        # Drop NaNs just in case
        s_df = s_df.dropna(subset=[col])
        data_map[s] = s_df.groupby("Benchmark")[col].apply(np.array).to_dict()

    boot_means = {s: [] for s in strategies}

    for _ in range(n_bootstrap):
        for s in strategies: # also works for models if mode==Model
            # Construct a synthetic 'pass' over the fixed benchmark suite
            iteration_scores = []
            for b in benchmarks:
                # Get all runs for this Strategy-Benchmark pair
                available_runs = data_map[s].get(b, [])
                if len(available_runs) > 0:
                    # Stratified step: Sample ONE run from this specific bin
                    selected = rng.choice(available_runs)
                    iteration_scores.append(selected)
                else:
                    continue
                    assert False, "missing data"

            # Record average score for this bootstrap iteration
            if iteration_scores:
                boot_means[s].append(fn(iteration_scores))
            else:
                continue
                assert False, "missing data"

    # Calculate CIs
    alpha = 1 - confidence
    results = []
    diff = 0
    for s in strategies:
        samples = np.array(boot_means[s])
        
        if len(samples) > 0:
            lower = np.percentile(samples, 100 * alpha / 2) * 100
            upper = np.percentile(samples, 100 * (1 - alpha / 2)) * 100
            results.append({
                mode: s,
                "Strat_CI_Formatted": f"[{lower:.2f}, {upper:.2f}]"
            })
            temp = upper - lower
            if temp > diff:
                diff = temp
                strat=s
            
    return pd.DataFrame(results).set_index(mode)

def bootstrap_relative_instability(df: pd.DataFrame, mode: str, col: str = "Score", n_bootstrap: int = 1000, 
                                 seed: int = 42, eps: float = 1e-7) -> pd.DataFrame:
    """
    Relative Instability (MAPE-like metric):
    Measures % deviation of individual runs from the strategy's own mean on that benchmark.
    """
    print(f"Bootstrapping Relative Instability for '{col}' (n={n_bootstrap})...")
    
    # 1. Precompute Strategy x Benchmark Means (The Reference)
    bench_means = df.groupby([mode, "Benchmark"])[col].mean()
    
    # 2. Attach reference mean to every row
    # Merge is safer than map for multi-index alignment
    df_aug = df.merge(bench_means.rename("Bench_Mean"), on=[mode, "Benchmark"])
    
    # 3. Calculate Relative Absolute Error for every run
    # |RunScore - BenchMean| / (BenchMean + eps)
    df_aug["Rel_Error"] = (
        (df_aug[col] - df_aug["Bench_Mean"]).abs() / 
        (df_aug["Bench_Mean"] + eps)
    )
    
    # We now want to Bootstrap the MEAN of these Rel_Errors per strategy
    # Stratified Bootstrap (Fixed Benchmarks, Random Runs) is appropriate for "Run Instability"
    rng = np.random.default_rng(seed)
    strategies = df[mode].unique()
    print(f"Strats: {strategies}")
    benchmarks = df["Benchmark"].unique()
    
    # Map for Stratified Sampling: {Strategy: {Bench: [Rel_Errors...]}}
    data_map = {}
    for s in strategies:
        s_df = df_aug[df_aug[mode] == s]
        data_map[s] = s_df.groupby("Benchmark")["Rel_Error"].apply(np.array).to_dict()

    boot_means = {s: [] for s in strategies}

    for _ in range(n_bootstrap):
        for s in strategies:
            iteration_errors = []
            for b in benchmarks:
                available = data_map[s].get(b, [])
                if len(available) > 0:
                    selected = rng.choice(available)
                    iteration_errors.append(selected)
            
            if iteration_errors:
                boot_means[s].append(np.mean(iteration_errors))

    # Compile Results (Mean + CI)
    results = []
    for s in strategies:
        samples = np.array(boot_means[s])
        if len(samples) > 0:
            mean_instability = np.mean(samples)
            lower = np.percentile(samples, 2.5)
            upper = np.percentile(samples, 97.5)
            results.append({
                mode: s,
                "Rel_Instability_Mean": mean_instability*100, 
                "Rel_Instability_CI": f"[{lower*100:.2f}, {upper*100:.2f}]"
            })
            
    return pd.DataFrame(results).set_index(mode)

def compare_ranking_methodologies(df: pd.DataFrame, mode: str):
    """
    Explore three definitions of 'Best Strategy':
    1. Magnitude (Z-Score): Rewards high peaks on hard tasks.
    2. Consistency (Borda): Average ranking position (1st, 2nd...) per benchmark.
    3. Dominance (Pairwise): Win-rate against other strategies in head-to-head comparisons.
    """
    print("Calculating Multi-View Rankings...")
    strategies = sorted(df[mode].unique())
    
    # 1. Z-Score Ranking (Magnitude)
    # Ensure Z-Score exists (calculated in stability step, but recalculate to be safe/independent)
    if "Z_Score" not in df.columns:
         df["Z_Score"] = df.groupby("Benchmark")["Score"].transform(lambda x: (x - x.mean()) / x.std())
    
    z_means = df.groupby(mode)["Z_Score"].mean()
    
    # 2. Borda Count (Average Rank of STRATEGIES, not runs)
    # Standard Borda: Each Benchmark is a "voter" that ranks the Strategies.
    # We rank the *Average Score* of each strategy on each benchmark.
    bench_strategy_means = df.groupby(["Benchmark", mode])["Score"].mean().reset_index()
    bench_strategy_means["Rank"] = bench_strategy_means.groupby("Benchmark")["Score"].rank(ascending=False, method="min")
    borda_scores = bench_strategy_means.groupby(mode)["Rank"].mean()
    
    # 3. Pairwise Win Rate (dominance)
    # Synthesize a Round-Robin tournament
    pivot = df.pivot_table(index="Benchmark", columns=mode, values="Score", aggfunc="mean")
    win_matrix = pd.DataFrame(0.0, index=strategies, columns=strategies)
    
    for s_a in strategies:
        for s_b in strategies:
            if s_a == s_b: continue
            # Vectorized comparison across all benchmarks
            diff = pivot[s_a] - pivot[s_b]
            wins = (diff > 0).sum()
            ties = (diff == 0).sum()
            total = diff.count() # Count benchmarks where both exist
            
            if total > 0:
                win_rate = (wins + 0.5 * ties) / total
                win_matrix.loc[s_a, s_b] = win_rate
            else:
                win_matrix.loc[s_a, s_b] = np.nan
                
    # Average Win Rate against all opponents
    avg_win_rate = win_matrix.mean(axis=1)
    
    # --- Compile The Summary ---
    summary = pd.DataFrame({
        "Score (Avg)": df.groupby(mode)["Score"].mean(),
        "Z-Score (Avg)": z_means,
        "Avg Rank (Borda)": borda_scores,
        "Win Rate (Pairwise)": avg_win_rate
    })
    
    # Create ordinal rankings for final comparison (1 = Best)
    summary["R_Raw"] = summary["Score (Avg)"].rank(ascending=False)
    summary["R_Z"] = summary["Z-Score (Avg)"].rank(ascending=False)
    summary["R_Borda"] = summary["Avg Rank (Borda)"].rank(ascending=True) # Lower is better
    summary["R_Win"] = summary["Win Rate (Pairwise)"].rank(ascending=False)
    
    # Order by Win Rate for the view
    summary = summary.sort_values("R_Win")
    
    print("\n=== Ranking Methodology Comparison ===")
    print("R_Raw   = Based on Raw Score Mean (Sensitive to scale/outliers)")
    print("R_Z     = Based on Z-Scores (Standardized difficulty)")
    print("R_Borda = Based on Avg Position (Robust to outliers)")
    print("R_Win   = Based on Head-to-Head Win % (Tournament style)")
    print("-" * 60)
    
    cols_metrics = ["Score (Avg)", "Z-Score (Avg)", "Avg Rank (Borda)", "Win Rate (Pairwise)"]
    print(summary[cols_metrics].round(3).to_string())
    
    print("\n--- The Ordinal Ranks (1=Best) ---")
    cols_ranks = ["R_Raw", "R_Z", "R_Borda", "R_Win"]
    print(summary[cols_ranks].astype(int).to_string())
    
    return summary


def analyze_correlations(df: pd.DataFrame, mode: str):
    """Metric 4: Correlations (Clustering)."""
    pivot = df.pivot_table(index="Benchmark", columns=mode, values="Score", aggfunc="mean")
    return pivot.corr(method="spearman")


import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


def _parse_ci(x) -> Tuple[float, float]:
    if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 2:
        return float(x[0]), float(x[1])
    if isinstance(x, str):
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x.strip())
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
    raise ValueError(f"Could not parse CI from value: {x!r}")


def _parse_percent(x) -> float:
    if isinstance(x, str):
        s = x.strip()
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        return float(s)
    v = float(x)
    return v / 100.0 if v > 1.0 else v


def _safe_yerr(mean, low, high) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)

    low, high = np.minimum(low, high), np.maximum(low, high)
    err_low = np.maximum(0.0, mean - low)
    err_high = np.maximum(0.0, high - mean)
    return np.vstack([err_low, err_high])


def plot_model_metrics_2x2(
    df: pd.DataFrame,
    *,
    model_col: Optional[str] = None,
    sort_by: Optional[str] = "Score (Avg)",
    ascending: bool = False,
    capsize: int = 4,
    figsize: Tuple[float, float] = (14, 10),
    rotate_xticks: int = 25,
    score_ylim: Optional[Tuple[float, float]] = None,
    instability_as_percent_axis: bool = True,
    legend_fontsize: int = 11,
    y_label_fontsize: int = 13,
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """
    2x2 plot:
      A) Score (Avg) with 95% CI (Stratified) errorbars
      B) Rel. Run Instability with 95% CI (Instability) errorbars
      C) Global Noise (Z-Var) bar plot
      D) Run Noise (Z-Var) bar plot

    Returns (fig, axes, parsed_df)
    """
    required = [
        "Score (Avg)",
        "95% CI (Stratified)",
        "Rel. Run Instability",
        "95% CI (Instability)",
        "Global Noise (Z-Var)",
        "Run Noise (Z-Var)",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame is missing required columns: {missing}")

    d = df.copy()

    # Model labels
    if model_col is not None:
        if model_col not in d.columns:
            raise KeyError(f"model_col={model_col!r} not found in DataFrame columns.")
        d["_model_label"] = d[model_col].astype(str)
    else:
        d["_model_label"] = d.index.astype(str)

    # Parse numeric fields
    d["_score"] = pd.to_numeric(d["Score (Avg)"], errors="raise")
    d[["_score_ci_low", "_score_ci_high"]] = d["95% CI (Stratified)"].apply(
        lambda x: pd.Series(_parse_ci(x))
    )

    d["_instab"] = d["Rel. Run Instability"].apply(_parse_percent)
    d[["_instab_ci_low", "_instab_ci_high"]] = d["95% CI (Instability)"].apply(
        lambda x: pd.Series(_parse_ci(x))
    )

    d["_global_noise"] = pd.to_numeric(d["Global Noise (Z-Var)"], errors="raise")
    d["_run_noise"] = pd.to_numeric(d["Run Noise (Z-Var)"], errors="raise")

    # Optional sort
    if sort_by is not None:
        if sort_by not in d.columns:
            raise KeyError(f"sort_by={sort_by!r} not found in DataFrame columns.")
        d = d.sort_values(sort_by, ascending=ascending)

    labels = d["_model_label"].tolist()
    n = len(d)
    x = np.arange(n)

    # --- stylistic choices (copied vibe from your function) ---
    markers = ["o", "s", "v", "D", "^", "P", "X", "<", ">"]
    model_to_marker = {m: markers[i % len(markers)] for i, m in enumerate(labels)}

    palette = sns.color_palette("colorblind")
    model_colors = {m: palette[i % len(palette)] for i, m in enumerate(labels)}

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex="col")
    axes = np.atleast_1d(axes).ravel()
    axA, axB, axC, axD = axes[0], axes[1], axes[2], axes[3]

    # A) Score errorbars (one point per model, styled like your panel)
    for i, m in enumerate(labels):
        mean = float(d["_score"].iloc[i])
        lo = float(d["_score_ci_low"].iloc[i])
        hi = float(d["_score_ci_high"].iloc[i])
        yerr = _safe_yerr([mean], [lo], [hi])
        axA.errorbar(
            i,
            mean,
            yerr=yerr,
            fmt=model_to_marker[m],
            color=model_colors[m],
            ecolor=model_colors[m],
            capsize=capsize,
            elinewidth=2.5,
            markersize=7,
            linewidth=0,
            alpha=0.95,
        )
    axA.set_title("A) Score (Avg) with 95% CI (Stratified)", fontsize=14)
    axA.set_xticks(x)
    axA.set_xticklabels(labels, rotation=rotate_xticks, ha="right")
    if score_ylim is not None:
        axA.set_ylim(*score_ylim)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)

    # B) Instability errorbars
    for i, m in enumerate(labels):
        mean = float(d["_instab"].iloc[i])
        lo = float(d["_instab_ci_low"].iloc[i])
        hi = float(d["_instab_ci_high"].iloc[i])
        yerr = _safe_yerr([mean], [lo], [hi])
        axB.errorbar(
            i,
            mean,
            yerr=yerr,
            fmt=model_to_marker[m],
            color=model_colors[m],
            ecolor=model_colors[m],
            capsize=capsize,
            elinewidth=2.5,
            markersize=7,
            linewidth=0,
            alpha=0.95,
        )
    axB.set_title("B) Rel. Run Instability with 95% CI (Instability)", fontsize=14)
    axB.set_xticks(x)
    axB.set_xticklabels(labels, rotation=rotate_xticks, ha="right")
    if instability_as_percent_axis:
        axB.yaxis.set_major_formatter(lambda v, pos: f"{v*100:.2f}%")
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    # C) Global noise bar plot (colored by model like your palette mapping)
    axC.bar(
        x,
        d["_global_noise"].to_numpy(),
        color=[model_colors[m] for m in labels],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.95,
    )
    axC.set_title("C) Global Noise (Z-Var)", fontsize=14)
    axC.set_xticks(x)
    axC.set_xticklabels(labels, rotation=rotate_xticks, ha="right")
    axC.spines["top"].set_visible(False)
    axC.spines["right"].set_visible(False)

    # D) Run noise bar plot
    axD.bar(
        x,
        d["_run_noise"].to_numpy(),
        color=[model_colors[m] for m in labels],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.95,
    )
    axD.set_title("D) Run Noise (Z-Var)", fontsize=14)
    axD.set_xticks(x)
    axD.set_xticklabels(labels, rotation=rotate_xticks, ha="right")
    axD.spines["top"].set_visible(False)
    axD.spines["right"].set_visible(False)

    # ---- Shared legend (patches), like your function ----
    handles = [
        Patch(facecolor=model_colors[m], edgecolor="black", label=str(m))
        for m in labels
    ]
    legend_ncol = min(len(labels), 11)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=legend_ncol,
        frameon=False,
        fontsize=legend_fontsize,
    )

    fig.supylabel("Quality", fontsize=y_label_fontsize)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)  # room for legend

    return fig, axes, d


# Example:
# fig, axes, parsed = plot_model_metrics_2x2(df)
# plt.show()
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


def _parse_ci(x) -> Tuple[float, float]:
    if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 2:
        return float(x[0]), float(x[1])
    if isinstance(x, str):
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x.strip())
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
    raise ValueError(f"Could not parse CI from value: {x!r}")


def _parse_percent(x) -> float:
    if isinstance(x, str):
        s = x.strip()
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        return float(s)
    v = float(x)
    return v / 100.0 if v > 1.0 else v


def _safe_yerr(mean, low, high) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)

    low, high = np.minimum(low, high), np.maximum(low, high)
    err_low = np.maximum(0.0, mean - low)
    err_high = np.maximum(0.0, high - mean)
    return np.vstack([err_low, err_high])


def plot_model_metrics_2x2(
    df: pd.DataFrame,
    benchmark: str,
    *,
    model_col: Optional[str] = None,
    sort_by: Optional[str] = "Score (Avg)",
    ascending: bool = False,
    capsize: int = 4,
    figsize: Tuple[float, float] = (14, 10),
    rotate_xticks: int = 25,
    score_ylim: Optional[Tuple[float, float]] = None,
    instability_as_percent_axis: bool = True,
    legend_fontsize: int = 11,
    y_label_fontsize: int = 13,
    outpath: str = None
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """
    2x2 plot:
      A) Score (Avg) with 95% CI (Stratified) errorbars
      B) Rel. Run Instability with 95% CI (Instability) errorbars
      C) Global Noise (Z-Var) bar plot
      D) Run Noise (Z-Var) bar plot

    Returns (fig, axes, parsed_df)
    """
    required = [
        "Score (Avg)",
        "95% CI (Stratified)",
        "Rel. Run Instability",
        "95% CI (Instability)",
        "Global Noise (Z-Var)",
        "Run Noise (Z-Var)",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame is missing required columns: {missing}")

    d = df.copy()

    # Model labels
    if model_col is not None:
        if model_col not in d.columns:
            raise KeyError(f"model_col={model_col!r} not found in DataFrame columns.")
        d["_model_label"] = d[model_col].astype(str)
    else:
        d["_model_label"] = d.index.astype(str)

    # Parse numeric fields
    d["_score"] = pd.to_numeric(d["Score (Avg)"], errors="raise")
    d[["_score_ci_low", "_score_ci_high"]] = d["95% CI (Stratified)"].apply(
        lambda x: pd.Series(_parse_ci(x))
    )

    d["_instab"] = d["Rel. Run Instability"].apply(_parse_percent)
    d[["_instab_ci_low", "_instab_ci_high"]] = d["95% CI (Instability)"].apply(
        lambda x: pd.Series(_parse_ci(x))
    )

    d["_global_noise"] = pd.to_numeric(d["Global Noise (Z-Var)"], errors="raise")
    d["_run_noise"] = pd.to_numeric(d["Run Noise (Z-Var)"], errors="raise")

    # Optional sort
    if sort_by is not None:
        if sort_by not in d.columns:
            raise KeyError(f"sort_by={sort_by!r} not found in DataFrame columns.")
        d = d.sort_values(sort_by, ascending=ascending)

    labels = d["_model_label"].tolist()
    n = len(d)
    x = np.arange(n)

    # --- stylistic choices (copied vibe from your function) ---
    markers = ["o", "s", "v", "D", "^", "P", "X", "<", ">"]
    model_to_marker = {m: markers[i % len(markers)] for i, m in enumerate(labels)}

    palette = sns.color_palette("colorblind")
    model_colors = {m: palette[i % len(palette)] for i, m in enumerate(labels)}

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex="col")
    axes = np.atleast_1d(axes).ravel()
    axA, axB, axC, axD = axes[0], axes[1], axes[2], axes[3]

    # A) Score errorbars (one point per model, styled like your panel)
    for i, m in enumerate(labels):
        mean = float(d["_score"].iloc[i])
        lo = float(d["_score_ci_low"].iloc[i])
        hi = float(d["_score_ci_high"].iloc[i])
        yerr = _safe_yerr([mean], [lo], [hi])
        axA.errorbar(
            i,
            mean,
            yerr=yerr,
            fmt=model_to_marker[m],
            color=model_colors[m],
            ecolor=model_colors[m],
            capsize=capsize,
            elinewidth=2.5,
            markersize=7,
            linewidth=0,
            alpha=0.95,
        )
    axA.set_title("A) Score (Avg) with 95% CI (Stratified)", fontsize=14)
    axA.set_xticks(x)
    axA.set_xticklabels(labels, rotation=rotate_xticks, ha="right")
    if score_ylim is not None:
        axA.set_ylim(*score_ylim)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)

    # B) Instability errorbars
    for i, m in enumerate(labels):
        mean = float(d["_instab"].iloc[i])
        lo = float(d["_instab_ci_low"].iloc[i])
        hi = float(d["_instab_ci_high"].iloc[i])
        yerr = _safe_yerr([mean], [lo], [hi])
        axB.errorbar(
            i,
            mean,
            yerr=yerr,
            fmt=model_to_marker[m],
            color=model_colors[m],
            ecolor=model_colors[m],
            capsize=capsize,
            elinewidth=2.5,
            markersize=7,
            linewidth=0,
            alpha=0.95,
        )
    axB.set_title("B) Rel. Run Instability with 95% CI (Instability)", fontsize=14)
    axB.set_xticks(x)
    axB.set_xticklabels(labels, rotation=rotate_xticks, ha="right")
    if instability_as_percent_axis:
        axB.yaxis.set_major_formatter(lambda v, pos: f"{v*100:.2f}%")
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    # C) Global noise bar plot (colored by model like your palette mapping)
    axC.bar(
        x,
        d["_global_noise"].to_numpy(),
        color=[model_colors[m] for m in labels],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.95,
    )
    axC.set_title("C) Global Noise (Z-Var)", fontsize=14)
    axC.set_xticks(x)
    #axC.set_xticklabels(labels, rotation=rotate_xticks, ha="right")
    axC.set_xticklabels([])
    axC.spines["top"].set_visible(False)
    axC.spines["right"].set_visible(False)

    # D) Run noise bar plot
    axD.bar(
        x,
        d["_run_noise"].to_numpy(),
        color=[model_colors[m] for m in labels],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.95,
    )
    axD.set_title("D) Run Noise (Z-Var)", fontsize=14)
    axD.set_xticks(x)
    #axD.set_xticklabels(labels, rotation=rotate_xticks, ha="right")
    axD.set_xticklabels([])

    axD.spines["top"].set_visible(False)
    axD.spines["right"].set_visible(False)

    # ---- Shared legend (patches), like your function ----
    handles = [
        Patch(facecolor=model_colors[m], edgecolor="black", label=str(m))
        for m in labels
    ]
    legend_ncol = min(len(labels), 11)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=legend_fontsize,
    )

    #fig.supylabel("Quality", fontsize=y_label_fontsize)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)  # room for legend

    fig.suptitle(f"{BENCHMARK_MAP[benchmark]}", fontsize=20)
    if outpath:
        os.makedirs(outpath, exist_ok=True)
        plt.savefig(f"{outpath}/{benchmark}.pdf", format='pdf', bbox_inches='tight', dpi=300)  

    return fig, axes, d


def plot_model_cost_metrics_2x2(
    df: pd.DataFrame,
    benchmark: str,
    *,
    model_col: Optional[str] = None,
    sort_by: Optional[str] = "Cost (Avg)",
    ascending: bool = False,
    capsize: int = 4,
    figsize: Tuple[float, float] = (14, 10),
    rotate_xticks: int = 25,
    cost_ylim: Optional[Tuple[float, float]] = None,
    instability_as_percent_axis: bool = True,
    legend_fontsize: int = 11,
    y_label_fontsize: int = 13,
    outpath: str = None,
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """
    EXACTLY the same styling/structure as plot_model_metrics_2x2, but:
      - uses 'Cost (Avg)' instead of 'Score (Avg)'
      - uses the same CI / instability / noise columns as provided

    Returns (fig, axes, parsed_df)
    """
    required = [
        "Cost (Avg)",
        "95% CI (Stratified)",
        "Rel. Run Instability",
        "95% CI (Instability)",
        "Global Noise (Z-Var)",
        "Run Noise (Z-Var)",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame is missing required columns: {missing}")

    d = df.copy()

    # Model labels
    if model_col is not None:
        if model_col not in d.columns:
            raise KeyError(f"model_col={model_col!r} not found in DataFrame columns.")
        d["_model_label"] = d[model_col].astype(str)
    else:
        d["_model_label"] = d.index.astype(str)

    # Parse numeric fields
    d["_cost"] = pd.to_numeric(d["Cost (Avg)"], errors="raise")
    d[["_cost_ci_low", "_cost_ci_high"]] = d["95% CI (Stratified)"].apply(
        lambda x: pd.Series(_parse_ci(x))
    )

    d["_instab"] = d["Rel. Run Instability"].apply(_parse_percent)
    d[["_instab_ci_low", "_instab_ci_high"]] = d["95% CI (Instability)"].apply(
        lambda x: pd.Series(_parse_ci(x))
    )

    d["_global_noise"] = pd.to_numeric(d["Global Noise (Z-Var)"], errors="raise")
    d["_run_noise"] = pd.to_numeric(d["Run Noise (Z-Var)"], errors="raise")

    # Optional sort
    if sort_by is not None:
        if sort_by not in d.columns:
            raise KeyError(f"sort_by={sort_by!r} not found in DataFrame columns.")
        d = d.sort_values(sort_by, ascending=ascending)

    labels = d["_model_label"].tolist()
    n = len(d)
    x = np.arange(n)

    # --- stylistic choices (same as your reference function) ---
    markers = ["o", "s", "v", "D", "^", "P", "X", "<", ">"]
    model_to_marker = {m: markers[i % len(markers)] for i, m in enumerate(labels)}

    palette = sns.color_palette("colorblind")
    model_colors = {m: palette[i % len(palette)] for i, m in enumerate(labels)}

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex="col")
    axes = np.atleast_1d(axes).ravel()
    axA, axB, axC, axD = axes[0], axes[1], axes[2], axes[3]

    # A) Cost errorbars
    for i, m in enumerate(labels):
        mean = float(d["_cost"].iloc[i])
        lo = float(d["_cost_ci_low"].iloc[i])
        hi = float(d["_cost_ci_high"].iloc[i])
        yerr = _safe_yerr([mean], [lo], [hi])
        axA.errorbar(
            i,
            mean,
            yerr=yerr,
            fmt=model_to_marker[m],
            color=model_colors[m],
            ecolor=model_colors[m],
            capsize=capsize,
            elinewidth=2.5,
            markersize=7,
            linewidth=0,
            alpha=0.95,
        )
    axA.set_title("A) Cost (Avg) with 95% CI (Stratified)", fontsize=14)
    axA.set_xticks(x)
    axA.set_xticklabels(labels, rotation=rotate_xticks, ha="right")
    if cost_ylim is not None:
        axA.set_ylim(*cost_ylim)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)

    # B) Instability errorbars
    for i, m in enumerate(labels):
        mean = float(d["_instab"].iloc[i])
        lo = float(d["_instab_ci_low"].iloc[i])
        hi = float(d["_instab_ci_high"].iloc[i])
        yerr = _safe_yerr([mean], [lo], [hi])
        axB.errorbar(
            i,
            mean,
            yerr=yerr,
            fmt=model_to_marker[m],
            color=model_colors[m],
            ecolor=model_colors[m],
            capsize=capsize,
            elinewidth=2.5,
            markersize=7,
            linewidth=0,
            alpha=0.95,
        )
    axB.set_title("B) Rel. Run Instability with 95% CI (Instability)", fontsize=14)
    axB.set_xticks(x)
    axB.set_xticklabels(labels, rotation=rotate_xticks, ha="right")
    if instability_as_percent_axis:
        axB.yaxis.set_major_formatter(lambda v, pos: f"{v*100:.2f}%")
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    # C) Global noise bars
    axC.bar(
        x,
        d["_global_noise"].to_numpy(),
        color=[model_colors[m] for m in labels],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.95,
    )
    axC.set_title("C) Global Noise (Z-Var)", fontsize=14)
    axC.set_xticks(x)
    # axC.set_xticklabels(labels, rotation=rotate_xticks, ha="right")
    axC.set_xticklabels([])
    axC.spines["top"].set_visible(False)
    axC.spines["right"].set_visible(False)

    # D) Run noise bars
    axD.bar(
        x,
        d["_run_noise"].to_numpy(),
        color=[model_colors[m] for m in labels],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.95,
    )
    axD.set_title("D) Run Noise (Z-Var)", fontsize=14)
    axD.set_xticks(x)
    # axD.set_xticklabels(labels, rotation=rotate_xticks, ha="right")
    axD.set_xticklabels([])
    axD.spines["top"].set_visible(False)
    axD.spines["right"].set_visible(False)

    # ---- Shared legend ----
    handles = [
        Patch(facecolor=model_colors[m], edgecolor="black", label=str(m))
        for m in labels
    ]
    legend_ncol = min(len(labels), 11)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=legend_fontsize,
    )

    fig.supylabel("Quality", fontsize=y_label_fontsize)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)

    fig.suptitle(f"{BENCHMARK_MAP[benchmark]}", fontsize=20)
    if outpath:
        os.makedirs(outpath, exist_ok=True)
        plt.savefig(f"{outpath}/{benchmark}.pdf", format='pdf', bbox_inches='tight', dpi=300)  

    return fig, axes, d


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

def plot_convergence(d, savepath="convergence.pdf", benchmark=None):
    # Colorblind-friendly palette (IBM design)
    C_HIGH = "#009E73"   # teal-green  (HighCI)
    C_MEAN = "#FE6100"   # orange      (Mean)
    C_LOW  = "#648FFF"   # blue        (LowCI)

    BG      = "#FAFAFA"
    GRID    = "#E4E4E4"
    TICK_C  = "#444444"
    TITLE_C = "#1A1A2E"

    FONT_TITLE = dict(fontsize=9, fontweight="bold", color=TITLE_C, fontfamily="DejaVu Sans")
    FONT_AXIS  = dict(fontsize=10, color=TICK_C, fontfamily="DejaVu Sans")

    LW         = 1.6
    ALPHA_FILL = 0.12

    keys = list(d.keys())
    n    = len(keys)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(10, 10), facecolor="white")
    gs  = GridSpec(nrows, ncols, figure=fig, hspace=0.50, wspace=0.38,
                   left=0.07, right=0.97, top=0.95, bottom=0.07)

    for i, key in enumerate(keys):
        ax    = fig.add_subplot(gs[i // ncols, i % ncols])
        inner = d[key]

        x  = list(inner.keys())
        y_high = [v[0] for v in inner.values()]
        y_mean = [v[1]*100 for v in inner.values()]
        y_low  = [v[2] for v in inner.values()]

        ax.set_facecolor(BG)

        # Shaded bands between LowCI↔Mean and Mean↔HighCI
        ax.fill_between(x, y_low, y_mean, alpha=ALPHA_FILL, color=C_LOW,  linewidth=0)
        ax.fill_between(x, y_mean, y_high, alpha=ALPHA_FILL, color=C_HIGH, linewidth=0)

        # Lines
        l1, = ax.plot(x, y_high, color=C_HIGH, lw=LW, zorder=3, label="HighCI")
        l2, = ax.plot(x, y_mean, color=C_MEAN, lw=LW, zorder=3, label="Mean")
        l3, = ax.plot(x, y_low,  color=C_LOW,  lw=LW, zorder=3, label="LowCI")

        # Grid
        ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

        # Spines
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#CCCCCC")
            ax.spines[spine].set_linewidth(0.8)

        # Ticks
        ax.tick_params(axis="both", which="major", labelsize=7,
                       colors=TICK_C, length=3, width=0.7)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        # Red vertical line at iteration 10
        ax.axvline(x=10, color="#E3170A", linewidth=1.2, linestyle="--", zorder=4)

        # Title and axis labels
        ax.set_title(key, **FONT_TITLE, pad=5)
        if i % ncols == 0:
            ax.set_ylabel("Score", **FONT_AXIS, labelpad=3)
        if i // ncols == nrows - 1:
            ax.set_xlabel("Iteration", **FONT_AXIS, labelpad=3)

        # Single shared legend in the first panel only
        if i == 0:
            ax.legend(handles=[l1, l2, l3],
                      frameon=True, framealpha=0.9,
                      edgecolor="#DDDDDD", facecolor="white",
                      prop={"size": 7, "family": "DejaVu Sans"},
                      loc="best", handlelength=1.6,
                      handletextpad=0.5, borderpad=0.6)

    # Hide unused subplots
    for j in range(n, nrows * ncols):
        fig.add_subplot(gs[j // ncols, j % ncols]).set_visible(False)
    if benchmark:
        plt.suptitle(f"{BENCHMARK_MAP[benchmark]}")

    plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_scores_with_ci(df, savepath):
    """
    Two-panel subplot:
      - Left:  Score (Avg) with 95% CI (Stratified) as box-style plot
      - Right: Rel. Run Instability with 95% CI (Instability) as box-style plot
    Expects MultiIndex columns with:
      'Score (Avg)', '95% CI (Stratified)',
      'Rel. Run Instability', '95% CI (Instability)'
    """
    strategies = df.index.tolist()
    temps = df['Score (Avg)'].columns.tolist()

    n_strategies = len(strategies)
    x = np.arange(n_strategies)
    width = 0.22
    colors = ['#4C72B0', '#DD8452', '#55A868']

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    def parse_ci(ci_series):
        return [
            tuple(float(v) for v in s.strip('[]').split(','))
            for s in ci_series
        ]

    def parse_pct(pct_series):
        return [float(str(v).strip('%')) / 100 for v in pct_series]

    def draw_ci_boxes(ax, means, ci_parsed, positions, color, width):
        for j, pos in enumerate(positions):
            lo, hi = ci_parsed[j]
            mean = means[j]
            ax.bar(
                pos, hi - lo, bottom=lo,
                width=width * 0.9, color=color, alpha=0.5,
                linewidth=1.2, edgecolor=color
            )
            ax.plot(
                [pos - width * 0.45, pos + width * 0.45],
                [mean, mean],
                color=color, linewidth=2.5
            )

    for i, (temp, color) in enumerate(zip(temps, colors)):
        positions = x + (i - 1) * width

        # --- Left: Score ---
        means_score = df['Score (Avg)'][temp].values
        ci_score = parse_ci(df['95% CI (Stratified)'][temp].values)
        draw_ci_boxes(axes[0], means_score, ci_score, positions, color, width)

        # --- Right: Instability ---
        means_inst = parse_pct(df['Rel. Run Instability'][temp].values)
        ci_inst = parse_ci(df['95% CI (Instability)'][temp].values)
        draw_ci_boxes(axes[1], means_inst, ci_inst, positions, color, width)

        # Legend entries
        for ax in axes:
            ax.bar(0, 0, color=color, alpha=0.5, edgecolor=color, label=temp)

    # --- Formatting ---
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, fontsize=12)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        ax.legend(title='Temperature', fontsize=10)

    axes[0].set_ylabel('Score (Avg)', fontsize=12)
    axes[0].set_title('Score by Strategy and Temperature (95% CI)', fontsize=13)

    axes[1].set_ylabel('Rel. Run Instability', fontsize=12)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    axes[1].set_title('Run Instability by Strategy and Temperature (95% CI)', fontsize=13)

    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def convergence_cis(df, mode, col, fn=np.mean):
    cis={}
    result_df = pd.DataFrame()
    i = 0
    while df.shape[0] > 0:
        i += 1
        # Sample 1 row per unique (mode, Benchmark) combination --> mode is either Strategy or Model
        sampled_df = df.groupby([mode, 'Benchmark']).sample(n=1)

        # Remove sampled rows from the original dataframe
        df = df.drop(sampled_df.index)

        # Reset indices 
        sampled_df = sampled_df.reset_index(drop=True)
        df = df.reset_index(drop=True)
        
        # Append sampled_df to result_df
        result_df = pd.concat([result_df, sampled_df], ignore_index=True)

        # Run Stratified Bootstrap
        strat_ci_df = stratified_bootstrap_ci(result_df, mode=mode ,col=col, fn=fn)

        # --- Compile Tables ---
        if fn == np.mean:
            raw_means = result_df.groupby(mode)[col].mean()
        elif fn == np.var:
            raw_means = result_df.groupby(mode)[col].var()
        else:
            raise ValueError(f"Unsupported function: {fn}. Only np.mean and np.var are allowed.")
        sorted_strategies = raw_means.sort_values(ascending=False).index

        # Table 1: Performance
        table1 = pd.DataFrame({
            f"{col} (Avg)": raw_means,
            "95% CI (Stratified)": strat_ci_df["Strat_CI_Formatted"]
        }).reindex(sorted_strategies)

        # Parse the table
        t = table1.copy()
        t["LowCI"] = table1["95% CI (Stratified)"].apply(lambda x: eval(x)[0])
        t["HighCI"] = table1["95% CI (Stratified)"].apply(lambda x: eval(x)[1])
        t.reset_index(inplace=True)

        # Saving results in the dectionary
        for index, row in t.iterrows():
            if row[mode] in cis:
                cis[row[mode]].update({i: (row["LowCI"], row[f"{col} (Avg)"], row["HighCI"])})
            else:
                cis[row[mode]] = {}
                cis[row[mode]].update({i: (row["LowCI"], row[f"{col} (Avg)"], row["HighCI"])})

    return cis