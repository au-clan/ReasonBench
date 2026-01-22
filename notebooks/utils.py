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

    log = eval_logs(result)
    return log

def get_logs(
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

        methods = [log["General information"]["Method"] for log in logs]
        count = Counter(methods)
        
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
    out_path = os.path.join(data_path, f"{experiment}/{model}/{now.strftime("%d")}/{now.strftime("%H:%M")}.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save dataframe to parquet
    df.to_parquet(out_path)
    df.to_parquet(os.path.join(data_path, f"{experiment}/{model}/latest.parquet"))

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