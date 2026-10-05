"""Load memprint_trace CSV files into one table per workload.

Trace file names (see pintool/memprint_trace.cpp):

    <Prefix>_<name>_<interval>_<pid>[_<args>][_SubSample_<binInterval>_bin_<bin>].csv

Prefix is Buffered (splitter) or Sampled (sampler). <name> is
"<workload>-<config>" (set with -name), or, for traces recorded without -name,
the binary name with the program arguments in <args> (e.g. "_-n_1024").
"""

import os
import re

import pandas as pd

TRACE_NAME = re.compile(r"^(?P<prefix>Buffered|Sampled)_(?P<name>.+?)_(?P<interval>\d+)_(?P<pid>\d+)(?P<args>_.*)?$")
BIN_SUFFIX = re.compile(r"_SubSample_(?P<bin_interval>\d+)_bin_(?P<bin>\d+)$")


def parse_trace_name(filename):
    """Return (workload, config, sampling interval, sample id) for a trace file
    name, or None if it is not a trace file."""
    if not filename.endswith(".csv"):
        return None
    stem = filename[: -len(".csv")]
    bin_match = BIN_SUFFIX.search(stem)
    if bin_match:
        stem = stem[: bin_match.start()]
    match = TRACE_NAME.match(stem)
    if not match:
        return None
    info = match.groupdict()
    if info["args"]:
        # Recorded without -name: the program arguments are the config.
        workload = info["name"]
        config = re.sub(r"^_(-n_)?", "", info["args"])
    else:
        workload, _, config = info["name"].rpartition("-")
    if bin_match:
        return workload, config, int(bin_match["bin_interval"]), f"{info['pid']}_{bin_match['bin']}"
    return workload, config, int(info["interval"]), info["pid"]


def load_traces(directory, prefix="Buffered"):
    """Read every <prefix>_* trace CSV in a directory.

    Returns the trace rows with PID (sample id: pid, or pid_bin for a bin) and
    Config columns added. Splitter traces (prefix Buffered) are the training data.
    """
    frames = []
    for filename in sorted(os.listdir(directory)):
        if not filename.startswith(prefix + "_"):
            continue
        parsed = parse_trace_name(filename)
        if parsed is None:
            print(f"skipping {filename}: unrecognised trace name")
            continue
        _, config, _, sample = parsed
        frame = pd.read_csv(os.path.join(directory, filename))
        frame["PID"] = sample
        frame["Config"] = config
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no {prefix}_*.csv traces in {directory}")
    return pd.concat(frames, ignore_index=True)


def add_sample_spread(traces):
    """Add the standard deviation across samples (bins) of each statistic for
    every (FunctionName, Config, SamplingInterval) group."""
    keys = ["FunctionName", "Config", "SamplingInterval"]
    spread = (
        traces.groupby(keys)
        .agg(
            SD_Unique=("UniqueAddresses", "std"),
            SD_MemUsage=("MemUsageObs", "std"),
            SD_CountObs=("CountObs", "std"),
        )
        .reset_index()
    )
    merged = traces.merge(spread, on=keys, how="left")
    return merged.sort_values(keys, kind="stable").reset_index(drop=True)
