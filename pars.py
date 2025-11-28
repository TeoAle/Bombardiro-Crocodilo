import pandas as pd
from typing import Iterable, List, Dict


# ---------- date helper ----------

def excel_serial_to_datetime(serial: float, system: str = "1900") -> pd.Timestamp:
    """
    Convert Excel-style serial date number to a pandas Timestamp.

    system: "1900" (default) or "1904" – if you want to branch on the date_system
    column later, you can pass it in here.
    """
    if system.startswith("1904"):
        origin = "1904-01-01"
    else:  # Excel 1900 system
        # Excel's "day 1" is 1899-12-31 but there is a leap-year bug;
        # pandas uses 1899-12-30 as origin for compatibility.
        origin = "1899-12-30"

    return pd.to_datetime(serial, unit="D", origin=origin)


# ---------- core parser (line iterator) ----------

def parse_dump_memstored_stream(lines: Iterable[str]) -> pd.DataFrame:
    """
    Parse a dump_memstored_PX_LAST.csv-like stream where, for each product,
    two consecutive lines are:

        line_dates: ts_info, date_system, source, code, mode, d1, d2, ...
        line_values: ts_info, date_system, source, code, mode, v1, v2, ...

    Returns a tidy DataFrame with columns:
        ['code', 'date', 'close', 'source', 'mode']
    """
    it = iter(lines)

    # Read & discard header
    header = next(it).strip().split(",")
    # Optional sanity check:
    # print("Header:", header)

    records: List[Dict] = []

    for line_dates, line_vals in zip(it, it):  # consume 2 lines at a time
        line_dates = line_dates.rstrip("\n")
        line_vals = line_vals.rstrip("\n")

        if not line_dates or not line_vals:
            continue

        parts_dates = line_dates.split(",")
        parts_vals = line_vals.split(",")

        # First 5 fields are metadata
        ts_info_d, date_system_d, source_d, code_d, mode_d = parts_dates[:5]
        ts_info_v, date_system_v, source_v, code_v, mode_v = parts_vals[:5]

        # Sanity checks
        if code_d != code_v:
            raise ValueError(f"Code mismatch between pair: {code_d} vs {code_v}")
        if source_d != source_v or mode_d != mode_v:
            raise ValueError(f"Meta mismatch for code {code_d}")

        code = code_d
        source = source_d
        mode = mode_d
        date_system = date_system_d

        date_tokens = parts_dates[5:]
        value_tokens = parts_vals[5:]

        if len(date_tokens) != len(value_tokens):
            raise ValueError(
                f"Length mismatch for {code}: {len(date_tokens)} dates vs {len(value_tokens)} values"
            )

        for d_str, v_str in zip(date_tokens, value_tokens):
            d_str = d_str.strip()
            v_str = v_str.strip()

            if not d_str or not v_str:
                continue

            # Convert serial date + price
            try:
                serial = float(d_str)
                dt = excel_serial_to_datetime(serial, date_system)
            except ValueError:
                # If the "date" is garbage, skip
                continue

            try:
                px = float(v_str)
            except ValueError:
                # Skip non-numeric prices
                continue

            records.append(
                {
                    "code": code,
                    "date": dt,
                    "close": px,
                    "source": source,
                    "mode": mode,
                }
            )

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df.sort_values(["code", "date"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


# ---------- convenience wrapper for a local file ----------

def parse_dump_memstored_file(path: str) -> pd.DataFrame:
    """
    Parse a local dump_memstored_PX_LAST.csv file into a tidy DataFrame.
    """
    with open(path, "r", encoding="utf-8") as f:
        return parse_dump_memstored_stream(f)


# ---------- example usage ----------

if __name__ == "__main__":
    # Local version – for S3, just pass a TextIOWrapper / iter_lines() to parse_dump_memstored_stream
    df = parse_dump_memstored_file("dump_memstored_PX_LAST.csv")

    # Optionally rename to match your Hawk format
    df = df.rename(
        columns={
            "code": "ric",             # or 'bbg_ticker' or whatever
            "close": "px_last_adj"
        }
    )

    # Save in a clean format
    df.to_parquet("dump_memstored_PX_LAST_clean.parquet")
    # or: df.to_csv("dump_memstored_PX_LAST_clean.csv", index=False)
