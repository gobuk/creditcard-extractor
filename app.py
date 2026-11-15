"""Credit card debtor extractor - Flask web app.

Upload a credit card CSV export and get:
  * the cleaned rows (selected columns, tidy "Debtor A/c" names)
  * a per-debtor summary (row count and total Doc Amt)
  * a filter by debtor (e.g. AGODA.COM)
  * a download of the cleaned CSV (all rows, or just the filtered debtor)

Run:  python app.py   ->  http://127.0.0.1:5000
"""
from __future__ import annotations

import io
import re
import tempfile
import time
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # reject uploads over 10 MB

# --- settings ---------------------------------------------------------------
ENCODING = "windows-1252"
HEADER_MARKER = "Trx Date"     # text used to locate the real header row (replaces hardcoded skiprows=8)
COLUMNS = [
    "Debtor Type", "Debtor A/c", "Trx Date", "Booking No", "Folio No",
    "Guest Name", "Ref No", "Doc Amt", "Mode", "Alloc Amt",
]
MONEY_COLUMNS = ["Doc Amt", "Alloc Amt"]
MAX_ROWS_SHOWN = 500
RESULT_MAX_AGE = 60 * 60       # processed files are deleted after 1 hour

STORE = Path(tempfile.gettempdir()) / "creditcard_extractor"
STORE.mkdir(exist_ok=True)
TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")


# --- data processing --------------------------------------------------------
def find_header_row(text: str) -> int:
    """Index of the line containing the real column headers."""
    for i, line in enumerate(text.splitlines()[:60]):
        if HEADER_MARKER in line:
            return i
    raise ValueError(f"Could not find the header row (a line containing '{HEADER_MARKER}').")


def process(raw: bytes) -> pd.DataFrame:
    """Turn the raw uploaded bytes into the cleaned DataFrame."""
    text = raw.decode(ENCODING, errors="replace")
    skip = find_header_row(text)
    try:
        df = pd.read_csv(io.StringIO(text), skiprows=skip, thousands=",")
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Could not read this file as a CSV ({exc}).") from exc

    df.columns = df.columns.astype(str).str.strip()
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("The file is missing expected columns: " + ", ".join(missing))

    df["Trx Date"] = pd.to_datetime(df["Trx Date"], dayfirst=True, errors="coerce")
    df = df.loc[df["Trx Date"].notna(), COLUMNS].copy()   # drops totals/footer rows
    if df.empty:
        raise ValueError("No rows with a valid 'Trx Date' were found.")

    for col in MONEY_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # "Debtor A/c" looks like "<code> <NAME>"; keep the name (2nd word).
    # Falls back to the whole value if there is only one word.
    parts = df["Debtor A/c"].astype("string").str.split()
    df["Debtor A/c"] = parts.str[1].fillna(parts.str[0]).fillna("UNKNOWN")
    return df.reset_index(drop=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Debtor A/c")["Doc Amt"]
        .agg(Transactions="size", **{"Total Doc Amt": "sum"})
        .reset_index()
        .sort_values("Total Doc Amt", ascending=False)
    )


# --- storage of processed results ------------------------------------------
def purge_old() -> None:
    cutoff = time.time() - RESULT_MAX_AGE
    for p in STORE.glob("*.csv"):
        if p.stat().st_mtime < cutoff:
            p.unlink(missing_ok=True)


def save_result(df: pd.DataFrame, original_name: str) -> str:
    purge_old()
    token = uuid.uuid4().hex
    stem = secure_filename(Path(original_name).stem) or "creditcard"
    df.to_csv(STORE / f"{token}--{stem}.csv", index=False, date_format="%Y-%m-%d")
    return token


def load_result(token: str) -> tuple[pd.DataFrame, str]:
    if not TOKEN_RE.match(token):
        abort(404)
    matches = list(STORE.glob(f"{token}--*.csv"))
    if not matches:
        abort(404)
    path = matches[0]
    stem = path.stem.split("--", 1)[1]
    return pd.read_csv(path, parse_dates=["Trx Date"]), stem


# --- presentation helpers ---------------------------------------------------
def money(x) -> str:
    return "" if pd.isna(x) else f"{x:,.2f}"


def to_html(df: pd.DataFrame, **kwargs) -> str:
    return df.to_html(index=False, border=0, classes="tbl", na_rep="", **kwargs)


# --- routes -----------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", error=None, result=None)

    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return render_template("index.html", error="Please choose a CSV file.", result=None), 400
    if not upload.filename.lower().endswith(".csv"):
        return render_template("index.html", error="Only .csv files are supported.", result=None), 400

    try:
        df = process(upload.read())
    except ValueError as exc:
        return render_template("index.html", error=str(exc), result=None), 400

    return redirect(url_for("show_result", token=save_result(df, upload.filename)))


@app.get("/result/<token>")
def show_result(token: str):
    df, _ = load_result(token)
    summary = summarise(df)
    debtor_list = sorted(df["Debtor A/c"].unique())

    selected = request.args.get("debtor", "")
    if selected not in debtor_list:
        selected = ""
    view = df[df["Debtor A/c"] == selected] if selected else df
    shown = view.head(MAX_ROWS_SHOWN)

    result = {
        "token": token,
        "rows": f"{len(df):,}",
        "debtors": len(debtor_list),
        "total": money(df["Doc Amt"].sum()),
        "summary_html": to_html(
            summary,
            formatters={"Total Doc Amt": money},
        ),
        "debtor_list": debtor_list,
        "selected": selected,
        "table_html": to_html(
            shown,
            formatters={
                "Trx Date": lambda d: d.strftime("%d/%m/%Y"),
                "Doc Amt": money,
                "Alloc Amt": money,
            },
        ),
        "filtered_rows": f"{len(view):,}",
        "shown": len(shown),
        "truncated": len(view) > MAX_ROWS_SHOWN,
    }
    return render_template("index.html", error=None, result=result)


@app.get("/download/<token>")
def download(token: str):
    df, stem = load_result(token)
    debtor = request.args.get("debtor", "")
    if debtor and debtor in set(df["Debtor A/c"]):
        df = df[df["Debtor A/c"] == debtor]
        stem = f"{stem}_{secure_filename(debtor)}"

    buf = io.BytesIO(df.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8"))
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name=f"{stem}_final.csv")


@app.errorhandler(413)
def too_large(_):
    return render_template("index.html", error="That file is too large (10 MB limit).", result=None), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
