import os
import sqlite3
import time
from datetime import datetime, timedelta
import pandas as pd
import requests
import yfinance as yf
from pytrends.request import TrendReq

# ==========================================
# CONFIGURATION & TARGETS
# ==========================================
SEC_USER_AGENT = "QuantResearcher research@example.com"
COMPANY_TICKERS = ["ASTS", "PLTR", "RDDT"]
LOOKBACK_DAYS = 30
DB_NAME = "alt_data_intelligence.db"

# Federal contractor agency/recipient mapping for USAspending API
FEDERAL_ENTITY_MAP = {
    "PLTR": {"recipient_name": "PALANTIR TECHNOLOGIES INC."},
    "ASTS": {"recipient_name": "AST SPACEBUILDS"}
}


def init_database():
    """Initializes a local SQLite database to persist pipeline runs and correlations."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS correlations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            filing_date TEXT,
            spike_date TEXT,
            offset_days INTEGER,
            relationship TEXT,
            search_interest INTEGER,
            accession TEXT,
            price_return_pct REAL,
            timestamp TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS federal_contracts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            award_id TEXT,
            amount REAL,
            action_date TEXT,
            description TEXT,
            timestamp TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contract_correlations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            contract_date TEXT,
            spike_date TEXT,
            lag_days INTEGER,
            contract_amount REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()


def get_cik_for_ticker(ticker, headers):
    """Dynamically fetches the official SEC CIK mapping file."""
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for entry in data.values():
                if entry['ticker'].upper() == ticker.upper():
                    return str(entry['cik_str']).zfill(10)
    except Exception as e:
        print(f"[!] Error fetching SEC ticker map: {e}")
    return None


def fetch_structured_sec_filings(tickers):
    print("[*] Fetching structured SEC EDGAR metadata via JSON API...")
    headers = {"User-Agent": SEC_USER_AGENT}
    filing_records = []

    for ticker in tickers:
        cik = get_cik_for_ticker(ticker, headers)
        if not cik:
            print(f"[!] Warning: No CIK mapping found for {ticker}")
            continue

        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        print(f"[-] Querying SEC feed for {ticker} (CIK: {cik})...")

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                recent = data['filings']['recent']

                df_filings = pd.DataFrame({
                    'ticker': ticker,
                    'accessionNumber': recent['accessionNumber'],
                    'filingDate': recent['filingDate'],
                    'form': recent['form']
                })

                df_8k = df_filings[df_filings['form'] == '8-K'].head(5)
                filing_records.append(df_8k)
            else:
                print(f"[!] SEC API returned status code {response.status_code} for {ticker}")
        except Exception as e:
            print(f"[!] Error querying SEC API for {ticker}: {e}")

        time.sleep(0.2)

    if filing_records:
        return pd.concat(filing_records, ignore_index=True)
    return pd.DataFrame()


def analyze_search_velocity(tickers):
    print("\n[*] Connecting to Google Trends API (PyTrends)...")
    pytrends = TrendReq(hl='en-US', tz=360)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    timeframe_str = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

    print(f"[-] Pulling search interest metrics for timeframe: {timeframe_str}")
    pytrends.build_payload(tickers, cat=0, timeframe=timeframe_str, geo='US')

    interest_df = pytrends.interest_over_time()

    if not interest_df.empty:
        if 'isPartial' in interest_df.columns:
            interest_df = interest_df.drop(columns=['isPartial'])
        return interest_df
    else:
        print("[!] Warning: Empty response returned from Google Trends.")
        return pd.DataFrame()


def detect_search_anomalies(trend_df):
    print("\n[+] Running enhanced anomaly detection (Z-Score + Velocity Filter)...")
    anomalies = {}

    for column in trend_df.columns:
        series = trend_df[column]
        rolling_mean = series.rolling(window=7, min_periods=1).mean()
        rolling_std = series.rolling(window=7, min_periods=1).std().fillna(1)

        z_scores = (series - rolling_mean) / rolling_std
        daily_velocity = series.diff().fillna(0)

        spike_mask = (z_scores > 2.0) & (daily_velocity > 15)
        spike_dates = trend_df.index[spike_mask].tolist()

        if spike_dates:
            anomalies[column] = {
                "dates": spike_dates,
                "values": series[spike_mask].to_dict(),
                "velocities": daily_velocity[spike_mask].to_dict()
            }

    return anomalies


def fetch_federal_contract_awards(tickers):
    """Queries USAspending.gov API for federal contract obligations with robust field keys."""
    print("\n[*] Querying USAspending.gov Federal Contract API...")
    contract_records = []
    api_url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    for ticker in tickers:
        entity_info = FEDERAL_ENTITY_MAP.get(ticker)
        if not entity_info:
            continue

        payload = {
            "filters": {
                "time_period": [
                    {"start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
                ],
                "recipient_search_text": [entity_info["recipient_name"]],
                "award_type_codes": ["A", "B", "C", "D"]
            },
            "fields": ["Award ID", "Recipient Name", "Award Amount", "Action Date", "Description"],
            "pagination": {"limit": 10, "page": 1, "sort": "Action Date", "order": "desc"}
        }

        try:
            response = requests.post(api_url, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])

                for award in results:
                    amount = float(award.get("Award Amount") or award.get("total_obligation") or 0.0)
                    action_date = award.get("Action Date") or award.get("action_date")
                    award_id = award.get("Award ID") or award.get("generated_unique_award_id")
                    desc = award.get("Description") or award.get("description") or "No description provided"

                    contract_records.append({
                        "ticker": ticker,
                        "awardId": award_id,
                        "amount": amount,
                        "actionDate": action_date,
                        "description": desc
                    })

                    cursor.execute('''
                        INSERT INTO federal_contracts (ticker, award_id, amount, action_date, description, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (ticker, str(award_id), amount, str(action_date), str(desc), datetime.now().isoformat()))

                print(f"[-] Fetched {len(results)} federal contract records for {ticker}")
            else:
                print(f"[!] USAspending API returned status code {response.status_code} for {ticker}")
        except Exception as e:
            print(f"[!] Error querying USAspending API for {ticker}: {e}")

    conn.commit()
    conn.close()
    return pd.DataFrame(contract_records)


def fetch_market_reaction(ticker, target_date):
    """Pulls daily stock price return around an event window safely."""
    try:
        dt = pd.to_datetime(target_date)
        start = (dt - timedelta(days=2)).strftime('%Y-%m-%d')
        end = (dt + timedelta(days=3)).strftime('%Y-%m-%d')

        df = yf.download(ticker, start=start, end=end, progress=False)
        if not df.empty and len(df) >= 2:
            close_series = df['Close'].squeeze()
            start_price = float(close_series.iloc[0])
            end_price = float(close_series.iloc[-1])
            pct_return = ((end_price - start_price) / start_price) * 100
            return round(pct_return, 2)
    except Exception as e:
        print(f"[!] Error fetching market data for {ticker}: {e}")
    return 0.0


def correlate_filings_with_search(df_filings, trend_df, detected_anomalies):
    print("\n[+] Cross-referencing SEC filings with Search Velocity anomalies & Market Price...")

    if df_filings.empty or not detected_anomalies:
        return pd.DataFrame()

    df_filings['filingDate'] = pd.to_datetime(df_filings['filingDate'])
    trend_df.index = pd.to_datetime(trend_df.index)

    correlation_results = []
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    for ticker in df_filings['ticker'].unique():
        ticker_filings = df_filings[df_filings['ticker'] == ticker]
        spike_data = detected_anomalies.get(ticker, {})
        spike_dates = spike_data.get("dates", [])
        spike_values = spike_data.get("values", {})

        for _, filing in ticker_filings.iterrows():
            filing_date = filing['filingDate']
            accession = filing['accessionNumber']

            for spike_date in spike_dates:
                delta_days = (spike_date - filing_date).days

                if -3 <= delta_days <= 3:
                    if delta_days < 0:
                        relation = "Leading (Potential Leak/Rumor)"
                    elif delta_days == 0:
                        relation = "Coincident (Same-Day Reaction)"
                    else:
                        relation = "Lagging (Post-Filing Momentum)"

                    price_return = fetch_market_reaction(ticker, spike_date)

                    result_record = {
                        "ticker": ticker,
                        "filingDate": filing_date.strftime('%Y-%m-%d'),
                        "spikeDate": spike_date.strftime('%Y-%m-%d'),
                        "offsetDays": delta_days,
                        "relationship": relation,
                        "searchInterest": int(spike_values.get(spike_date, 0)),
                        "accession": accession,
                        "priceReturn": price_return
                    }
                    correlation_results.append(result_record)

                    cursor.execute('''
                        INSERT INTO correlations (ticker, filing_date, spike_date, offset_days, relationship, search_interest, accession, price_return_pct, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ticker, filing_date.strftime('%Y-%m-%d'), spike_date.strftime('%Y-%m-%d'),
                        delta_days, relation, int(spike_values.get(spike_date, 0)), accession,
                        price_return, datetime.now().isoformat()
                    ))

    conn.commit()
    conn.close()
    return pd.DataFrame(correlation_results)


def correlate_contracts_with_search(trend_df, detected_anomalies):
    """
    Cross-references USAspending federal contract award dates with
    Google Trends search anomalies safely, handling missing or None dates.
    """
    print("\n[+] Cross-referencing Federal Contracts with Search Velocity anomalies...")
    conn = sqlite3.connect(DB_NAME)
    df_contracts = pd.read_sql("SELECT ticker, amount, action_date, description FROM federal_contracts", conn)
    conn.close()

    if df_contracts.empty:
        print("[!] Warning: No contract records found in SQLite database.")
        return pd.DataFrame()

    # Drop rows where action_date is null, 'None', or empty string
    df_contracts = df_contracts.dropna(subset=['action_date'])
    df_contracts = df_contracts[~df_contracts['action_date'].isin(['None', 'null', ''])]

    if df_contracts.empty:
        print("[!] Warning: All fetched contract records contain invalid or missing action dates.")
        return pd.DataFrame()

    # Coerce dates safely with format specification
    df_contracts['action_date'] = pd.to_datetime(df_contracts['action_date'], format='mixed', errors='coerce')
    df_contracts = df_contracts.dropna(subset=['action_date'])

    trend_df.index = pd.to_datetime(trend_df.index)
    contract_correlations = []

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    for _, contract in df_contracts.iterrows():
        ticker = contract['ticker']
        contract_date = contract['action_date']
        amount = contract['amount']

        spike_data = detected_anomalies.get(ticker, {})
        spike_dates = spike_data.get("dates", [])

        for spike_date in spike_dates:
            delta_days = (spike_date - contract_date).days

            # Look for search spikes occurring within 0 to 7 days after contract award
            if 0 <= delta_days <= 7:
                contract_correlations.append({
                    "ticker": ticker,
                    "contractDate": contract_date.strftime('%Y-%m-%d'),
                    "spikeDate": spike_date.strftime('%Y-%m-%d'),
                    "lagDays": delta_days,
                    "contractAmount": amount
                })

                cursor.execute('''
                    INSERT INTO contract_correlations (ticker, contract_date, spike_date, lag_days, contract_amount, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    ticker, contract_date.strftime('%Y-%m-%d'), spike_date.strftime('%Y-%m-%d'),
                    delta_days, amount, datetime.now().isoformat()
                ))

    conn.commit()
    conn.close()
    return pd.DataFrame(contract_correlations)

if __name__ == "__main__":
    print("=== Alternative Data Intelligence Pipeline Initialized ===")
    init_database()

    # Step 1: Gather structured regulatory filing metadata
    df_sec_filings = fetch_structured_sec_filings(COMPANY_TICKERS)
    if not df_sec_filings.empty:
        print(f"\n--- Recent SEC 8-K Filings Summary ({len(df_sec_filings)} found) ---")
        print(df_sec_filings.head())

    # Step 2: Gather federal procurement/contract milestones
    df_contracts = fetch_federal_contract_awards(COMPANY_TICKERS)
    if not df_contracts.empty:
        print(f"\n--- Recent Federal Contracts Summary ({len(df_contracts)} found) ---")
        print(df_contracts[['ticker', 'amount', 'actionDate']].head())

    # Step 3: Gather behavioral proxy data (Public attention velocity)
    trend_data = analyze_search_velocity(COMPANY_TICKERS)

    if not trend_data.empty:
        print("\n--- Recent Search Interest Summary ---")
        print(trend_data.tail())

        detected_spikes = detect_search_anomalies(trend_data)

        if detected_spikes:
            print("\n[ALERT] Detected abnormal public search velocity spikes:")
            for ticker, details in detected_spikes.items():
                print(f"    -> {ticker}: Spikes noted on {len(details['dates'])} date(s).")
        else:
            print("\n[INFO] No significant search velocity anomalies detected in the window.")

        # Step 4: Temporal Alignment / Event Studies & SQLite Persistence
        df_correlation = correlate_filings_with_search(df_sec_filings, trend_data, detected_spikes)
        df_contract_corr = correlate_contracts_with_search(trend_data, detected_spikes)

        if not df_correlation.empty:
            print("\n==================================================")
            print(" [CORRELATION ALERT] SEC Filing vs. Search Spikes")
            print("==================================================")
            for _, row in df_correlation.iterrows():
                print(f"Ticker: {row['ticker']}")
                print(f" ├── 8-K Filed Date : {row['filingDate']}")
                print(f" ├── Search Spike   : {row['spikeDate']} ({row['relationship']}, Offset: {row['offsetDays']}d)")
                print(f" ├── Interest Score : {row['searchInterest']}")
                print(f" ├── Price Return   : {row['priceReturn']}%")
                print(f" └── Accession No.  : {row['accession']}")
                print("-" * 50)

        if not df_contract_corr.empty:
            print("\n==================================================")
            print(" [PROCUREMENT ALERT] Federal Contract vs. Search Spike")
            print("==================================================")
            for _, row in df_contract_corr.iterrows():
                print(f"Ticker: {row['ticker']}")
                print(f" ├── Contract Date  : {row['contractDate']}")
                print(f" ├── Search Spike   : {row['spikeDate']} (Lag: +{row['lagDays']}d)")
                print(f" ├── Contract Amt   : ${row['contractAmount']:,.2f}")
                print("-" * 50)
        else:
            print("\n[INFO] No temporal overlaps found between federal contracts and search spikes.")

    print("\n=== Pipeline Execution Complete (All Signals Persisted to SQLite) ===")