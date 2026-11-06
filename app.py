import os
import sqlite3
import time
from datetime import datetime, timedelta
import numpy as np
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

# Entity classification for custom alternative data ingestion routing
ENTITY_CONFIG = {
    "PLTR": {"type": "DEFENSE_ENTERPRISE", "recipient_name": "PALANTIR TECHNOLOGIES INC."},
    "ASTS": {"type": "COMMERCIAL_SPACE", "recipient_name": "AST SPACEBUILDS"},
    "RDDT": {"type": "CONSUMER_TECH", "app_id": "1064216828"}  # Apple App Store ID for Reddit
}


def init_database():
    """Initializes SQLite database with tables for federal contracts, consumer metrics, and correlations."""
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
        CREATE TABLE IF NOT EXISTS consumer_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            metric_name TEXT,
            metric_value REAL,
            recorded_date TEXT,
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
        except Exception as e:
            print(f"[!] Error querying SEC API for {ticker}: {e}")
        time.sleep(0.2)

    return pd.concat(filing_records, ignore_index=True) if filing_records else pd.DataFrame()


def analyze_search_velocity(tickers):
    print("\n[*] Connecting to Google Trends API (PyTrends)...")
    pytrends = TrendReq(hl='en-US', tz=360)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    timeframe_str = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

    print(f"[-] Pulling search interest metrics for timeframe: {timeframe_str}")
    pytrends.build_payload(tickers, cat=0, timeframe=timeframe_str, geo='US')
    interest_df = pytrends.interest_over_time()

    if not interest_df.empty and 'isPartial' in interest_df.columns:
        interest_df = interest_df.drop(columns=['isPartial'])
    return interest_df


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
    print("\n[*] Querying USAspending.gov Federal Contract API...")
    contract_records = []
    api_url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    for ticker in tickers:
        config = ENTITY_CONFIG.get(ticker, {})
        if config.get("type") != "DEFENSE_ENTERPRISE":
            continue

        payload = {
            "filters": {
                "time_period": [
                    {"start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}],
                "recipient_search_text": [config["recipient_name"]],
                "award_type_codes": ["A", "B", "C", "D"]
            },
            "fields": ["Award ID", "Recipient Name", "Award Amount", "Action Date", "Description"],
            "pagination": {"limit": 10, "page": 1, "sort": "Action Date", "order": "desc"}
        }

        try:
            response = requests.post(api_url, json=payload, timeout=10)
            if response.status_code == 200:
                results = response.json().get("results", [])
                for award in results:
                    amount = float(award.get("Award Amount") or 0.0)
                    action_date = award.get("Action Date")
                    award_id = award.get("Award ID")
                    desc = award.get("Description", "No description provided")

                    contract_records.append({"ticker": ticker, "amount": amount, "actionDate": action_date})
                    cursor.execute('''
                        INSERT INTO federal_contracts (ticker, award_id, amount, action_date, description, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (ticker, str(award_id), amount, str(action_date), str(desc), datetime.now().isoformat()))
                print(f"[-] Fetched {len(results)} federal contract records for {ticker}")
        except Exception as e:
            print(f"[!] Error querying USAspending API for {ticker}: {e}")

    conn.commit()
    conn.close()
    return pd.DataFrame(contract_records)


def fetch_consumer_app_metrics(tickers):
    """
    Fetches consumer engagement metrics (e.g., Apple App Store rank / rating velocity)
    for consumer tech companies like Reddit (RDDT).
    """
    print("\n[*] Querying Consumer App Store Metrics & Engagement API...")
    consumer_records = []
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    for ticker in tickers:
        config = ENTITY_CONFIG.get(ticker, {})
        if config.get("type") == "CONSUMER_TECH":
            app_id = config["app_id"]
            lookup_url = f"https://itunes.apple.com/lookup?id={app_id}"

            try:
                response = requests.get(lookup_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    if results:
                        app_info = results[0]
                        avg_rating = float(app_info.get("averageUserRating", 0.0))
                        rating_count = int(app_info.get("userRatingCount", 0))

                        consumer_records.append({
                            "ticker": ticker,
                            "metric_name": "userRatingCount",
                            "metric_value": rating_count,
                            "date": datetime.now().strftime('%Y-%m-%d')
                        })

                        cursor.execute('''
                            INSERT INTO consumer_metrics (ticker, metric_name, metric_value, recorded_date, timestamp)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (ticker, "userRatingCount", rating_count, datetime.now().strftime('%Y-%m-%d'),
                              datetime.now().isoformat()))
                        print(f"[-] Fetched App Store engagement metrics for {ticker} (Rating Count: {rating_count:,})")
            except Exception as e:
                print(f"[!] Error fetching app store metrics for {ticker}: {e}")

    conn.commit()
    conn.close()
    return pd.DataFrame(consumer_records)


def fetch_market_reaction(ticker, target_date):
    try:
        dt = pd.to_datetime(target_date)
        start = (dt - timedelta(days=2)).strftime('%Y-%m-%d')
        end = (dt + timedelta(days=3)).strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start, end=end, progress=False)
        if not df.empty and len(df) >= 2:
            close_series = df['Close'].squeeze()
            start_price = float(close_series.iloc[0])
            end_price = float(close_series.iloc[-1])
            return round(((end_price - start_price) / start_price) * 100, 2)
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
                    relation = "Leading" if delta_days < 0 else ("Coincident" if delta_days == 0 else "Lagging")
                    price_return = fetch_market_reaction(ticker, spike_date)

                    correlation_results.append({
                        "ticker": ticker, "filingDate": filing_date.strftime('%Y-%m-%d'),
                        "spikeDate": spike_date.strftime('%Y-%m-%d'), "offsetDays": delta_days,
                        "relationship": relation, "searchInterest": int(spike_values.get(spike_date, 0)),
                        "accession": accession, "priceReturn": price_return
                    })

                    cursor.execute('''
                        INSERT INTO correlations (ticker, filing_date, spike_date, offset_days, relationship, search_interest, accession, price_return_pct, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                    ticker, filing_date.strftime('%Y-%m-%d'), spike_date.strftime('%Y-%m-%d'), delta_days, relation,
                    int(spike_values.get(spike_date, 0)), accession, price_return, datetime.now().isoformat()))

    conn.commit()
    conn.close()
    return pd.DataFrame(correlation_results)


def compute_adaptive_alpha_scores(tickers, detected_anomalies):
    """
    Adaptive Alpha Confidence Score model:
    - Evaluates Federal Contracts for Enterprise/Defense tickers (PLTR).
    - Evaluates Consumer Engagement/App Velocity for Consumer Tech tickers (RDDT).
    - Evaluates Search Velocity & SEC Proximity across all tickers.
    """
    print("\n[+] Computing Adaptive Multi-Signal Alpha Confidence Scores...")
    conn = sqlite3.connect(DB_NAME)
    df_contracts = pd.read_sql("SELECT ticker, amount FROM federal_contracts", conn)
    df_consumer = pd.read_sql("SELECT ticker, metric_value FROM consumer_metrics", conn)
    df_correlations = pd.read_sql("SELECT ticker FROM correlations", conn)
    conn.close()

    scores = {}

    for ticker in tickers:
        score = 0.0
        config = ENTEntityConfig = ENTITY_CONFIG.get(ticker, {})
        entity_type = config.get("type")

        # 1. Fundamental / Engagement Vector (Max 40 pts)
        if entity_type == "DEFENSE_ENTERPRISE":
            ticker_contracts = df_contracts[df_contracts['ticker'] == ticker]
            total_awarded = ticker_contracts['amount'].fillna(0).sum()
            if total_awarded > 0:
                score += min(40.0, np.log10(total_awarded + 1) * 6.5)
        elif entity_type == "CONSUMER_TECH":
            # For consumer tech, reward massive active user engagement base / app velocity
            ticker_consumer = df_consumer[df_consumer['ticker'] == ticker]
            if not ticker_consumer.empty:
                engagement_val = ticker_consumer['metric_value'].iloc[-1]
                # Scale rating count or community velocity
                score += min(40.0, np.log10(engagement_val + 1) * 8.0)

        # 2. Search Anomaly Intensity (Max 40 pts)
        spike_data = detected_anomalies.get(ticker, {})
        if spike_data:
            values = list(spike_data.get("values", {}).values())
            velocities = list(spike_data.get("velocities", {}).values())
            if values and velocities:
                avg_intensity = np.mean(values)
                max_velocity = max(velocities)
                score += min(40.0, (avg_intensity / 100.0 * 20.0) + (max_velocity / 50.0 * 20.0))

        # 3. Compliance Proximity Bonus (20 pts)
        if not df_correlations[df_correlations['ticker'] == ticker].empty:
            score += 20.0

        scores[ticker] = round(min(100.0, score), 2)

    print("\n==================================================")
    print(" 📊 ADAPTIVE ALPHA CONFIDENCE SCOREBOARD")
    print("==================================================")
    for ticker, score in scores.items():
        rating = "HIGH CONVICTION" if score >= 70 else ("MODERATE" if score >= 40 else "LOW / NOISE")
        print(f"Ticker: {ticker:<5} | Score: {score:>5.2f} / 100 | Signal: {rating}")
    print("--------------------------------------------------")
    return scores


if __name__ == "__main__":
    print("=== Alternative Data Intelligence Pipeline Initialized ===")
    init_database()

    df_sec_filings = fetch_structured_sec_filings(COMPANY_TICKERS)
    df_contracts = fetch_federal_contract_awards(COMPANY_TICKERS)
    df_consumer_metrics = fetch_consumer_app_metrics(COMPANY_TICKERS)

    trend_data = analyze_search_velocity(COMPANY_TICKERS)

    if not trend_data.empty:
        detected_spikes = detect_search_anomalies(trend_data)
        df_correlation = correlate_filings_with_search(df_sec_filings, trend_data, detected_spikes)

        if not df_correlation.empty:
            print("\n==================================================")
            print(" [CORRELATION ALERT] SEC Filing vs. Search Spikes")
            print("==================================================")
            for _, row in df_correlation.iterrows():
                print(
                    f"Ticker: {row['ticker']} | Filing: {row['filingDate']} | Spike: {row['spikeDate']} | Return: {row['priceReturn']}%")

        compute_adaptive_alpha_scores(COMPANY_TICKERS, detected_spikes)

    print("\n=== Pipeline Execution Complete ===")