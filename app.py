import os
import time
from datetime import datetime, timedelta
import pandas as pd
import requests
from pytrends.request import TrendReq

# ==========================================
# CONFIGURATION & TARGETS
# ==========================================
# SEC EDGAR requires a descriptive User-Agent (Name/Email)
SEC_USER_AGENT = "QuantResearcher research@example.com"
COMPANY_TICKERS = ["SOUN", "PLTR", "RDDT"]
LOOKBACK_DAYS = 30

# Official SEC CIK Mapping for target companies
CIK_MAP = {
    "AAPL": "0000320193",
    "TSLA": "0001318605",
    "MSFT": "0000789019"
}

def get_cik_for_ticker(ticker, headers):
    """
    Dynamically fetches the official SEC CIK mapping file
    so you never have to hardcode ticker-to-CIK dictionaries.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # The SEC JSON structure is a dict of indexed objects: {"0": {"cik_str": ..., "ticker": "...", "title": "..."}, ...}
            for entry in data.values():
                if entry['ticker'].upper() == ticker.upper():
                    # CIK needs to be 10 digits zero-padded for the submissions API
                    return str(entry['cik_str']).zfill(10)
    except Exception as e:
        print(f"[!] Error fetching SEC ticker map: {e}")
    return None



def fetch_structured_sec_filings(tickers):
    print("[*] Fetching structured SEC EDGAR metadata via JSON API...")
    headers = {"User-Agent": SEC_USER_AGENT}
    filing_records = []

    for ticker in tickers:
        # Dynamically fetch CIK instead of relying on CIK_MAP
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
    """
    Leverages PyTrends to pull public search volume interest over time,
    formatting it into a structured time-series DataFrame.
    """
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
    """
    Calculates rolling z-scores AND daily rate of change (velocity) on public
    search volume to filter out low-volume noise and capture true behavioral spikes.
    """
    print("\n[+] Running enhanced anomaly detection (Z-Score + Velocity Filter)...")
    anomalies = {}

    for column in trend_df.columns:
        series = trend_df[column]
        rolling_mean = series.rolling(window=7, min_periods=1).mean()
        rolling_std = series.rolling(window=7, min_periods=1).std().fillna(1)

        # Quantitative indicators
        z_scores = (series - rolling_mean) / rolling_std
        daily_velocity = series.diff().fillna(0)

        # Dual-gate condition: Statistical outlier (Z > 2.0) + Momentum jump (> 15 pts)
        spike_mask = (z_scores > 2.0) & (daily_velocity > 15)
        spike_dates = trend_df.index[spike_mask].tolist()

        if spike_dates:
            anomalies[column] = {
                "dates": spike_dates,
                "values": series[spike_mask].to_dict(),
                "velocities": daily_velocity[spike_mask].to_dict()
            }

    return anomalies


def correlate_filings_with_search(df_filings, trend_df, detected_anomalies):
    """
    Performs a temporal event study aligning SEC 8-K filing dates with
    Google Trends search anomalies within a +/- 3 day window.
    """
    print("\n[+] Cross-referencing SEC filings with Search Velocity anomalies...")

    if df_filings.empty or not detected_anomalies:
        return pd.DataFrame()

    df_filings['filingDate'] = pd.to_datetime(df_filings['filingDate'])
    trend_df.index = pd.to_datetime(trend_df.index)

    correlation_results = []

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

                # Look for anomalies within a +/- 3 day window
                if -3 <= delta_days <= 3:
                    if delta_days < 0:
                        relation = "Leading (Potential Leak/Rumor)"
                    elif delta_days == 0:
                        relation = "Coincident (Same-Day Reaction)"
                    else:
                        relation = "Lagging (Post-Filing Momentum)"

                    correlation_results.append({
                        "ticker": ticker,
                        "filingDate": filing_date.strftime('%Y-%m-%d'),
                        "spikeDate": spike_date.strftime('%Y-%m-%d'),
                        "offsetDays": delta_days,
                        "relationship": relation,
                        "searchInterest": spike_values.get(spike_date, 0),
                        "accession": accession
                    })

    return pd.DataFrame(correlation_results)


if __name__ == "__main__":
    print("=== Alternative Data Pipeline Initialized ===")

    # Step 1: Gather structured regulatory filing metadata
    df_sec_filings = fetch_structured_sec_filings(COMPANY_TICKERS)
    if not df_sec_filings.empty:
        print(f"\n--- Recent SEC 8-K Filings Summary ({len(df_sec_filings)} found) ---")
        print(df_sec_filings.head())

    # Step 2: Gather behavioral proxy data (Public attention velocity)
    trend_data = analyze_search_velocity(COMPANY_TICKERS)

    if not trend_data.empty:
        print("\n--- Recent Search Interest Summary ---")
        print(trend_data.tail())

        # Step 3: Run quantitative enhanced anomaly check
        detected_spikes = detect_search_anomalies(trend_data)

        if detected_spikes:
            print("\n[ALERT] Detected abnormal public search velocity spikes:")
            for ticker, details in detected_spikes.items():
                print(f"   -> {ticker}: Spikes noted on {len(details['dates'])} date(s).")
                for d in details['dates']:
                    date_str = d.strftime('%Y-%m-%d')
                    print(
                        f"       • Date: {date_str} | Interest Score: {details['values'][d]} | Daily Jump: +{details['velocities'][d]}")
        else:
            print("\n[INFO] No significant search velocity anomalies detected in the window.")

        # Step 4: Temporal Alignment / Event Study
        df_correlation = correlate_filings_with_search(df_sec_filings, trend_data, detected_spikes)

        if not df_correlation.empty:
            print("\n==================================================")
            print(" [CORRELATION ALERT] SEC Filing vs. Search Spikes")
            print("==================================================")
            for _, row in df_correlation.iterrows():
                print(f"Ticker: {row['ticker']}")
                print(f" ├── 8-K Filed Date : {row['filingDate']}")
                print(f" ├── Search Spike   : {row['spikeDate']} ({row['relationship']}, Offset: {row['offsetDays']}d)")
                print(f" ├── Interest Score : {row['searchInterest']}")
                print(f" └── Accession No.  : {row['accession']}")
                print("-" * 50)
        else:
            print("\n[INFO] No temporal overlaps found between 8-K filings and search spikes in this window.")

    print("\n=== Pipeline Execution Complete ===")
