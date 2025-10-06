import os
import json
import logging
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
from sec_edgar_api import EdgarClient
import pandas as pd
import datetime
import numpy as np

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
CORS(app)

# --- Pre-load SEC Ticker Data ---
company_tickers_df = None
def load_company_tickers():
    global company_tickers_df
    try:
        logging.info("Downloading SEC company ticker list...")
        headers = {'User-Agent': "Francesco Battaglia franco.batt2007@gmail.com"}
        response = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
        response.raise_for_status()
        company_data = pd.DataFrame.from_dict(response.json(), orient='index')
        company_data['cik_str'] = company_data['cik_str'].astype(str).str.zfill(10)
        company_tickers_df = company_data
        logging.info("Successfully loaded SEC company ticker list.")
    except Exception as e:
        logging.error(f"Failed to load SEC company tickers: {e}", exc_info=True)
        company_tickers_df = pd.DataFrame()

def get_cik_from_ticker(ticker):
    if company_tickers_df is None or company_tickers_df.empty: return None
    match = company_tickers_df[company_tickers_df['ticker'] == ticker.upper()]
    return match.iloc[0]['cik_str'] if not match.empty else None

# --- Initialize Clients ---
DEFAULT_USER_AGENT = "Francesco Battaglia franco.batt2007@gmail.com"
EDGAR_API_KEY = os.environ.get("EDGAR_API_KEY", DEFAULT_USER_AGENT)
edgar_client = EdgarClient(user_agent=EDGAR_API_KEY)


# --- Helper Functions ---
def get_latest_edgar_value(facts, metric_name):
    try:
        metric_data = facts['facts']['us-gaap'][metric_name]['units']['USD']
        if metric_data:
            return metric_data[-1]['val']
    except (KeyError, IndexError):
        logging.warning(f"Could not find EDGAR metric '{metric_name}' for this ticker.")
        return None
    return None

def process_charts_data(stock, facts):
    charts_data = {}
    
    try:
        def get_edgar_quarterly_data(metric_name):
            try:
                filings = facts['facts']['us-gaap'][metric_name]['units']['USD']
                df = pd.DataFrame(filings)
                df['end'] = pd.to_datetime(df['end'])
                df = df[df['fp'].str.startswith('Q') | (df['fp'] == 'FY')]
                df = df.sort_values('end').drop_duplicates(subset=['end'], keep='last')
                return df.tail(20).to_dict('records')
            except (KeyError, IndexError): return []
        revenue_data, net_income_data, eps_data, assets_data, liabilities_data = get_edgar_quarterly_data('Revenues'), get_edgar_quarterly_data('NetIncomeLoss'), get_edgar_quarterly_data('EarningsPerShareDiluted'), get_edgar_quarterly_data('Assets'), get_edgar_quarterly_data('Liabilities')
        if revenue_data:
            labels = [f"{f['fy']} {f['fp']}" for f in revenue_data]
            charts_data["revenueAndEarnings"] = {"title": "Quarterly Revenue & Net Income", "type": "bar", "labels": labels, "datasets": [{"label": "Total Revenue", "data": [f['val'] for f in revenue_data]}, {"label": "Net Income", "data": [next((ni['val'] for ni in net_income_data if ni['end'] == r['end']), None) for r in revenue_data]}]}
        if eps_data:
            labels = [f"{f['fy']} {f['fp']}" for f in eps_data]
            charts_data["earningsPerShare"] = {"title": "Quarterly Diluted EPS", "type": "bar", "labels": labels, "datasets": [{"label": "Diluted EPS", "data": [f['val'] for f in eps_data]}]}
        if assets_data and liabilities_data:
            labels = [f"{f['fy']} {f['fp']}" for f in assets_data]
            charts_data["assetsVsLiabilities"] = {"title": "Quarterly Assets vs. Liabilities", "type": "bar", "labels": labels, "datasets": [{"label": "Total Assets", "data": [f['val'] for f in assets_data]}, {"label": "Total Liabilities", "data": [next((li['val'] for li in liabilities_data if li['end'] == a['end']), None) for a in assets_data]}]}
    except Exception as e:
        logging.warning(f"Could not generate EDGAR-based charts. Error: {e}")

    q_financials, q_balance_sheet, q_cash_flow = stock.quarterly_financials, stock.quarterly_balance_sheet, stock.quarterly_cashflow
    try:
        if q_financials is not None and not q_financials.empty:
            df = q_financials.T.head(8).iloc[::-1]
            labels = [f"Q{q.quarter} {q.year}" for q in df.index]
            if 'EBITDA' in df.columns: charts_data["ebitda"] = {"title": "Quarterly EBITDA (Recent)", "type": "bar", "labels": labels, "datasets": [{"label": "EBITDA", "data": df["EBITDA"].tolist()}]}
            shares_row_name = 'Basic Average Shares'
            if shares_row_name in df.columns: charts_data["sharesOutstanding"] = {"title": "Quarterly Shares Outstanding (Recent)", "type": "bar", "labels": labels, "datasets": [{"label": "Shares", "data": df[shares_row_name].tolist()}]}
    except Exception as e:
        logging.warning(f"Could not generate yfinance financial charts. Error: {e}")
    try:
        if q_balance_sheet is not None and not q_balance_sheet.empty:
            df = q_balance_sheet.T.head(8).iloc[::-1]
            labels = [f"Q{q.quarter} {q.year}" for q in df.index]
            cash_col, debt_col = ('Cash And Cash Equivalents' if 'Cash And Cash Equivalents' in df else 'Total Cash'), ('Total Debt' if 'Total Debt' in df else None)
            if cash_col in df and debt_col and debt_col in df: charts_data["cashVsDebt"] = {"title": "Quarterly Cash vs. Debt (Recent)", "type": "bar", "labels": labels, "datasets": [{"label": "Total Cash", "data": df[cash_col].tolist()}, {"label": "Total Debt", "data": df[debt_col].tolist()}]}
    except Exception as e:
        logging.warning(f"Could not generate yfinance balance sheet charts. Error: {e}")
    
    try:
        if q_cash_flow is not None and not q_cash_flow.empty:
            df = q_cash_flow.T.head(8).iloc[::-1]
            labels = [f"Q{q.quarter} {q.year}" for q in df.index]
            buyback_col, dividend_col = 'Repurchase Of Capital Stock', 'Cash Dividends Paid'
            buybacks, dividends = None, None
            if buyback_col in df.columns:
                buybacks = pd.to_numeric(df[buyback_col], errors='coerce').fillna(0) * -1
            if dividend_col in df.columns:
                dividends = pd.to_numeric(df[dividend_col], errors='coerce').fillna(0) * -1
            if buybacks is not None and dividends is not None:
                charts_data["returnOfCapital"] = {"title": "Quarterly Return of Capital (Recent)", "type": "bar", "labels": labels, "datasets": [{"label": "Share Buybacks", "data": buybacks.tolist()}, {"label": "Dividends Paid", "data": dividends.tolist()}]}
    except Exception as e:
        logging.warning(f"Could not generate Return of Capital chart. Error: {e}")

    try:
        divs = stock.dividends
        if divs is not None and not divs.empty:
            q_divs = divs.resample('QE').sum()
            q_divs = q_divs[q_divs > 0].tail(20)
            if not q_divs.empty: charts_data["dividends"] = {"title": "Quarterly Dividends per Share", "type": "bar", "labels": [f"Q{q.quarter} {q.year}" for q in q_divs.index], "datasets": [{"label": "Dividend/Share", "data": q_divs.tolist()}]}
    except Exception as e:
        logging.warning(f"Could not generate dividend chart. Error: {e}")
        
    try:
        hist_5y = stock.history(period="5y")
        if not hist_5y.empty:
            hist_5y.index = hist_5y.index.tz_localize(None)
        if not hist_5y.empty and q_financials is not None and not q_financials.empty and q_balance_sheet is not None and not q_balance_sheet.empty:
            all_dates = pd.to_datetime(q_financials.columns)[-20:]
            ps_vals, pb_vals = [], []
            shares_out = stock.info.get('sharesOutstanding')
            def find_first_row(df, candidates):
                for name in candidates:
                    if name in df.index: return name
                return None
            revenue_row_name = find_first_row(q_financials, ['Total Revenue', 'Revenues'])
            equity_row_candidates = ['Total Stockholder Equity', 'Total Stockholders Equity', 'Total Equity', 'Stockholders Equity', 'Total Shareholder Equity']
            equity_row_name = find_first_row(q_balance_sheet, equity_row_candidates)
            
            for q_date in all_dates:
                price_series = hist_5y['Close'].loc[:q_date]
                price = price_series.iloc[-1] if not price_series.empty else None
                market_cap = price * shares_out if shares_out and price is not None else None
                ps_val, pb_val = None, None
                if market_cap is not None:
                    if revenue_row_name:
                        revenue_series = pd.to_numeric(q_financials.loc[revenue_row_name], errors='coerce')
                        revenue_ttm = revenue_series.rolling(4, min_periods=1).sum().get(q_date)
                        if revenue_ttm and revenue_ttm != 0: ps_val = market_cap / revenue_ttm
                    if equity_row_name and q_date in q_balance_sheet.columns:
                        equity = pd.to_numeric(q_balance_sheet.loc[equity_row_name].get(q_date), errors='coerce')
                        if equity and equity != 0: pb_val = market_cap / equity
                ps_vals.append(ps_val)
                pb_vals.append(pb_val)
            charts_data["historicalRatios"] = {"title": "Historical P/S and P/B Ratios", "type": "line", "labels": [d.strftime('%Y-%m') for d in all_dates], "datasets": [{"label": "Price-to-Sales (TTM)", "data": ps_vals, "yAxisID": "y"}, {"label": "Price-to-Book", "data": pb_vals, "yAxisID": "y"}]}
    except Exception as e:
        logging.warning(f"Could not generate historical ratios chart. Error: {e}")

    return charts_data

def sanitize_for_json(data):
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(i) for i in data]
    elif isinstance(data, (np.floating, float)) and not np.isfinite(data):
        return None
    return data

@app.route('/api/stock-data/<ticker>')
def get_stock_data(ticker):
    logging.info(f"--- Starting data fetch for {ticker.upper()} ---")
    try:
        cik = get_cik_from_ticker(ticker)
        if not cik: return jsonify({"error": f"Ticker '{ticker}' not found."}), 404
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or not info.get('longName'):
            return jsonify({"error": f"Could not retrieve valid data for ticker '{ticker}' from Yahoo Finance."}), 404
        facts, current_price = edgar_client.get_company_facts(cik=cik), info.get('currentPrice') or info.get('regularMarketPrice')
        response_data = {
            "companyInfo": { "name": info.get('longName'), "ceo": next((o.get('name') for o in info.get('companyOfficers', []) if 'CEO' in o.get('title', '')), 'N/A'), "employees": info.get('fullTimeEmployees'), "sector": info.get('sector'), "industry": info.get('industry'), "summary": info.get('longBusinessSummary'), "cik": cik, "ticker": ticker },
            "quote": { "currentPrice": current_price },
            "financials_edgar": {"assets": get_latest_edgar_value(facts, 'Assets'), "revenues": get_latest_edgar_value(facts, 'Revenues'), "netIncome": get_latest_edgar_value(facts, 'NetIncomeLoss'), "liabilities": get_latest_edgar_value(facts, 'Liabilities')},
            "financials_yahoo": {"revenuePerShare": info.get('revenuePerShare'), "revenueGrowth": info.get('revenueGrowth'), "earningsGrowth": info.get('earningsQuarterlyGrowth'), "grossProfits": info.get('grossProfits'), "ebitda": info.get('ebitda'), "netIncomeToCommon": info.get('netIncomeToCommon'), "trailingEps": info.get('trailingEps'), "totalCash": info.get('totalCash'), "totalDebt": info.get('totalDebt'), "currentRatio": info.get('currentRatio'), "bookValue": info.get('bookValue'), "operatingCashflow": info.get('operatingCashflow'), "freeCashflow": info.get('freeCashflow'), "netDebt": info.get('totalDebt', 0) - info.get('totalCash', 0)},
            "valuation": { "marketCap": info.get('marketCap'), "enterpriseValue": info.get('enterpriseValue'), "trailingPE": info.get('trailingPE'), "forwardPE": info.get('forwardPE'), "pegRatio": info.get('pegRatio'), "priceToSales": info.get('priceToSalesTrailing12Months'), "priceToBook": info.get('priceToBook'), "evToEbitda": info.get('enterpriseToEbitda') },
            "cashFlowMetrics": { "fcfYield": (info.get('freeCashflow', 0) / info.get('sharesOutstanding', 1) / current_price) if info.get('sharesOutstanding') and current_price else 0, "fcfPerShare": info.get('freeCashflow', 0) / info.get('sharesOutstanding', 1) if info.get('sharesOutstanding') else 0 },
            "profitability": { "profitMargin": info.get('profitMargins'), "operatingMargin": info.get('operatingMargins'), "returnOnAssets": info.get('returnOnAssets'), "returnOnEquity": info.get('returnOnEquity') },
            "dividends": { "dividendYield": (info.get('dividendRate', 0) / current_price) if info.get('dividendRate') and current_price else info.get('dividendYield'), "payoutRatio": info.get('payoutRatio'), "exDividendDate": datetime.datetime.fromtimestamp(info['exDividendDate']).strftime('%Y-%m-%d') if info.get('exDividendDate') else 'N/A' },
            "shareStats": { "sharesOutstanding": info.get('sharesOutstanding'), "floatShares": info.get('floatShares'), "heldByInsiders": info.get('heldPercentInsiders'), "heldByInstitutions": info.get('heldPercentInstitutions') },
            "charts": process_charts_data(stock, facts)
        }
        safe_response_data = sanitize_for_json(response_data)
        return jsonify(safe_response_data)
    except Exception as e:
        logging.error(f"An error occurred while fetching data for {ticker}: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve complete stock data. Check if the ticker is valid."}), 500

@app.route('/api/price-history/<ticker>/<range_str>')
def get_price_history(ticker, range_str):
    logging.info(f"--- Fetching price history for {ticker} over {range_str} ---")
    try:
        range_map = {
            '1d': {'period': '1d', 'interval': '5m'}, '5d': {'period': '5d', 'interval': '30m'},
            '1m': {'period': '1mo', 'interval': '1d'}, '6m': {'period': '6mo', 'interval': '1d'},
            '1y': {'period': '1y', 'interval': '1d'}, '5y': {'period': '5y', 'interval': '1d'}
        }
        params = range_map.get(range_str)
        if not params: return jsonify({"error": "Invalid range"}), 400
        stock = yf.Ticker(ticker)
        hist = stock.history(period=params['period'], interval=params['interval'])
        if hist.empty: return jsonify({"error": "No price data found for this range."}), 404
        price_data = {
            "labels": hist.index.strftime('%Y-%m-%d %H:%M').tolist(),
            "datasets": [{"label": "Close Price", "data": hist['Close'].tolist(), "fill": True, "pointRadius": 0, "tension": "0.1"}]
        }
        return jsonify(price_data)
    except Exception as e:
        logging.error(f"Error fetching price history for {ticker}: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve price history."}), 500

@app.route('/api/stock-news/<ticker>')
def get_stock_news(ticker):
    logging.info(f"--- Fetching news for {ticker.upper()} ---")
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        formatted_news = []
        for article in news:
            if article.get('title') and article.get('link'):
                publish_time = article.get('providerPublishTime')
                if publish_time:
                    published_date = datetime.datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d %H:%M')
                else:
                    published_date = 'Date not available'
                formatted_news.append({'title': article.get('title'),'link': article.get('link'),'publisher': article.get('publisher', 'Unknown Publisher'),'published_date': published_date})
            if len(formatted_news) >= 16: break
        return jsonify(formatted_news)
    except Exception as e:
        logging.error(f"Error fetching news for {ticker}: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve news."}), 500

if __name__ == '__main__':
    load_company_tickers()
    app.run(debug=True, port=5001)