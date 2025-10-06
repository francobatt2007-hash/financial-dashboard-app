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

# --- Pre-load SEC Ticker Data from local file ---
ticker_cik_map = {}
def load_local_ticker_data():
    global ticker_cik_map
    try:
        # Load the pre-processed JSON file from the project directory
        with open('ticker_to_cik.json', 'r') as f:
            ticker_cik_map = json.load(f)
        logging.info(f"Successfully loaded {len(ticker_cik_map)} ticker-CIK mappings from local file.")
    except Exception as e:
        logging.error(f"Failed to load local ticker_to_cik.json: {e}", exc_info=True)

def get_cik_from_ticker(ticker):
    # The map is now a simple dictionary, which is very fast.
    return ticker_cik_map.get(ticker.upper())

# --- Initialize Clients ---
DEFAULT_USER_AGENT = "Francesco Battaglia franco.batt2007@gmail.com"
EDGAR_API_KEY = os.environ.get("EDGAR_API_KEY", DEFAULT_USER_AGENT)
edgar_client = EdgarClient(user_agent=EDGAR_API_KEY)

# --- STARTUP ---
# Load the data when the module is imported by Vercel
load_local_ticker_data()

# --- Helper Functions ---
# ... (rest of your code remains the same) ...

# Remove the old if __name__ == '__main__' block or modify it for local testing
if __name__ == '__main__':
    # The load function is already called at the top level
    app.run(debug=True, port=5001)