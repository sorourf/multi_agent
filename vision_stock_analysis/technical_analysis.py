# Libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import tempfile
import os
import json
import kaleido
from io import BytesIO
import pandas as pd
import mplfinance as mpf
import ta  

from datetime import datetime, timedelta

# Configure the API key - IMPORTANT: Use Streamlit secrets or environment variables for security
# For now, using hardcoded API key - REPLACE WITH YOUR ACTUAL API KEY SECURELY
GOOGLE_API_KEY = "AIzaSyA0cUqkZYqV8Q3UeMTQD0hQ_fIkVUaH4Jk" 
genai.configure(api_key=GOOGLE_API_KEY)

# Select the Gemini model - using 'gemini-2.0-flash' as a general-purpose model
MODEL_NAME = 'gemini-2.0-flash' # or other model
gen_model = genai.GenerativeModel(MODEL_NAME)

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# Input for multiple stock tickers (comma-separated)
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated):", "AAPL,MSFT,GOOG")
# Parse tickers by stripping extra whitespace and splitting on commas
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

# Set the date range: start date = one year before today, end date = today
end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)

# Technical indicators selection (applied to every ticker)
st.sidebar.subheader("Technical Indicators")
indicators = st.sidebar.multiselect(
    "Select Indicators:",
    ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
    default=["20-Day SMA"]
)

# Button to fetch data for all tickers
if st.sidebar.button("Fetch Data"):
    stock_data = {}
    for ticker in tickers:
        # Download data for each ticker using yfinance
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            stock_data[ticker] = data
        else:
            st.warning(f"No data found for {ticker}.")
    st.session_state["stock_data"] = stock_data
    st.success("Stock data loaded successfully for: " + ", ".join(stock_data.keys()))
    # st.write(stock_data)
# Ensure we have data to analyze
if "stock_data" in st.session_state and st.session_state["stock_data"]:

    # Define a function to build chart, call the Gemini API and return structured result
    def analyze_ticker(ticker, data):
  
            # Extract data for the specific ticker
        df = data.xs(ticker, axis=1, level=1)

        # Ensure data has the required columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col} in data")

        # Compute technical indicators
        df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)

        bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["BB_High"] = bb.bollinger_hband()
        df["BB_Low"] = bb.bollinger_lband()

        df["MACD"] = ta.trend.macd(df["Close"])
        df["MACD_Signal"] = ta.trend.macd_signal(df["Close"])

        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

        # Compute VWAP
        df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

        # Create mplfinance additional plots
        apds = [
            mpf.make_addplot(df["SMA_20"], color="blue", width=1.2, label="SMA 20"),
            mpf.make_addplot(df["SMA_50"], color="red", width=1.2, label="SMA 50"),
            mpf.make_addplot(df["EMA_20"], color="purple", width=1.2, label="EMA 20"),
            mpf.make_addplot(df["BB_High"], color="black", linestyle="dashed", width=1, label="Bollinger High"),
            mpf.make_addplot(df["BB_Low"], color="black", linestyle="dashed", width=1, label="Bollinger Low"),
            mpf.make_addplot(df["VWAP"], color="orange", width=1.2, label="VWAP"),
            mpf.make_addplot(df["MACD"], panel=1, color="purple", secondary_y=False, label="MACD"),
            mpf.make_addplot(df["MACD_Signal"], panel=1, color="orange", secondary_y=False, label="MACD Signal"),
            mpf.make_addplot(df["RSI"], panel=2, color="green", ylim=(0, 100), secondary_y=False, label="RSI"),
        ]

    
        # st.write(f"✅ Image successfully saved at: {os.path.abspath(save_path)}")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            tmpfile_path = tmpfile.name  # Store temp file path before closing

        # Generate and save the candlestick chart
        mpf.plot(
            df,
            type="candle",
            volume=True,
            style="charles",
            title=f"{ticker} Stock Price with Technical Indicators",
            ylabel="Price",
            addplot=apds,
            panel_ratios=(6, 2, 2),  # Set panel sizes (main chart, MACD, RSI)
            savefig=tmpfile_path  # Save chart as an image
        )

        # Read the saved image file
        with open(tmpfile_path, "rb") as f:
            image_bytes = f.read()

        # Delete temporary file after reading (Windows requires this order)
        os.remove(tmpfile_path)

        # Create an image Part
        image_part = {
            "data": image_bytes,  
            "mime_type": "image/png"
        }

        # Updated prompt asking for a detailed justification of technical analysis and a recommendation.
        analysis_prompt = (
            f"You are a Stock Trader specializing in Technical Analysis at a top financial institution. "
            f"Analyze the stock chart for {ticker} based on its candlestick chart and the displayed technical indicators. "
            f"Provide a detailed justification of your analysis, explaining what patterns, signals, and trends you observe. "
            f"Then, based solely on the chart, provide a recommendation from the following options: "
            f"'Strong Buy', 'Buy', 'Weak Buy', 'Hold', 'Weak Sell', 'Sell', or 'Strong Sell'. "
            f"Return your output as a JSON object with two keys: 'action' and 'justification'."
        )

        # Call the Gemini API with text and image input - Roles added: "user" for both text and image
        contents = [
            {"role": "user", "parts": [analysis_prompt]},  # Text prompt with role "user"
            {"role": "user", "parts": [image_part]}       # Image part with role "user"
        ]

        response = gen_model.generate_content(
            contents=contents  # Pass the restructured 'contents' with roles
        )

        try:
            # Attempt to parse JSON from the response text
            result_text = response.text
            st.write(result_text)
            # Find the start and end of the JSON object within the text (if Gemini includes extra text)
            json_start_index = result_text.find('{')
            json_end_index = result_text.rfind('}') + 1  # +1 to include the closing brace
            if json_start_index != -1 and json_end_index > json_start_index:
                json_string = result_text[json_start_index:json_end_index]
                result = json.loads(json_string)
            else:
                raise ValueError("No valid JSON object found in the response")

        except json.JSONDecodeError as e:
            result = {"action": "Error", "justification": f"JSON Parsing error: {e}. Raw response text: {response.text}"}
        except ValueError as ve:
            result = {"action": "Error", "justification": f"Value Error: {ve}. Raw response text: {response.text}"}
        except Exception as e:
            result = {"action": "Error", "justification": f"General Error: {e}. Raw response text: {response.text}"}

        return image_bytes, result

    # Create tabs: first tab for overall summary, subsequent tabs per ticker
    tab_names = ["Overall Summary"] + list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)

    # List to store overall results
    overall_results = []

    # Process each ticker and populate results
    for i, ticker in enumerate(st.session_state["stock_data"]):
        data = st.session_state["stock_data"][ticker]
        # Analyze ticker: get chart figure and structured output result
        fig, result = analyze_ticker(ticker, data)
        overall_results.append({"Stock": ticker, "Recommendation": result.get("action", "N/A")})
        # In each ticker-specific tab, display the chart and detailed justification
        with tabs[i + 1]:
            st.subheader(f"Analysis for {ticker}")
            st.image(fig, caption=f"{ticker} Candlestick Chart with Technical Indicators")
            st.write("**Detailed Justification:**")
            st.write(result.get("justification", "No justification provided."))
    # In the Overall Summary tab, display a table of all results
    with tabs[0]:
        st.subheader("Overall Structured Recommendations")
        df_summary = pd.DataFrame(overall_results)
        st.table(df_summary)
else:
    st.info("Please fetch stock data using the sidebar.")