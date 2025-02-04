# Stock Price Direction Prediction System Documentation

**Version:** 1.1
**Date:** February 3, 2025

## 1. System Overview

This document provides a comprehensive description of a stock price direction prediction system. The system is designed to predict whether the price of a stock will move up or down on the next trading day. It achieves this by collecting and processing historical stock price data and social sentiment data from Reddit, and then training a machine learning model to make predictions.

The system is composed of four main Python scripts:

1.  **`stock_collector_yfinance.py`:** Collects historical stock price data from Yahoo Finance using the `yfinance` library.
2.  **`stock_collector_reddit.py`:** Collects social sentiment data from Reddit using the `praw` library, focusing on stock-related discussions in relevant subreddits.
3.  **`data_processor.py`:** Processes and combines the collected stock price and sentiment data, engineers features, and prepares the data for model training.
4.  **`model_trainer.py`:** Trains a machine learning model (XGBoost) using the processed data to predict the next-day stock price direction, performs hyperparameter optimization, evaluates the model, and saves model artifacts.

This document will detail each script's functionality, data flow, architecture, usage, and potential improvements.

## 2. Script 1: `stock_collector_yfinance.py` - Historical Stock Data Collection

### 2.1. Purpose and Functionality

The `stock_collector_yfinance.py` script is responsible for collecting historical stock price data for a predefined list of stocks across different sectors. It uses the `yfinance` library to fetch data from Yahoo Finance.

### 2.2. Code Structure and Key Components

*   **Class `StockDataCollector`:** Encapsulates the data collection logic.
    *   **`TRACKED_STOCKS` (Class Attribute):** A dictionary defining the sectors and the stock tickers to be tracked within each sector. This is configurable to add or remove stocks.
    *   **`__init__(self)` (Constructor):**
        *   Initializes the output directory (`historical_data`) where collected data will be saved.
        *   Creates sector-specific subdirectories within the output directory to organize data.
    *   **`_fetch_stock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame` (Method):**
        *   Fetches historical stock data for a given `ticker` and date range using `yf.Ticker(ticker).history(...)`.
        *   Sets the data interval to '1d' (daily).
        *   Adds a 'Ticker' column to the DataFrame.
        *   Calculates several technical indicators and features:
            *   `Daily_Return`: Daily percentage change in 'Close' price.
            *   `Volatility`: 20-day rolling standard deviation of 'Daily_Return'.
            *   `MA50`: 50-day moving average of 'Close' price.
            *   `MA200`: 200-day moving average of 'Close' price.
            *   `Volume_MA20`: 20-day moving average of 'Volume'.
        *   Handles potential errors during data fetching and returns `None` if an error occurs.
    *   **`_save_stock_data(self, df: pd.DataFrame, ticker: str, sector: str) -> bool` (Method):**
        *   Saves the fetched stock data DataFrame to a CSV file within the appropriate sector subdirectory in `historical_data`.
        *   Filenames are timestamped with the collection date.
        *   Saves metadata in a JSON file alongside the CSV, including: ticker, sector, data date range, number of rows and columns, collection timestamp, and whether missing data exists.
        *   Returns `True` on successful save, `False` otherwise.
    *   **`collect_data(self, start_date: str, end_date: str, max_workers: int) -> None` (Method):**
        *   The main function to initiate data collection for all tracked stocks.
        *   Takes `start_date`, `end_date`, and `max_workers` (for multithreading) as parameters.
        *   Creates a list of collection tasks (ticker and sector pairs).
        *   Uses `ThreadPoolExecutor` for concurrent data fetching to speed up collection.
        *   Iterates through the results of each thread, saves successful collections, and handles errors.
        *   Adds a 1-second delay between stock data fetches to respect API rate limits.
        *   Saves a collection summary in a JSON file in the `historical_data` directory, including collection start/end times, duration, date range, successful collections, and total/successful stock counts.
*   **`main()` Function:**
    *   Creates an instance of `StockDataCollector`.
    *   Calls `collector.collect_data()` to initiate data collection for a specified date range (default '2021-01-01' to current date) and with a specified number of worker threads (default 5).
    *   Includes error handling for the entire data collection process.

### 2.3. Key Libraries Used

*   **`yfinance (yf)`:**  For fetching historical stock data from Yahoo Finance.
*   **`pandas (pd)`:** For data manipulation and storage in DataFrames.
*   **`logging`:** For logging information, warnings, and errors to both file and console.
*   **`datetime`, `timezone`:** For handling dates and timezones, ensuring data is timestamped correctly in UTC.
*   **`pathlib.Path`:** For working with file paths in a platform-independent way.
*   **`json`:** For saving metadata in JSON format.
*   **`concurrent.futures.ThreadPoolExecutor`, `as_completed`:** For concurrent execution using threads to speed up data collection.

### 2.4. Configuration

*   **`TRACKED_STOCKS` (Class Attribute):**  This dictionary within the `StockDataCollector` class serves as the main configuration for which stocks to collect data for.  To track different stocks, modify this dictionary directly in the script.

### 2.5. Execution Instructions

1.  **Install Required Libraries:** Ensure you have installed the necessary libraries (listed in section 2.3) using `pip install yfinance pandas`.
2.  **Run the Script:** Open your terminal, navigate to the project directory, and execute:

    ```bash
    python stock_collector_yfinance.py
    ```

3.  **Check Output:** After execution, historical stock data CSV files and metadata JSON files will be saved in the `historical_data` directory, organized by sector. A collection summary JSON file will also be created in the `historical_data` directory. Check the `stock_collector.log` file for any logs or errors.

### 2.6. Output

*   **CSV Files:**  Historical stock data CSV files are saved in `historical_data/{Sector}/{Ticker}_historical_{YYYYMMDD}.csv`. Each file contains daily stock data and calculated technical indicators for a specific stock.
*   **JSON Metadata Files:**  Metadata files corresponding to each CSV are saved as `historical_data/{Sector}/{Ticker}_historical_{YYYYMMDD}.json`.
*   **Collection Summary JSON:**  A summary of the entire data collection process is saved as `historical_data/collection_summary_{YYYYMMDD_HHMMSS}.json`.

### 2.7. Error Handling and Logging

*   **Error Logging:**  The script uses `logging` to record errors and warnings. Errors during data fetching and saving are logged to both the `stock_collector.log` file and the console.
*   **Data Fetching Error Handling:** The `_fetch_stock_data` method includes `try-except` blocks to catch exceptions during data retrieval from `yfinance`. If data fetching fails for a ticker, it logs an error and returns `None`, allowing the script to continue with other stocks.
*   **Data Saving Error Handling:** The `_save_stock_data` method also includes `try-except` to handle potential file saving errors.

### 2.8. Potential Improvements

*   **More Robust Error Handling:** Implement more specific error handling for different types of `yfinance` exceptions (e.g., network errors, ticker not found errors).
*   **Configuration File for Stocks:** Move the `TRACKED_STOCKS` dictionary to an external configuration file (e.g., YAML or JSON) for easier modification without altering the script code.
*   **Data Validation:** Add data validation steps to check the integrity and completeness of the downloaded stock data.
*   **Dynamic Date Range:** Allow the date range to be specified as command-line arguments or in a configuration file for more flexible data collection periods.
*   **Data Caching:** Implement a caching mechanism to avoid redundant data fetches if the script is run multiple times for the same date range.

---

## 3. Script 2: `stock_collector_reddit.py` - Reddit Sentiment Data Collection

### 3.1. Purpose and Functionality

The `stock_collector_reddit.py` script collects social sentiment data from Reddit. It uses the `praw` library to access the Reddit API and fetches top posts from relevant subreddits discussing stocks. The goal is to capture public sentiment and discussions related to the tracked stocks.

### 3.2. Code Structure and Key Components

*   **Class `StockDataCollector`:**  Encapsulates the Reddit data collection logic.
    *   **`TRACKED_STOCKS` (Class Attribute):**  Same as in `stock_collector_yfinance.py`, defining the stocks to track.
    *   **`SUBREDDITS` (Class Attribute):** A list of relevant subreddit names to collect data from (e.g., 'wallstreetbets', 'stocks', etc.). Configurable to adjust the source subreddits.
    *   **`__init__(self, config_path: str = 'config.yaml')` (Constructor):**
        *   Loads Reddit API credentials from a configuration file (`config.yaml`) using `_load_config`.
        *   Initializes the Reddit API client using `praw.Reddit` with credentials from the config file via `_init_reddit_client`.
        *   Initializes the output directory (`data`) for saving Reddit data.
    *   **`_load_config(self, config_path: str) -> Dict[str, Any]` (Method):**
        *   Loads configuration settings from a YAML file specified by `config_path`.  Expected to contain Reddit API credentials (`client_id`, `client_secret`, `user_agent`).
        *   Handles `FileNotFoundError` if the config file is missing.
    *   **`_init_reddit_client(self) -> praw.Reddit` (Method):**
        *   Initializes the `praw.Reddit` client using credentials loaded from the configuration.
        *   Includes error handling to catch exceptions during Reddit client initialization.
    *   **`_fetch_subreddit_posts(self, subreddit_name: str, time_filter: str, limit: int) -> List[Dict]` (Method):**
        *   Fetches top posts from a specified `subreddit_name` for a given `time_filter` (e.g., 'week', 'day') and `limit` (number of posts).
        *   Uses `backoff.on_exception` decorator with exponential backoff and retry logic to handle transient Reddit API errors (like rate limits or network issues). Retries up to 3 times.
        *   Extracts relevant post data: `id`, `subreddit`, `title`, `text` (selftext), `score`, `upvote_ratio`, `num_comments`, `created_utc`, `permalink`.
        *   Respects Reddit API rate limits by adding a `time.sleep(0.5)` delay after each post retrieval.
        *   Handles general exceptions during post fetching and re-raises them for backoff to handle.
    *   **`_identify_stock_mentions(self, text: str) -> List[str]` (Method):**
        *   Identifies mentioned stock tickers within a given `text` (post title and body).
        *   Iterates through `TRACKED_STOCKS` and checks if each ticker (in uppercase) is present in the text (also converted to uppercase for case-insensitive matching).
        *   Returns a list of unique mentioned stock tickers.
    *   **`collect_data(self, time_filter: str = 'week', posts_per_subreddit: int = 500) -> None` (Method):**
        *   The main function to collect Reddit data.
        *   Takes `time_filter` (default 'week') and `posts_per_subreddit` (default 500) as parameters.
        *   Iterates through the `SUBREDDITS` list.
        *   Calls `_fetch_subreddit_posts` for each subreddit to retrieve posts.
        *   Combines posts from all subreddits into `all_posts`.
        *   Creates a Pandas DataFrame from `all_posts`.
        *   Adds a 'mentioned_stocks' column to the DataFrame by applying `_identify_stock_mentions` to the combined title and text of each post.
        *   Filters the DataFrame to keep only posts that mention at least one tracked stock.
        *   Saves the relevant posts DataFrame to a timestamped CSV file in the `data` directory (`stock_mentions_{timestamp}.csv`).
        *   Saves collection metadata in a JSON file (`metadata_{timestamp}.json`) in the `data` directory, including collection timestamp, total and relevant post counts, subreddits used, and tracked stocks.
        *   Logs warnings if no posts are collected.
*   **`main()` Function:**
    *   Example configuration file (`config.yaml`) structure is documented as a comment within the `main()` function, showing the expected format for Reddit API credentials.
    *   Creates an instance of `StockDataCollector`.
    *   Calls `collector.collect_data()` to start the data collection process.
    *   Includes error handling for the overall data collection process.

### 3.3. Key Libraries Used

*   **`praw`:**  Python Reddit API Wrapper, used to interact with the Reddit API.
*   **`pandas (pd)`:** For data manipulation and storage in DataFrames.
*   **`logging`:** For logging information, warnings, and errors.
*   **`datetime`, `timezone`:** For handling timestamps and timezones.
*   **`time`:** For pausing execution to respect rate limits.
*   **`os`:** (Although imported, not directly used in the provided snippet, might be used in a fuller version for path manipulation).
*   **`typing.List, Dict, Any`:** For type hinting to improve code readability and maintainability.
*   **`yaml`:** For loading configuration settings from YAML files.
*   **`pathlib.Path`:** For working with file paths.
*   **`backoff`:** For implementing exponential backoff retry logic to handle transient API errors.
*   **`json`:** For saving metadata in JSON format.

### 3.4. Configuration

*   **`config.yaml` File:**  Reddit API credentials (`client_id`, `client_secret`, `user_agent`) are loaded from a `config.yaml` file. This file *must* be created in the same directory as `stock_collector_reddit.py` and should have the structure documented in the `main()` function comment.  **You need to obtain your own Reddit API credentials by creating a Reddit app.**
*   **`TRACKED_STOCKS` (Class Attribute):** Similar to `stock_collector_yfinance.py`, this dictionary defines the stocks to track and can be modified within the script.
*   **`SUBREDDITS` (Class Attribute):**  The list of subreddits to scrape is defined as `SUBREDDITS` and can be adjusted within the script to target different communities.

### 3.5. Execution Instructions

1.  **Install Required Libraries:**  Install the necessary libraries using `pip install praw pandas pyyaml backoff`.
2.  **Create `config.yaml`:** Create a file named `config.yaml` in the same directory as `stock_collector_reddit.py`. Populate it with your Reddit API credentials as shown in the example in the `main()` function comment. **Replace `"your_client_id"`, `"your_client_secret"`, and `"StockDataCollector/1.0"` with your actual credentials and a descriptive user agent.**
3.  **Run the Script:** Execute the script from your terminal:

    ```bash
    python stock_collector_reddit.py
    ```

4.  **Check Output:** After running, CSV files (`stock_mentions_YYYYMMDD_HHMMSS.csv`) containing Reddit post data and corresponding metadata JSON files will be saved in the `data` directory. Check `stock_collector.log` for logs and potential errors.

### 3.6. Output

*   **CSV Files:** Reddit post data CSV files are saved as `data/stock_mentions_{YYYYMMDD_HHMMSS}.csv`. Each file contains details of Reddit posts mentioning tracked stocks, including post content, score, comments, and timestamps.
*   **JSON Metadata Files:** Metadata for each collection run is saved as `data/metadata_{YYYYMMDD_HHMMSS}.json`.

### 3.7. Error Handling and Logging

*   **Error Logging:**  Uses `logging` to record errors, warnings, and informational messages to both `stock_collector.log` and the console.
*   **Reddit API Error Handling with Backoff:** The `_fetch_subreddit_posts` function uses the `@backoff.on_exception` decorator. This automatically retries fetching posts if `praw.exceptions.PRAWException` or a general `Exception` occurs, using exponential backoff between retries. This is crucial for handling rate limits and transient network errors from the Reddit API.
*   **Configuration Error Handling:**  The `_load_config` method handles `FileNotFoundError` if the `config.yaml` file is not found.
*   **Reddit Client Initialization Error Handling:** `_init_reddit_client` handles potential errors during Reddit client initialization.

### 3.8. Potential Improvements

*   **Sentiment Scoring within Collector:** Integrate sentiment analysis (e.g., using VADER or TextBlob) directly within the collector script to add sentiment scores to the collected Reddit data before saving. This would reduce processing load in the `data_processor.py` script.
*   **More Granular Stock Mention Identification:** Enhance `_identify_stock_mentions` to be more robust and potentially use NLP techniques to improve accuracy in identifying stock tickers, especially in noisy text.
*   **Comment Collection:** Extend the script to also collect comments from relevant Reddit posts to get a richer dataset of sentiment and discussions.
*   **Data Filtering at Collection Time:** Add more filtering criteria during data collection (e.g., filter by keywords, minimum post score, etc.) to collect more targeted and relevant data.
*   **Dynamic Subreddit List:**  Allow the `SUBREDDITS` list to be configurable via a file or command-line arguments.
*   **Rate Limit Monitoring:** Implement more sophisticated rate limit monitoring and handling to optimize data collection speed while staying within Reddit API limits.

---

## 4. Script 3: `data_processor.py` - Data Processing and Feature Engineering

*(This section is largely similar to Section 3 of the previous documentation, but included here for completeness within this full document)*

### 4.1. Purpose and Functionality

The `data_processor.py` script processes and combines the historical stock price data (collected by `stock_collector_yfinance.py`) and Reddit sentiment data (collected by `stock_collector_reddit.py`). It engineers features from both datasets and prepares the final dataset for training the stock prediction model.

### 4.2. Code Structure and Key Components

*(Details of classes, methods, and functions are already well described in Section 3 of the previous "Stock Price Direction Prediction Model Documentation".  Refer to that section for a detailed breakdown of `StockDataProcessor` class, its methods like `_validate_data_quality`, `_load_stock_data`, `_load_sentiment_data`, `_process_sentiment_features`, `_calculate_price_features`, `_calculate_rsi`, `_merge_data`, `_create_target_variables`, and `process_data`.)*

### 4.3. Key Libraries Used

*   **`pandas (pd)`:** For data manipulation, merging, and feature engineering.
*   **`numpy (np)`:** For numerical operations, array manipulations, and mathematical functions used in feature calculations.
*   **`logging`:** For logging processing steps, warnings, and errors.
*   **`datetime`, `timedelta`, `timezone`:** For date and time handling.
*   **`pathlib.Path`:** For file path management.
*   **`json`:** For saving metadata in JSON format.
*   **`sklearn.preprocessing.StandardScaler`:** For feature scaling.
*   **`vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer`:** For sentiment analysis (although, in the provided `data_processor.py`, it's imported but *not* used.  Sentiment features are currently based on aggregated Reddit metrics, not direct text sentiment analysis within this script itself. This is a point of potential improvement).

### 4.4. Configuration

*   **`TRACKED_STOCKS` (Class Attribute):**  Used to ensure consistent tracking of stocks across all scripts.
*   **`lookback_days` (Parameter in `process_data`):**  Controls how many days of historical data are used for processing (default 90 days).

### 4.5. Execution Instructions

1.  **Install Required Libraries:** Ensure you have installed required libraries (listed in section 4.3) using `pip install pandas numpy scikit-learn vaderSentiment`. (Note: `vaderSentiment` is imported but not currently used for direct sentiment scoring within this script's feature engineering).
2.  **Ensure Data is Collected:** Make sure you have run `stock_collector_yfinance.py` and potentially `stock_collector_reddit.py` (if you intend to use sentiment data) to collect the raw data first.
3.  **Run the Script:** Execute the data processing script:

    ```bash
    python data_processor.py
    ```

4.  **Check Output:**  Processed data CSV files and metadata JSON files will be saved in the `processed_data` directory. Check `data_processor.log` for processing logs and errors.

### 4.6. Output

*   **CSV Files:** Processed data CSV files are saved in `processed_data/{Ticker}_processed_{YYYYMMDD}.csv`. These files contain the combined stock price and sentiment data with engineered features and target variables.
*   **JSON Metadata Files:**  Metadata files corresponding to each processed data CSV are saved as `processed_data/{Ticker}_processed_{YYYYMMDD}.json`.

### 4.7. Error Handling and Logging

*(Error handling and logging mechanisms are described in detail in Section 3 of the previous "Stock Price Direction Prediction Model Documentation".)*

### 4.8. Potential Improvements

*(Potential improvements are described in detail in Section 3 of the previous "Stock Price Direction Prediction Model Documentation". These include: More comprehensive sentiment analysis, advanced feature engineering, data validation enhancements, configuration improvements, and pipeline optimizations.)*

---

## 5. Script 4: `model_trainer.py` - Model Training and Evaluation

*(This section is largely similar to Sections 4-11 of the previous documentation, but included here for completeness.)*

### 5.1. Purpose and Functionality

The `model_trainer.py` script is responsible for training a machine learning model to predict the next-day direction of stock prices. It uses the processed data generated by `data_processor.py`, trains an XGBoost classifier, optimizes hyperparameters, evaluates the model's performance, and saves the trained model and related artifacts.

### 5.2. Code Structure and Key Components

*(Details of classes, methods, and functions are already well described in Sections 4-11 of the previous "Stock Price Direction Prediction Model Documentation". Refer to those sections for a detailed breakdown of `StockPredictor` class, its methods like `_load_data`, `_prepare_features`, `_create_pipeline`, `_optimize_hyperparameters`, `_evaluate_model`, `_save_model_artifacts`, `train`, and `predict`, as well as the `FEATURE_GROUPS` and `param_space`.)*

### 5.3. Key Libraries Used

*   **`pandas (pd)`:** For data loading and manipulation.
*   **`numpy (np)`:** For numerical operations and calculations.
*   **`logging`:** For logging training progress, errors, and results.
*   **`datetime`, `timedelta`:** For timestamping and time-related operations.
*   **`pathlib.Path`:** For file path management.
*   **`json`:** For saving metadata in JSON format.
*   **`joblib`:** For saving and loading trained models.
*   **`sklearn.preprocessing.StandardScaler`:** For feature scaling.
*   **`sklearn.model_selection.TimeSeriesSplit`:** For time series cross-validation.
*   **`sklearn.metrics`:** For model evaluation metrics (accuracy, precision, recall, F1-score, classification report, confusion matrix).
*   **`xgboost (xgb)`:** For the XGBoost gradient boosting algorithm.
*   **`optuna`:** For hyperparameter optimization.
*   **`shap`:** For model explainability using SHAP values.
*   **`matplotlib.pyplot (plt)`:** For generating SHAP summary plots.
*   **`imblearn.over_sampling.SMOTE`:** For handling class imbalance using SMOTE.
*   **`imblearn.pipeline.Pipeline as ImbPipeline`:** For creating a pipeline that includes SMOTE.

### 5.4. Configuration

*   **`FEATURE_GROUPS` (Class Attribute):**  Defines groups of features (price and sentiment) to be used in the model. Can be modified to adjust feature selection.
*   **`param_space` (Class Attribute):**  Defines the hyperparameter search space for Optuna optimization. Can be adjusted to modify the ranges of hyperparameters explored.
*   **`processed_data_dir`, `models_dir` (Constructor Parameters):**  Specify the directories for processed data and saved models, allowing for customization of data and model storage locations.
*   **`n_trials` (Parameter in `train` method):** Controls the number of Optuna trials for hyperparameter optimization (default 100).

### 5.5. Execution Instructions

1.  **Install Required Libraries:** Ensure you have installed all required libraries (listed in section 5.3) using `pip install pandas numpy scikit-learn joblib xgboost optuna shap matplotlib imbalanced-learn`.
2.  **Ensure Data is Processed:** Run `data_processor.py` *before* running `model_trainer.py` to generate the processed data files.
3.  **Run the Script:** Execute the model training script:

    ```bash
    python model_trainer.py
    ```

4.  **Check Output:** Trained models, SHAP plots, feature importance CSVs, and metadata JSON files will be saved in the `models` directory, organized by stock ticker and timestamped directories. Training results summary will be saved in `training_results.json`. Check `model_trainer.log` for training logs and errors.

### 5.6. Output

*   **Model Artifacts (per stock):** Saved in `models/{Ticker}/{YYYYMMDD_HHMMSS}/`:
    *   `model.joblib`: Trained XGBoost model pipeline.
    *   `shap_summary.png`: SHAP summary plot.
    *   `feature_importance.csv`: Feature importance scores.
    *   `metadata.json`: Model training metadata.
*   **Training Results Summary:** `training_results.json` in the project root directory, summarizing the training status for each stock.

### 5.7. Error Handling and Logging

*(Error handling and logging mechanisms are described in detail in Sections 4-11 of the previous "Stock Price Direction Prediction Model Documentation".)*

### 5.8. Potential Improvements

*(Potential improvements are described in detail in Section 10 of the previous "Stock Price Direction Prediction Model Documentation". These include: Expanding feature set, advanced sentiment analysis, ensemble models, deep learning models, real-time data pipeline, risk management integration, backtesting, and continuous monitoring.)*

---

## 6. Conclusion

This detailed document provides a complete overview of the stock price direction prediction system, covering data collection, data processing, model training, and evaluation. Each script's purpose, architecture, usage, and potential improvements have been described in depth. This documentation serves as a guide for understanding, using, and further developing this stock prediction system.  Remember to consult the individual "Potential Improvements" sections within each script's description for specific ideas on how to enhance each component of the system.

---

**End of Document**
