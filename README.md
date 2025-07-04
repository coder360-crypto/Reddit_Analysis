# Reddit Analysis Project

This project analyzes Reddit data to extract insights about industry trends, user engagement, and popular topics. It scrapes data from Reddit, processes it, and generates various visualizations and reports.

## Features

*   **Data Scraping**: Scrapes Reddit data using both an API-based approach (`reddit_analyser.py`) and a Selenium-based approach (`reddit_analyser_Selenium.py`).
*   **Data Processing**: Processes the scraped data to identify mentions of industries and performs text analysis.
*   **Visualization**: Generates various plots and interactive visualizations to represent the findings.
    *   Industry Distribution Pie Chart
    *   N-gram Analysis
    *   Temporal Analysis of posts
    *   Word Clouds
    *   Subreddit-level analysis
*   **Reporting**: Saves the processed data into CSV files and generated plots as PNG and HTML files.

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/coder360-crypto/Reddit_Analysis.git
    cd Reddit_Analysis
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    uv pip install -r requirements.txt
    ```

3.  **Configure Scraping (if necessary):**
    The scraping scripts might require API credentials or specific configurations. Check the top of `reddit_analyser.py` or `reddit_analyser_Selenium.py` for any required setup.

4.  **Run the analysis:**
    To run the main analysis script:
    ```bash
    python reddit_analyser.py
    ```
    Or the Selenium-based scraper:
    ```bash
    python reddit_analyser_Selenium.py
    ```

## Outputs

The analysis generates the following files:

*   `reddit_industry_analysis_processed_*.csv`: Processed data with industry mentions.
*   `interactive_industry_pie.html`: Interactive pie chart of industry distribution.
*   `interactive_scatter.html`: Interactive scatter plot.
*   `industry_distribution.png`: Static pie chart of industry distribution.
*   `ngram_analysis.png`: N-gram analysis visualization.
*   `subreddit_analysis.png`: Subreddit-level analysis visualization.
*   `temporal_analysis.png`: Temporal analysis of posts.
*   `text_statistics.png`: Text statistics visualization.
*   `wordcloud.png`: Word cloud of popular terms.
*   `reddit_scraped.json` / `reddit_selenium_scraped.json`: Raw scraped data.
