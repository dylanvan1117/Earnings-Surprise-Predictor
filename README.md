README outline
# Earnings Surprise Predictor

ML model that predicts whether a company will beat or miss Wall Street earnings estimates.

## What it does
Enter any S&P 500 ticker → model outputs beat probability using financial ratios, analyst estimate revision trends, historical beat rates, and SEC filing sentiment.

## Motivation
"Earnings surprise" is one of the strongest short-term stock price drivers. This model tests whether public signals (financial ratios, analyst revisions, filing tone) contain predictive signal beyond the consensus estimate itself.

## Model performance (Logistic Regression)
- AUC-ROC: .6721
- Directional accuracy on test set: 70%
- Test period: 2022–2024 earnings events
- Training data: ~3327 earnings events across S&P 500



## Tech stack
Python, LightGBM, yfinance, SEC EDGAR API, Streamlit, Plotly

## Setup
```bash
pip install -r requirements.txt
python src/data_pull.py    # pulls historical earnings data
python src/features.py
python src/sentiment.py    # pulls SEC filings
python src/train.py
streamlit run src/app.py
```

## Demo
<img width="1283" height="632" alt="image" src="https://github.com/user-attachments/assets/65af89f5-0e05-4bd1-9f7b-ddb0dfe7e7b2" /> <img width="1396" height="873" alt="image" src="https://github.com/user-attachments/assets/330c9425-f543-42e4-9be1-594d29dd8949" /> <img width="1368" height="353" alt="image" src="https://github.com/user-attachments/assets/fb8a71f5-8015-447c-aeb9-162b96586c62" />




## Disclaimer
For educational purposes only. Not financial advice.
