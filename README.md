📈 Stock Market Forecasting using LSTM, ARIMA, SARIMA & Prophet
This project predicts stock market trends using various forecasting models, including:

🧠 LSTM (Long Short-Term Memory) – Deep Learning
📊 ARIMA (AutoRegressive Integrated Moving Average)
📈 SARIMA (Seasonal ARIMA)
🔮 Prophet – Forecasting tool by Meta

The models are applied to real-world stock data (apple_data.csv) for Apple Inc.

🗂️ Project Structure
📁 stock-market-forecasting/
│
├── models/                          # (Optional) For future model-specific scripts
├── venv/                            # Virtual environment
│
├── app.py                           # Streamlit app for interactive forecasting
├── apple_data.csv                   # Historical stock price dataset
├── requirements.txt                 # Required Python packages
├── runtime.txt                      # Python version for deployment (e.g., Streamlit Cloud)
├── Stock_market_forecasting.ipynb   # Jupyter notebook with full model implementation
└── README.md                        # You're reading it! 📘


⚙️ Setup Instructions
1. Clone the repository
git clone https://github.com/your-username/stock-market-forecasting.git
cd stock-market-forecasting

2. Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt


🚀 How to Use
👉 Run the notebook
jupyter notebook Stock_market_forecasting.ipynb

This walks through:

Data loading and preprocessing
Training each model (LSTM, ARIMA, SARIMA, Prophet)
Evaluation metrics and visualizations

👉 OR launch the Streamlit app
streamlit run app.py

This gives you an interactive UI to visualize forecasts from each model.

📊 Model Comparison



Model
Seasonality
Trend
Handles Noise
Training Time



ARIMA
❌
✅
❌
⏱️ Fast


SARIMA
✅
✅
❌
⏱️ Fast


Prophet
✅
✅
✅ (basic)
⏱️ Medium


LSTM
✅
✅
✅ (deep)
🐢 Slow (needs tuning)



📈 Sample Output

Line plots of actual vs predicted stock prices
Model performance scores (RMSE, MAE, R²)
Forecasts for next 30 days


📚 Dependencies

tensorflow – for LSTM
statsmodels – for ARIMA/SARIMA
prophet
pandas, numpy, matplotlib, plotly
streamlit – for web app


📌 Future Enhancements

Integrate technical indicators (MACD, RSI, etc.)
Deploy on Streamlit Cloud or Hugging Face Spaces
Add model selection toggle in UI
Include backtesting and rolling forecast


🙋‍♂️ Author
Kishan Kumar📧 kishankumar1047@gmail.com🔗 LinkedIn🐙 GitHub


This project showcases the power of combining classical and deep learning techniques for real-world time series forecasting.
