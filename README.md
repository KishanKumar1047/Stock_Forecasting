
# 📈 Stock Market Forecasting using LSTM, ARIMA, SARIMA & Prophet

This project predicts stock market trends using multiple forecasting models, including:

- 🧠 **LSTM** (Long Short-Term Memory) – Deep Learning
- 📊 **ARIMA** (AutoRegressive Integrated Moving Average)
- 📈 **SARIMA** (Seasonal ARIMA)
- 🔮 **Prophet** – Forecasting tool by Meta

The models are applied to real-world historical stock data of **Apple Inc.** (`apple_data.csv`).

---

## 🗂️ Project Structure

```

📁 stock-market-forecasting/
│
├── models/                          # (Optional) For future model-specific scripts
├── venv/                            # Virtual environment
│
├── app.py                           # Streamlit app for interactive forecasting
├── apple\_data.csv                   # Historical stock price dataset
├── requirements.txt                 # Required Python packages
├── runtime.txt                      # Python version for deployment (e.g., Streamlit Cloud)
├── Stock\_market\_forecasting.ipynb   # Jupyter notebook with full model implementation
└── README.md                        # You're reading it! 📘

````

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/stock-market-forecasting.git
cd stock-market-forecasting
````

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Use

### 👉 Run the Jupyter notebook

```bash
jupyter notebook Stock_market_forecasting.ipynb
```

You'll walk through:

* ✅ Data loading and preprocessing
* ✅ Model training (LSTM, ARIMA, SARIMA, Prophet)
* ✅ Evaluation metrics and visualizations

### 👉 Or launch the Streamlit app

```bash
streamlit run app.py
```

This provides an **interactive UI** to visualize forecasts from each model.

---

## 📊 Model Comparison

| Model   | Seasonality | Trend | Handles Noise     | Training Time          |
| ------- | ----------- | ----- | ----------------- | ---------------------- |
| ARIMA   | ❌           | ✅     | ❌                 | ⏱️ Fast                |
| SARIMA  | ✅           | ✅     | ❌                 | ⏱️ Fast                |
| Prophet | ✅           | ✅     | ✅ (basic)         | ⏱️ Medium              |
| LSTM    | ✅           | ✅     | ✅ (deep learning) | 🐢 Slow (needs tuning) |

---

## 📈 Sample Output

* 📉 Line plots of actual vs. predicted stock prices
* 📋 Model performance metrics: RMSE, MAE, R²
* 🔮 Forecasts for the next 30 days

---

## 📚 Dependencies

* `tensorflow` – for LSTM
* `statsmodels` – for ARIMA/SARIMA
* `prophet` – for time series forecasting
* `pandas`, `numpy`, `matplotlib`, `plotly` – for data handling and visualization
* `streamlit` – for interactive app interface

---

## 📌 Future Enhancements

* ➕ Integrate technical indicators (MACD, RSI, etc.)
* ☁️ Deploy on **Streamlit Cloud** or **Hugging Face Spaces**
* 🔘 Add model selection toggle in the UI
* 🔁 Include backtesting and rolling forecast support

---

## 🙋‍♂️ Author

**Kishan Kumar**
📧 [kishankumar1047@gmail.com](mailto:kishankumar1047@gmail.com)
🔗 [LinkedIn](https://linkedin.com/in/kishankumar098)
🐙 [GitHub](https://github.com/kishankumar1047)

---

> *This project showcases the power of combining classical statistical models and deep learning for real-world time series forecasting.*
