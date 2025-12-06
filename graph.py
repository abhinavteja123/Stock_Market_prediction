import importlib
import importlib.util

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import yfinance as yf
from datetime import datetime

tf_spec = importlib.util.find_spec("tensorflow")
if tf_spec is not None:
    tf = importlib.import_module("tensorflow")
    keras_models = importlib.import_module("tensorflow.keras.models")
    keras_layers = importlib.import_module("tensorflow.keras.layers")
    keras_callbacks = importlib.import_module("tensorflow.keras.callbacks")
    Sequential = getattr(keras_models, "Sequential")
    LSTM = getattr(keras_layers, "LSTM")
    Dense = getattr(keras_layers, "Dense")
    Dropout = getattr(keras_layers, "Dropout")
    EarlyStopping = getattr(keras_callbacks, "EarlyStopping")
    TENSORFLOW_AVAILABLE = True
else:
    tf = None
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None
    EarlyStopping = None
    TENSORFLOW_AVAILABLE = False


def find_optimal_threshold(y_true, probas, metric_priority="f1"):
    """Determine the decision threshold that maximizes the chosen metric."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best = {
        "threshold": 0.5,
        "f1": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "metric": 0.0
    }
    for threshold in thresholds:
        preds = (probas >= threshold).astype(int)
        accuracy = metrics.accuracy_score(y_true, preds)
        f1 = metrics.f1_score(y_true, preds, zero_division=0)
        precision = metrics.precision_score(y_true, preds, zero_division=0)
        recall = metrics.recall_score(y_true, preds, zero_division=0)
        metric_value = f1 if metric_priority == "f1" else accuracy
        if metric_value > best["metric"]:
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "metric": float(metric_value)
            }
    return best


def compute_walk_forward_metrics(model_pipeline, X, y, max_splits=5, min_samples_per_split=120):
    """Run a walk-forward (time-series) evaluation for tabular models."""
    if len(X) < min_samples_per_split * 2:
        return None

    possible_splits = len(X) // min_samples_per_split
    n_splits = min(max_splits, max(2, possible_splits))
    if n_splits < 2:
        return None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    records = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < min_samples_per_split or len(val_idx) == 0:
            continue

        cloned_model = clone(model_pipeline)
        cloned_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probas = cloned_model.predict_proba(X.iloc[val_idx])[:, 1]
        preds = (probas >= 0.5).astype(int)

        records.append({
            "fold": fold_idx + 1,
            "train_samples": int(len(train_idx)),
            "valid_samples": int(len(val_idx)),
            "accuracy": float(metrics.accuracy_score(y.iloc[val_idx], preds)),
            "f1": float(metrics.f1_score(y.iloc[val_idx], preds, zero_division=0)),
            "roc_auc": float(metrics.roc_auc_score(y.iloc[val_idx], probas))
        })

    if not records:
        return None

    df_records = pd.DataFrame(records)
    summary = {
        "folds": records,
        "mean_accuracy": float(df_records["accuracy"].mean()),
        "mean_f1": float(df_records["f1"].mean()),
        "mean_roc_auc": float(df_records["roc_auc"].mean()),
        "n_folds": int(len(records))
    }
    return summary


def extract_feature_importance(estimator, feature_names):
    """Return normalized feature importance scores for supported estimators."""
    if estimator is None:
        return None

    model = estimator
    if isinstance(estimator, Pipeline):
        model = estimator.named_steps.get("model", estimator)

    if isinstance(model, VotingClassifier):
        aggregated = {}
        count = 0
        for sub_estimator in model.estimators_:
            if isinstance(sub_estimator, tuple):
                sub_estimator = sub_estimator[1]
            sub_importance = extract_feature_importance(sub_estimator, feature_names)
            if sub_importance:
                count += 1
                for feat, score in sub_importance.items():
                    aggregated[feat] = aggregated.get(feat, 0.0) + float(score)
        if count == 0:
            return None
        for feat in aggregated:
            aggregated[feat] /= count
        return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))

    if isinstance(model, Pipeline):
        model = model.named_steps.get("model", model)

    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        importances = np.abs(coef.astype(float))
    else:
        return None

    if importances.size != len(feature_names):
        return None

    return dict(sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True))


# Custom CSS for modern styling
st.markdown("""
<style>
    /* v2.0 - Cache busting */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background-color: #3498db !important;
        color: white !important;
        border-radius: 5px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        font-weight: bold !important;
    }
    .stButton>button:hover {
        background-color: #2980b9 !important;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 1px solid #dee2e6;
    }
    .sidebar .sidebar-content h2 {
        color: #2c3e50 !important;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content .stTextInput input {
        background-color: white !important;
        color: #2c3e50 !important;
        border: 1px solid #ced4da !important;
        border-radius: 6px !important;
        padding: 0.5rem 0.75rem !important;
        font-size: 0.9rem !important;
        transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out !important;
    }
    .sidebar .sidebar-content .stTextInput input:focus {
        border-color: #3498db !important;
        box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25) !important;
        outline: 0 !important;
    }
    .sidebar .sidebar-content .stDateInput input {
        background-color: white !important;
        color: #2c3e50 !important;
        border: 1px solid #ced4da !important;
        border-radius: 6px !important;
        padding: 0.5rem 0.75rem !important;
        font-size: 0.9rem !important;
        cursor: pointer !important;
        transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out !important;
    }
    .sidebar .sidebar-content .stDateInput input:focus {
        border-color: #3498db !important;
        box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25) !important;
        outline: 0 !important;
    }
    .sidebar .sidebar-content label {
        color: #495057 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        margin-bottom: 0.25rem !important;
    }
    .sidebar .sidebar-content hr {
        border: none;
        height: 1px;
        background-color: #dee2e6;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: #f8f9fa;
        color: #6b7280;
        border-radius: 8px 8px 0 0;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 500;
        margin-right: 4px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
        color: #495057;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #2c3e50;
        border: 1px solid #34495e;
        border-radius: 0 8px 8px 8px;
        padding: 2rem;
        color: #ffffff;
    }
    /* Dataframe styling for dark theme */
    .stDataFrame {
        background-color: #2c3e50 !important;
    }
    .stDataFrame [data-testid="stDataFrame"] {
        background-color: #2c3e50 !important;
        border: 1px solid #34495e !important;
        border-radius: 8px !important;
    }
    .stDataFrame [data-testid="stDataFrame"] thead th {
        background-color: #34495e !important;
        color: #ffffff !important;
        border-bottom: 1px solid #495057 !important;
    }
    .stDataFrame [data-testid="stDataFrame"] tbody td {
        background-color: #2c3e50 !important;
        color: #ffffff !important;
        border-bottom: 1px solid #34495e !important;
    }
    .stDataFrame [data-testid="stDataFrame"] tbody tr:hover {
        background-color: #34495e !important;
    }
    /* Pyplot chart containers */
    .stPlotlyChart, .stPyplot {
        background-color: #2c3e50 !important;
        border: 1px solid #34495e !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    /* General component containers to prevent white backgrounds */
    .stContainer, .element-container {
        background-color: transparent !important;
    }
    /* Override any remaining white backgrounds */
    [data-testid="stVerticalBlock"] {
        background-color: transparent !important;
    }
    /* Matplotlib figure containers */
    .stPlotlyChart div, .stPyplot div {
        background-color: #2c3e50 !important;
    }
    /* Ensure all text in dark areas is visible */
    .stTabs [data-baseweb="tab-panel"] * {
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-panel"] h1, 
    .stTabs [data-baseweb="tab-panel"] h2, 
    .stTabs [data-baseweb="tab-panel"] h3,
    .stTabs [data-baseweb="tab-panel"] h4,
    .stTabs [data-baseweb="tab-panel"] h5,
    .stTabs [data-baseweb="tab-panel"] h6 {
        color: #ffffff !important;
    }
    /* Form elements in dark theme */
    .stTabs [data-baseweb="tab-panel"] .stSelectbox div[data-baseweb="select"] {
        background-color: #34495e !important;
        border: 1px solid #495057 !important;
    }
    .stTabs [data-baseweb="tab-panel"] .stSelectbox div[data-baseweb="select"]:hover {
        border-color: #3498db !important;
    }
    .selected-params-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        transition: box-shadow 0.2s ease;
    }
    .selected-params-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
    }
    .selected-params-title {
        color: #6b7280;
        font-size: 0.75rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 0.875rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .params-top-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.625rem;
        margin-bottom: 0.625rem;
    }
    .params-bottom-row {
        display: grid;
        grid-template-columns: 1fr;
        gap: 0.625rem;
    }
    .param-item {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 0.75rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e9ecef;
        transition: all 0.2s ease;
    }
    .param-item:hover {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-color: #3498db;
        transform: translateY(-1px);
    }
    .param-label {
        font-size: 0.6rem;
        color: #9ca3af;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.075em;
        margin-bottom: 0.375rem;
    }
    .param-value {
        font-weight: 700;
        font-size: 0.9rem;
        color: #1f2937;
        letter-spacing: -0.025em;
    }
    .metrics-container {
        background: #2c3e50;
        border: 1px solid #34495e;
        border-radius: 12px;
        padding: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        overflow: hidden;
    }
    .metric-card-enhanced {
        background: #2c3e50;
        padding: 1rem 0.75rem;
        text-align: center;
        border-bottom: 1px solid #34495e;
        transition: background-color 0.15s ease;
    }
    .metric-card-enhanced:last-child {
        border-bottom: none;
    }
    .metric-card-enhanced:hover {
        background: #34495e;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.25rem;
        line-height: 1;
        letter-spacing: -0.025em;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #bdc3c7;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📈 Stock Price Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown("---")

# Stock Symbol Input
symbol = st.sidebar.text_input("Stock Symbol", value="INFY.NS", help="Enter stock ticker (e.g., AAPL, GOOGL, INFY.NS)")

# Date Range Inputs
start_date = st.sidebar.date_input("Start Date", value=datetime(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime(2025, 11, 15))

st.sidebar.markdown("---")
validation_split_pct = st.sidebar.slider(
    "Validation Split (%)", min_value=10, max_value=30, value=20, step=5,
    help="Percentage of the most recent data reserved for validation."
)

st.sidebar.markdown("### 🧠 LSTM Settings")
sequence_length = st.sidebar.slider(
    "Lookback Window (Days)", min_value=15, max_value=120, value=60, step=5,
    help="Number of past trading days used by the LSTM to predict the next move."
)
lstm_epochs = st.sidebar.slider(
    "Training Epochs", min_value=20, max_value=200, value=80, step=10,
    help="Higher epochs can improve learning but take longer and risk overfitting."
)
lstm_batch_size = st.sidebar.selectbox(
    "Batch Size", options=[16, 32, 64, 128], index=1,
    help="Batch size for LSTM training."
)
if not TENSORFLOW_AVAILABLE:
    st.sidebar.warning("TensorFlow not available in this environment. LSTM training will be skipped.")

# Download Button
if st.sidebar.button("🚀 Download & Analyze Data", type="primary"):
    with st.spinner("📥 Downloading stock data..."):
        df = yf.download(symbol, start=start_date, end=end_date)

        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten MultiIndex columns
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)

        # Check if we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.error("This stock may not have complete price data. Try a different symbol.")
            st.stop()

        st.session_state['df'] = df
        st.success(f"✅ Data downloaded successfully! Shape: {df.shape}")
        st.write(f"Columns: {list(df.columns)}")

if 'df' in st.session_state:
    df = st.session_state['df']

    # Data Overview Section
    st.markdown('<h2 class="section-header">📊 Data Overview</h2>', unsafe_allow_html=True)

    # Enhanced selected parameters display
    st.markdown(f'''
    <div class="selected-params-card">
        <div class="selected-params-title">⚙️ Analysis Configuration</div>
        <div class="params-top-row">
            <div class="param-item">
                <div class="param-label">Stock Symbol</div>
                <div class="param-value">{symbol}</div>
            </div>
            <div class="param-item">
                <div class="param-label">Start Date</div>
                <div class="param-value">{start_date.strftime('%Y-%m-%d')}</div>
            </div>
            <div class="param-item">
                <div class="param-label">End Date</div>
                <div class="param-value">{end_date.strftime('%Y-%m-%d')}</div>
            </div>
        </div>
        <div class="params-bottom-row">
            <div class="param-item">
                <div class="param-label">Duration</div>
                <div class="param-value">{(end_date - start_date).days} days</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Compute real trading days (distinct dates)
    trading_days = df['Date'].nunique()

    # Enhanced metrics display
    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
    st.markdown(f'''
        <div class="metric-card-enhanced">
            <div class="metric-value">{df.shape[0]:,}</div>
            <div class="metric-label">Total Records</div>
        </div>
        <div class="metric-card-enhanced">
            <div class="metric-value">{df.shape[1]}</div>
            <div class="metric-label">Total Features</div>
        </div>
        <div class="metric-card-enhanced">
            <div class="metric-value">{df.isnull().sum().sum()}</div>
            <div class="metric-label">Missing Values</div>
        </div>
        <div class="metric-card-enhanced">
            <div class="metric-value">{trading_days}</div>
            <div class="metric-label">Trading Days</div>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("🔍 View Raw Data"):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📈 Statistical Summary"):
        st.dataframe(df.describe(), use_container_width=True)

    # EDA Section
    st.markdown('<h2 class="section-header">📈 Exploratory Data Analysis</h2>', unsafe_allow_html=True)

    # Close Price Plot
    st.subheader("📊 Close Price Trend")
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df['Close'], linewidth=2, color='#1f77b4')
    ax.set_title(f'{symbol} Close Price Trend', fontsize=16, fontweight='bold')
    ax.set_ylabel('Price (INR)', fontsize=12)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Distribution Plots
    st.subheader("📊 Feature Distributions")
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    for i, col in enumerate(features):
        if i < 5:
            axes[i // 3, i % 3].hist(
                df[col].dropna(), bins=50, alpha=0.7,
                color=colors[i], edgecolor='black', linewidth=0.5
            )
            axes[i // 3, i % 3].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
            axes[i // 3, i % 3].set_xlabel(col, fontsize=10)
            axes[i // 3, i % 3].set_ylabel('Frequency', fontsize=10)
            axes[i // 3, i % 3].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Box Plots
    st.subheader("📦 Box Plots")
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Feature Box Plots', fontsize=16, fontweight='bold')
    for i, col in enumerate(features):
        if i < 5:
            axes[i // 3, i % 3].boxplot(
                df[col].dropna(), patch_artist=True,
                boxprops=dict(facecolor=colors[i], alpha=0.7),
                medianprops=dict(color='black', linewidth=2)
            )
            axes[i // 3, i % 3].set_title(f'{col} Box Plot', fontsize=12, fontweight='bold')
            axes[i // 3, i % 3].set_ylabel(col, fontsize=10)
            axes[i // 3, i % 3].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Feature Engineering
    st.markdown('<h2 class="section-header">🔧 Feature Engineering</h2>', unsafe_allow_html=True)

    df_processed = df.copy()

    # Initial cleaning
    essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    initial_rows = len(df_processed)
    df_processed = df_processed.dropna(subset=essential_cols)
    after_essential_clean = len(df_processed)

    # Returns and momentum (past-only)
    df_processed['return_1'] = df_processed['Close'].pct_change()
    df_processed['return_5'] = df_processed['Close'].pct_change(5)
    df_processed['return_10'] = df_processed['Close'].pct_change(10)
    df_processed['momentum_10'] = df_processed['Close'] / df_processed['Close'].shift(10) - 1

    # Volatility
    df_processed['volatility_10'] = df_processed['return_1'].rolling(window=10).std()

    # Moving averages and MACD (past-only)
    df_processed['MA_10'] = df_processed['Close'].rolling(window=10).mean()
    df_processed['MA_20'] = df_processed['Close'].rolling(window=20).mean()
    ema_12 = df_processed['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_processed['Close'].ewm(span=26, adjust=False).mean()
    df_processed['MACD'] = ema_12 - ema_26

    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df_processed['RSI'] = calculate_rsi(df_processed['Close'])

    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)

    feature_fill_cols = [
        'return_1', 'return_5', 'return_10',
        'momentum_10', 'volatility_10',
        'MA_10', 'MA_20', 'MACD', 'RSI'
    ]

    # ✅ Only forward-fill to avoid using future data, then drop remaining NaNs
    df_processed[feature_fill_cols] = df_processed[feature_fill_cols].fillna(method='ffill')
    df_processed = df_processed.dropna(subset=feature_fill_cols)

    after_feature_clean = len(df_processed)

    df_processed['target'] = np.where(df_processed['Close'].shift(-1) > df_processed['Close'], 1, 0)
    df_processed = df_processed.iloc[:-1]

    st.write("**Data Cleaning Summary:**")
    st.write(f"- Initial data points: {initial_rows}")
    st.write(f"- After removing rows with missing OHLCV: {after_essential_clean}")
    st.write(f"- After feature engineering: {after_feature_clean}")
    st.write(f"- Final dataset: {len(df_processed)} rows")

    df = df_processed

    if df.empty or len(df) < 30:  # Reduced minimum requirement
        st.error(f"❌ Not enough data after feature engineering. Only {len(df)} rows remain after processing.")
        st.error("**Possible causes:**")
        st.error("• Stock has missing price data (gaps in trading)")
        st.error("• Very short date range for technical indicators")
        st.error("• Stock symbol not found or delisted")
        st.error("")
        st.error("**Try:**")
        st.error("• A different stock symbol (AAPL, GOOGL, MSFT)")
        st.error("• A longer date range (1-2 years)")
        st.error("• Check if the stock symbol is correct")
        st.stop()

    st.success(f"✅ Feature engineering completed! Final dataset shape: {df.shape}")

    # Target Distribution
    st.subheader("🎯 Target Distribution")
    
    # Check target values
    st.write(f"**Target value range:** min={df['target'].min()}, max={df['target'].max()}")
    st.write(f"**Target unique values:** {sorted(df['target'].unique())}")
    st.write(f"**Target dtype:** {df['target'].dtype}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#e74c3c', '#27ae60']
    target_counts = df['target'].value_counts()
    st.write(f"**Full dataset:** DOWN={target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.1f}%), UP={target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    df['target'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=colors, startangle=90)
    ax.set_title('Price Movement Distribution', fontsize=14, fontweight='bold')
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("🔗 Feature Correlation")
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = df.drop('Date', axis=1).corr()
    sb.heatmap(
        corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
        cbar_kws={'shrink': 0.8}, square=True, linewidths=0.5, fmt='.2f'
    )
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)

    # Model Training Section
    st.markdown('<h2 class="section-header">🤖 Model Training</h2>', unsafe_allow_html=True)

    features_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'return_1', 'return_5', 'return_10',
        'RSI', 'MA_10', 'MA_20', 'MACD',
        'momentum_10', 'volatility_10'
    ]
    features = df[features_cols]
    target = df['target']

    validation_size = max(int(len(df) * (validation_split_pct / 100)), 1)
    train_size = len(df) - validation_size

    if train_size <= 0:
        st.error("❌ Validation split too large for the available data. Reduce the validation percentage or download more history.")
        st.stop()

    X_train = features.iloc[:train_size]
    X_valid = features.iloc[train_size:]
    Y_train = target.iloc[:train_size]
    Y_valid = target.iloc[train_size:]

    st.write(f"Training set shape: {X_train.shape}")
    st.write(f"Validation set shape: {X_valid.shape}")
    
    # Show class distribution
    train_counts = Y_train.value_counts()
    valid_counts = Y_valid.value_counts()
    st.write(f"**Training class distribution:** DOWN={train_counts.get(0, 0)}, UP={train_counts.get(1, 0)}")
    st.write(f"**Validation class distribution:** DOWN={valid_counts.get(0, 0)}, UP={valid_counts.get(1, 0)}")
    
    if len(train_counts) < 2 or len(valid_counts) < 2:
        st.error("⚠️ WARNING: One or both sets have only one class! This will cause prediction issues.")
        st.error("Try adjusting the validation split or using a longer date range.")

    # ✅ Simplified XGBoost with better class handling
    class_counts = Y_train.value_counts()
    n_down = class_counts.get(0, 0)
    n_up = class_counts.get(1, 0)
    scale_pos_weight = n_down / n_up if n_up > 0 else 1.0
    
    st.write(f"📊 **Class Balance Info:**")
    st.write(f"- Training DOWN: {n_down}, UP: {n_up}")
    st.write(f"- Ratio DOWN/UP: {scale_pos_weight:.4f}")
    
    # Much simpler, more robust parameters
    xgb_params = {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 2022,
        "use_label_encoder": False,
        "eval_metric": 'logloss'
    }

    classical_templates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight='balanced', random_state=2022))
        ]),
        "SVM (Poly)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel='poly', probability=True, class_weight='balanced', random_state=2022))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),  # Add scaling for XGBoost too
            ("model", XGBClassifier(**xgb_params))
        ]),
        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=2022
            ))
        ]),
        "Voting Ensemble": Pipeline([
            ("scaler", StandardScaler()),
            ("model", VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000, class_weight='balanced', random_state=2022)),
                    ("xgb", XGBClassifier(**xgb_params)),
                    ("rf", RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_leaf=5,
                        class_weight='balanced',
                        random_state=2022
                    ))
                ],
                voting='soft'
            ))
        ])
    }
    classical_models = list(classical_templates.items())

    lstm_feature_cols = ['Close', 'Volume', 'RSI', 'MA_20', 'volatility_10']

    def create_lstm_sequences(data_array, labels_array, window):
        X_seq, y_seq = [], []
        for idx in range(window, len(data_array)):
            X_seq.append(data_array[idx - window:idx])
            y_seq.append(labels_array[idx])
        return np.array(X_seq), np.array(y_seq)

    if st.button("🎯 Train Models", type="primary"):
        # Pre-training validation
        st.write("### 🔍 Pre-Training Validation")
        
        # Check for data quality issues
        if X_train.isnull().any().any():
            st.error("❌ Training data contains NaN values!")
            st.stop()
        
        if X_valid.isnull().any().any():
            st.error("❌ Validation data contains NaN values!")
            st.stop()
            
        # Check class distribution
        train_class_dist = Y_train.value_counts()
        if len(train_class_dist) < 2:
            st.error(f"❌ Training set has only ONE class: {train_class_dist.to_dict()}")
            st.error("Cannot train a binary classifier with only one class!")
            st.stop()
        
        st.success(f"✅ Data validation passed!")
        st.write(f"- Features have no missing values")
        st.write(f"- Both classes present in training set")
        
        with st.spinner("🔄 Training machine learning and deep learning models..."):
            trained_models = []
            results = []
            walk_forward_table = []

            # Classical models
            for name, pipeline_template in classical_models:
                pipeline_model = clone(pipeline_template)
                pipeline_model.fit(X_train, Y_train)
                train_probs = pipeline_model.predict_proba(X_train)[:, 1]
                valid_probs = pipeline_model.predict_proba(X_valid)[:, 1]

                # Debug: Check probability distribution
                st.write(f"**{name} - Probability Stats:**")
                st.write(f"- Min prob: {valid_probs.min():.4f}, Max prob: {valid_probs.max():.4f}")
                st.write(f"- Mean prob: {valid_probs.mean():.4f}, Median prob: {np.median(valid_probs):.4f}")
                st.write(f"- Probs > 0.5: {(valid_probs > 0.5).sum()}/{len(valid_probs)}")
                
                train_auc = metrics.roc_auc_score(Y_train, train_probs)
                valid_auc = metrics.roc_auc_score(Y_valid, valid_probs)
                
                # Use default 0.5 threshold first to check
                valid_preds_default = (valid_probs >= 0.5).astype(int)
                default_f1 = metrics.f1_score(Y_valid, valid_preds_default, zero_division=0)
                st.write(f"- Default threshold (0.5) F1: {default_f1:.4f}")
                st.write(f"- Predictions at 0.5: DOWN={np.sum(valid_preds_default==0)}, UP={np.sum(valid_preds_default==1)}")
                
                threshold_info = find_optimal_threshold(Y_valid, valid_probs, metric_priority="f1")
                st.write(f"- Optimal threshold: {threshold_info['threshold']:.4f}, F1: {threshold_info['f1']:.4f}")
                
                valid_preds_threshold = (valid_probs >= threshold_info["threshold"]).astype(int)
                valid_accuracy = metrics.accuracy_score(Y_valid, valid_preds_threshold)
                valid_f1 = metrics.f1_score(Y_valid, valid_preds_threshold, zero_division=0)

                walk_forward_summary = compute_walk_forward_metrics(pipeline_template, X_train, Y_train)
                if walk_forward_summary:
                    walk_forward_table.append({
                        "Model": name,
                        "WF Folds": walk_forward_summary["n_folds"],
                        "WF Mean Accuracy": f"{walk_forward_summary['mean_accuracy']:.4f}",
                        "WF Mean F1": f"{walk_forward_summary['mean_f1']:.4f}",
                        "WF Mean AUC": f"{walk_forward_summary['mean_roc_auc']:.4f}"
                    })

                feature_importances = extract_feature_importance(pipeline_model, features_cols)

                trained_models.append({
                    "name": name,
                    "type": "classical",
                    "model": pipeline_model,
                    "eval": {
                        "X": X_valid,
                        "y": Y_valid
                    },
                    "latest_features": X_valid.iloc[[-1]] if not X_valid.empty else X_train.iloc[[-1]],
                    "threshold": threshold_info,
                    "walk_forward": walk_forward_summary,
                    "feature_importances": feature_importances,
                    "metrics": {
                        "Train AUC": train_auc,
                        "Valid AUC": valid_auc,
                        "Validation Accuracy": valid_accuracy,
                        "Validation F1": valid_f1,
                        "Validation Precision": threshold_info["precision"],
                        "Validation Recall": threshold_info["recall"]
                    }
                })

                results.append({
                    'Model': name,
                    'Train AUC': f"{train_auc:.4f}",
                    'Valid AUC': f"{valid_auc:.4f}",
                    'Validation Accuracy': f"{valid_accuracy:.4f}",
                    'Validation F1': f"{valid_f1:.4f}",
                    'Optimal Threshold': f"{threshold_info['threshold']:.2f}"
                })

            # LSTM model training (only if enough data and TensorFlow available)
            lstm_ready = TENSORFLOW_AVAILABLE
            if not TENSORFLOW_AVAILABLE:
                st.warning("⚠️ TensorFlow is not installed; skipping LSTM training. Install TensorFlow to enable deep learning.")

            if lstm_ready:
                if len(X_train) <= sequence_length or len(X_valid) <= sequence_length:
                    lstm_ready = False
                else:
                    lstm_scaler = MinMaxScaler()
                    train_lstm_features = X_train[lstm_feature_cols]
                    valid_lstm_features = X_valid[lstm_feature_cols]

                    train_scaled = lstm_scaler.fit_transform(train_lstm_features)
                    valid_scaled = lstm_scaler.transform(valid_lstm_features)

                    X_train_seq, Y_train_seq = create_lstm_sequences(train_scaled, Y_train.values, sequence_length)
                    X_valid_seq, Y_valid_seq = create_lstm_sequences(valid_scaled, Y_valid.values, sequence_length)

                    if len(X_train_seq) == 0 or len(X_valid_seq) == 0:
                        lstm_ready = False

            if lstm_ready:
                lstm_model = Sequential([
                    LSTM(64, return_sequences=True, input_shape=(sequence_length, len(lstm_feature_cols))),
                    Dropout(0.3),
                    LSTM(32),
                    Dropout(0.3),
                    Dense(16, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])

                lstm_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
                )

                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )

                history = lstm_model.fit(
                    X_train_seq, Y_train_seq,
                    validation_data=(X_valid_seq, Y_valid_seq),
                    epochs=lstm_epochs,
                    batch_size=lstm_batch_size,
                    callbacks=[early_stop],
                    verbose=0
                )

                train_probs_lstm = lstm_model.predict(X_train_seq, verbose=0).ravel()
                valid_probs_lstm = lstm_model.predict(X_valid_seq, verbose=0).ravel()
                threshold_info_lstm = find_optimal_threshold(Y_valid_seq, valid_probs_lstm, metric_priority="f1")
                valid_preds_lstm = (valid_probs_lstm >= threshold_info_lstm["threshold"]).astype(int)

                train_auc_lstm = metrics.roc_auc_score(Y_train_seq, train_probs_lstm)
                valid_auc_lstm = metrics.roc_auc_score(Y_valid_seq, valid_probs_lstm)
                valid_accuracy_lstm = metrics.accuracy_score(Y_valid_seq, valid_preds_lstm)
                valid_f1_lstm = metrics.f1_score(Y_valid_seq, valid_preds_lstm, zero_division=0)

                trained_models.append({
                    "name": f"LSTM (window={sequence_length})",
                    "type": "lstm",
                    "model": lstm_model,
                    "eval": {
                        "X": X_valid_seq,
                        "y": Y_valid_seq
                    },
                    "feature_importances": None,
                    "lstm_features": lstm_feature_cols,
                    "scaler": lstm_scaler,
                    "threshold": threshold_info_lstm,
                    "metrics": {
                        "Train AUC": train_auc_lstm,
                        "Valid AUC": valid_auc_lstm,
                        "Validation Accuracy": valid_accuracy_lstm,
                        "Validation F1": valid_f1_lstm,
                        "Validation Precision": threshold_info_lstm["precision"],
                        "Validation Recall": threshold_info_lstm["recall"]
                    }
                })

                results.append({
                    'Model': f"LSTM (window={sequence_length})",
                    'Train AUC': f"{train_auc_lstm:.4f}",
                    'Valid AUC': f"{valid_auc_lstm:.4f}",
                    'Validation Accuracy': f"{valid_accuracy_lstm:.4f}",
                    'Validation F1': f"{valid_f1_lstm:.4f}",
                    'Optimal Threshold': f"{threshold_info_lstm['threshold']:.2f}"
                })
            elif TENSORFLOW_AVAILABLE:
                st.warning("⚠️ Not enough sequential data to train the LSTM model with the current configuration.")

            results_df = pd.DataFrame(results)
            walk_results_df = pd.DataFrame(walk_forward_table) if walk_forward_table else pd.DataFrame()
            st.session_state['trained_models'] = trained_models
            st.session_state['results_df'] = results_df
            st.session_state['walk_results_df'] = walk_results_df
            st.session_state['features_cols'] = features_cols
            st.session_state['classical_data'] = {
                "X_train": X_train,
                "X_valid": X_valid,
                "Y_train": Y_train,
                "Y_valid": Y_valid
            }
            st.success("✅ Models trained successfully!")

    if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
        st.subheader("📊 Model Performance Comparison")
        st.dataframe(st.session_state['results_df'], use_container_width=True)

        walk_results_df = st.session_state.get('walk_results_df')
        if walk_results_df is not None and not walk_results_df.empty:
            with st.expander("🔁 Walk-Forward Validation Summary"):
                st.dataframe(walk_results_df, use_container_width=True)

        feature_importance_data = [
            (model['name'], model['feature_importances'])
            for model in st.session_state['trained_models']
            if model.get('feature_importances')
        ]
        if feature_importance_data:
            with st.expander("🏅 Top Feature Importance (by model)"):
                for name, importance_dict in feature_importance_data:
                    top_items = list(importance_dict.items())[:10]
                    st.markdown(f"**{name}**")
                    st.table(pd.DataFrame(top_items, columns=['Feature', 'Importance']))

        trained_models = st.session_state.get('trained_models', [])
        if trained_models:
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Detailed Metrics", "🔍 Confusion Matrix", "🎯 Prediction", "📊 Prediction vs Actual"])

            with tab1:
                st.subheader("Detailed Model Evaluation")
                model_idx = st.selectbox(
                    "Select Model for Evaluation",
                    range(len(trained_models)),
                    format_func=lambda x: trained_models[x]["name"],
                    key="eval_model"
                )

                selected_model = trained_models[model_idx]
                X_eval = selected_model['eval']['X']
                y_eval = selected_model['eval']['y']
                threshold_value = selected_model.get('threshold', {}).get('threshold', 0.5)

                if selected_model['type'] == 'classical':
                    probas = selected_model['model'].predict_proba(X_eval)[:, 1]
                else:
                    probas = selected_model['model'].predict(X_eval, verbose=0).ravel()

                preds = (probas >= threshold_value).astype(int)

                accuracy = metrics.accuracy_score(y_eval, preds)
                f1 = metrics.f1_score(y_eval, preds)
                roc_auc = metrics.roc_auc_score(y_eval, probas)

                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric("F1 Score", f"{f1:.4f}")
                col3.metric("ROC-AUC", f"{roc_auc:.4f}")
                st.metric("Decision Threshold", f"{threshold_value:.2f}")

                st.subheader("Classification Report")
                report = metrics.classification_report(
                    y_eval, preds,
                    target_names=['Down (0)', 'Up (1)'],
                    output_dict=True
                )
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

                feature_importances = selected_model.get('feature_importances')
                if feature_importances:
                    st.subheader("Top Influencing Features")
                    top_items = list(feature_importances.items())[:5]
                    st.table(pd.DataFrame(top_items, columns=['Feature', 'Importance']))
                elif selected_model['type'] == 'lstm':
                    st.info("Feature-level attribution is not available for the LSTM model.")

            with tab2:
                st.subheader("Confusion Matrix")
                model_idx = st.selectbox(
                    "Select Model for Confusion Matrix",
                    range(len(trained_models)),
                    format_func=lambda x: trained_models[x]["name"],
                    key="cm_model"
                )

                selected_model = trained_models[model_idx]
                X_eval = selected_model['eval']['X']
                y_eval = selected_model['eval']['y']
                threshold_value = selected_model.get('threshold', {}).get('threshold', 0.5)

                if selected_model['type'] == 'classical':
                    probas = selected_model['model'].predict_proba(X_eval)[:, 1]
                    # Use model's default predict instead of threshold
                    preds = selected_model['model'].predict(X_eval)
                else:
                    probas = selected_model['model'].predict(X_eval, verbose=0).ravel()
                    preds = (probas >= 0.5).astype(int)  # Use fixed 0.5 threshold for LSTM
                
                # Debug info
                st.write(f"**Debug Info:**")
                st.write(f"- Probabilities range: [{probas.min():.4f}, {probas.max():.4f}]")
                st.write(f"- Predictions: DOWN={np.sum(preds==0)}, UP={np.sum(preds==1)}")
                st.write(f"- Actual: DOWN={np.sum(y_eval==0)}, UP={np.sum(y_eval==1)}")

                cm = metrics.confusion_matrix(y_eval, preds)
                fig, ax = plt.subplots(figsize=(8, 6))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down (0)', 'Up (1)'])
                disp.plot(ax=ax, cmap='Blues', colorbar=False)
                ax.set_title(f'Confusion Matrix - {trained_models[model_idx]["name"]}', fontsize=14, fontweight='bold')
                st.pyplot(fig)

            with tab3:
                st.subheader("🎯 Next Trading Day Prediction")

                best_model_entry = max(
                    trained_models,
                    key=lambda m: m['metrics']['Validation F1']
                )
                st.info(
                    f"🏆 Best Model Selected: **{best_model_entry['name']}** "
                    f"(Validation F1: {best_model_entry['metrics']['Validation F1']:.4f})"
                )

                if st.button("🔮 Predict Next Day Movement", type="primary"):
                    with st.spinner("🔍 Analyzing latest market data..."):
                        threshold_value = best_model_entry.get('threshold', {}).get('threshold', 0.5)

                        if best_model_entry['type'] == 'classical':
                            latest_features = df[features_cols].iloc[-1:].copy()
                            prob_prediction = float(best_model_entry['model'].predict_proba(latest_features)[0][1])
                            top_features = best_model_entry.get('feature_importances')
                        else:
                            if len(df) < sequence_length:
                                st.warning("⚠️ Not enough recent data points to generate an LSTM prediction.")
                                st.stop()
                            lstm_features = best_model_entry.get('lstm_features', lstm_feature_cols)
                            latest_window = df[lstm_features].iloc[-sequence_length:]
                            scaled_window = best_model_entry['scaler'].transform(latest_window)
                            lstm_sequence = scaled_window.reshape(
                                1, scaled_window.shape[0], scaled_window.shape[1]
                            )
                            prob_prediction = float(best_model_entry['model'].predict(lstm_sequence, verbose=0).ravel()[0])
                            top_features = None

                        prediction = int(prob_prediction >= threshold_value)
                        confidence = (prob_prediction if prediction == 1 else 1 - prob_prediction) * 100
                        confidence = float(confidence)

                        pred_str = '📈 UP' if prediction == 1 else '📉 DOWN'

                        if prediction == 1:
                            st.success(f"🚀 Prediction: **{pred_str}**")
                        else:
                            st.error(f"📉 Prediction: **{pred_str}**")

                        st.write(f"🎯 Decision Threshold: **{threshold_value:.2f}**")
                        st.write(f"🔵 Probability of UP: **{prob_prediction * 100:.2f}%**")
                        st.write(f"💪 Confidence: **{confidence:.2f}%**")
                        st.write(
                            "📊 Expected Direction: Stock price likely to **MOVE UP**"
                            if prediction == 1 else
                            "📊 Expected Direction: Stock price likely to **MOVE DOWN**"
                        )

                        if top_features:
                            st.markdown("**Top contributing features (validation importance):**")
                            for feature_name, importance_value in list(top_features.items())[:3]:
                                st.write(f"- {feature_name}: {importance_value:.4f}")
                        elif best_model_entry['type'] == 'lstm':
                            st.info("LSTM contributions are opaque; consider using attention/SHAP for deeper explanations.")

                        st.markdown("---")
                        st.markdown(
                            f"**📋 Summary:** The best model ({best_model_entry['name']}) "
                            f"predicts the stock will move **{pred_str}** on the next trading day "
                            f"with **{confidence:.2f}%** confidence."
                        )

            with tab4:
                st.subheader("📊 Prediction vs Actual Comparison")
                st.markdown("Compare model predictions against actual outcomes for all trained models.")
                
                # Create comparison for all models
                for model_entry in trained_models:
                    with st.expander(f"🔍 {model_entry['name']}", expanded=False):
                        X_eval = model_entry['eval']['X']
                        y_eval = model_entry['eval']['y']
                        threshold_value = model_entry.get('threshold', {}).get('threshold', 0.5)
                        
                        # Get predictions
                        if model_entry['type'] == 'classical':
                            probas = model_entry['model'].predict_proba(X_eval)[:, 1]
                        else:
                            probas = model_entry['model'].predict(X_eval, verbose=0).ravel()
                        
                        preds = (probas >= threshold_value).astype(int)
                        
                        # Create comparison dataframe
                        comparison_df = pd.DataFrame({
                            'Actual': y_eval,
                            'Predicted': preds,
                            'Probability': probas,
                            'Correct': (y_eval == preds).astype(int)
                        })
                        
                        # Calculate metrics
                        accuracy = metrics.accuracy_score(y_eval, preds)
                        correct_predictions = (y_eval == preds).sum()
                        total_predictions = len(y_eval)
                        
                        # Display summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", f"{accuracy:.2%}")
                        col2.metric("Correct", f"{correct_predictions}/{total_predictions}")
                        col3.metric("F1 Score", f"{model_entry['metrics']['Validation F1']:.4f}")
                        col4.metric("Threshold", f"{threshold_value:.2f}")
                        
                        # Create visualization similar to the reference
                        fig, ax = plt.subplots(figsize=(18, 8))
                        
                        # Shade UP and DOWN zones
                        ax.fill_between(range(len(probas)), threshold_value, 1.0, alpha=0.2, color='lightgreen', 
                                       label='PREDICTED UP ZONE')
                        ax.fill_between(range(len(probas)), 0.0, threshold_value, alpha=0.2, color='lightcoral', 
                                       label='PREDICTED DOWN ZONE')
                        
                        # Add zone labels
                        ax.text(len(probas) / 2, 0.85, 'PREDICTED UP ZONE', 
                               fontsize=14, ha='center', va='center', alpha=0.6, color='darkgreen', fontweight='bold')
                        ax.text(len(probas) / 2, 0.15, 'PREDICTED DOWN ZONE', 
                               fontsize=14, ha='center', va='center', alpha=0.6, color='darkred', fontweight='bold')
                        
                        # Get indices for actual outcomes
                        up_indices = np.where(y_eval == 1)[0]
                        down_indices = np.where(y_eval == 0)[0]
                        
                        # Determine correct predictions
                        correct_up = up_indices[preds[up_indices] == 1]
                        wrong_up = up_indices[preds[up_indices] == 0]
                        correct_down = down_indices[preds[down_indices] == 0]
                        wrong_down = down_indices[preds[down_indices] == 1]
                        
                        # Plot actual outcomes with triangular markers
                        # Correct UP predictions (green upward triangles)
                        if len(correct_up) > 0:
                            ax.scatter(correct_up, probas[correct_up], marker='^', s=80, c='darkgreen', 
                                      edgecolors='black', linewidths=0.5, alpha=0.7, zorder=5, 
                                      label='△ Actual UP (Green=Correct)')
                        
                        # Wrong UP predictions (red upward triangles)
                        if len(wrong_up) > 0:
                            ax.scatter(wrong_up, probas[wrong_up], marker='^', s=80, c='red', 
                                      edgecolors='black', linewidths=0.5, alpha=0.7, zorder=5,
                                      label='△ Actual UP (Red=Wrong)')
                        
                        # Correct DOWN predictions (green downward triangles)
                        if len(correct_down) > 0:
                            ax.scatter(correct_down, probas[correct_down], marker='v', s=80, c='darkgreen', 
                                      edgecolors='black', linewidths=0.5, alpha=0.7, zorder=5, 
                                      label='▽ Actual DOWN (Green=Correct)')
                        
                        # Wrong DOWN predictions (red downward triangles)
                        if len(wrong_down) > 0:
                            ax.scatter(wrong_down, probas[wrong_down], marker='v', s=80, c='red', 
                                      edgecolors='black', linewidths=0.5, alpha=0.7, zorder=5,
                                      label='▽ Actual DOWN (Red=Wrong)')
                        
                        # Plot decision threshold
                        ax.axhline(y=threshold_value, color='blue', linestyle='--', linewidth=2.5, 
                                  label=f'▬ Decision Threshold', zorder=4)
                        
                        # Formatting
                        ax.set_xlabel('Time (Sample Index)', fontsize=14, fontweight='bold')
                        ax.set_ylabel('Prediction Probability (UP)', fontsize=14, fontweight='bold')
                        ax.set_title(f'Prediction Probabilities with Actual Outcomes\n{model_entry["name"]} | Accuracy: {accuracy:.2%} | Correct: {(y_eval == preds).sum()}/{len(y_eval)}', 
                                    fontsize=16, fontweight='bold', pad=20)
                        ax.legend(loc='upper left', fontsize=11, framealpha=0.95, ncol=3)
                        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                        ax.set_ylim([0, 1.0])
                        ax.set_xlim([0, len(probas)])
                        
                        # Add subtle background
                        ax.set_facecolor('#fafafa')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Show detailed comparison table (sample)
                        st.markdown("**📋 Detailed Predictions (Last 20 samples):**")
                        display_df = comparison_df.tail(20).copy()
                        display_df['Actual'] = display_df['Actual'].map({0: '📉 Down', 1: '📈 Up'})
                        display_df['Predicted'] = display_df['Predicted'].map({0: '📉 Down', 1: '📈 Up'})
                        display_df['Probability'] = display_df['Probability'].apply(lambda x: f"{x:.2%}")
                        display_df['Match'] = display_df['Correct'].map({0: '❌', 1: '✅'})
                        display_df = display_df[['Actual', 'Predicted', 'Probability', 'Match']]
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Additional statistics
                        st.markdown("**📈 Prediction Statistics:**")
                        stats_col1, stats_col2 = st.columns(2)
                        
                        with stats_col1:
                            st.write("**By Prediction:**")
                            pred_up = (preds == 1).sum()
                            pred_down = (preds == 0).sum()
                            st.write(f"- Predicted UP: {pred_up} ({pred_up/len(preds)*100:.1f}%)")
                            st.write(f"- Predicted DOWN: {pred_down} ({pred_down/len(preds)*100:.1f}%)")
                        
                        with stats_col2:
                            st.write("**By Actual:**")
                            actual_up = (y_eval == 1).sum()
                            actual_down = (y_eval == 0).sum()
                            st.write(f"- Actual UP: {actual_up} ({actual_up/len(y_eval)*100:.1f}%)")
                            st.write(f"- Actual DOWN: {actual_down} ({actual_down/len(y_eval)*100:.1f}%)")

else:
    st.info("👆 Please configure the stock symbol and date range in the sidebar, then click 'Download & Analyze Data' to begin.") 