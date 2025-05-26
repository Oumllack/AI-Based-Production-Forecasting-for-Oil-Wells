import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# XGBoost
from xgboost import XGBRegressor

# Prophet
try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_OK = True
    except ImportError:
        PROPHET_OK = False
        print("[AVERTISSEMENT] Prophet/fbprophet n'est pas installé. Les prévisions Prophet seront ignorées.")

# Statsmodels pour ARIMA et Holt-Winters
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    ARIMA = None
    ExponentialSmoothing = None

import xgboost as xgb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

def prepare_data(df):
    """
    Prépare les features et la cible pour la prédiction de Oil_rate.
    Utilise les variables disponibles sauf la date et la cible.
    """
    features = ['Water_cut', 'Gas_rate', 'Tubing_Pressure', 'Casing_Pressure', 'Choke_size']
    X = df[features]
    y = df['Oil_rate']
    return X, y

def split_train_test(df, train_years=6):
    """
    Sépare le DataFrame en train et test selon le nombre d'années.
    """
    df = df.sort_values('Date')
    split_date = df['Date'].min() + pd.DateOffset(years=train_years)
    train_df = df[df['Date'] < split_date]
    test_df = df[df['Date'] >= split_date]
    return train_df, test_df

def eval_and_plot(test_df, y_test, y_pred, model_name, color, results, plot_ax):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({'Modèle': model_name, 'MSE': mse, 'R2': r2})
    plot_ax.plot(test_df['Date'], y_pred, label=f'{model_name}', linestyle='--', color=color)
    return mse, r2

def run_all_models(df):
    """Run all models and compare their performance on year 7."""
    # Prepare data
    features = ['Water_cut', 'Gas_rate', 'Tubing_Pressure', 'Casing_Pressure', 'Choke_size']
    X = df[features]
    y = df['Oil_rate']
    
    # Split data: years 1-6 for training, year 7 for testing
    train_idx = df['Date'] < df['Date'].min() + pd.DateOffset(years=6)
    test_idx = (df['Date'] >= df['Date'].min() + pd.DateOffset(years=6)) & (df['Date'] < df['Date'].min() + pd.DateOffset(years=7))
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    dates_test = df['Date'][test_idx]
    
    # Initialize results storage
    results = []
    predictions = {}
    
    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    results.append(('LinearRegression', mse_lr, r2_lr))
    predictions['LinearRegression'] = y_pred_lr
    
    # 2. Random Forest
    rf = train_random_forest(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    results.append(('RandomForest', mse_rf, r2_rf))
    predictions['RandomForest'] = y_pred_rf
    
    # 3. XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    results.append(('XGBoost', mse_xgb, r2_xgb))
    predictions['XGBoost'] = y_pred_xgb
    
    # 4. Prophet
    if PROPHET_OK:
        prophet_model = train_prophet(df[train_idx | test_idx][['Date', 'Oil_rate']])
        future_dates = df['Date'][test_idx]
        y_pred_prophet = predict_prophet(prophet_model, future_dates)
        mse_prophet = mean_squared_error(y_test, y_pred_prophet)
        r2_prophet = r2_score(y_test, y_pred_prophet)
        results.append(('Prophet', mse_prophet, r2_prophet))
        predictions['Prophet'] = y_pred_prophet
    
    # 5. ARIMA
    try:
        arima_model = train_arima(df[train_idx]['Oil_rate'])
        y_pred_arima = predict_arima(arima_model, len(y_test))
        mse_arima = mean_squared_error(y_test, y_pred_arima)
        r2_arima = r2_score(y_test, y_pred_arima)
        results.append(('ARIMA', mse_arima, r2_arima))
        predictions['ARIMA'] = y_pred_arima
    except Exception as e:
        print(f"ARIMA model failed: {e}")
    
    # 6. LSTM
    try:
        lstm_model = train_lstm(X_train, y_train)
        y_pred_lstm = predict_lstm(lstm_model, X_test)
        mse_lstm = mean_squared_error(y_test, y_pred_lstm)
        r2_lstm = r2_score(y_test, y_pred_lstm)
        results.append(('LSTM', mse_lstm, r2_lstm))
        predictions['LSTM'] = y_pred_lstm
    except Exception as e:
        print(f"LSTM model failed: {e}")
    
    # 7. Holt-Winters
    try:
        hw_model = train_holt_winters(df[train_idx]['Oil_rate'])
        y_pred_hw = predict_holt_winters(hw_model, len(y_test))
        mse_hw = mean_squared_error(y_test, y_pred_hw)
        r2_hw = r2_score(y_test, y_pred_hw)
        results.append(('Holt-Winters', mse_hw, r2_hw))
        predictions['Holt-Winters'] = y_pred_hw
    except Exception as e:
        print(f"Holt-Winters model failed: {e}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results, columns=['Modèle', 'MSE', 'R2'])
    results_df = results_df.sort_values('MSE')
    
    # Print results
    print("\nComparaison des modèles sur la 7e année :")
    print(results_df.to_string(index=False))
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label='Données réelles', color='black')
    
    for model_name, pred in predictions.items():
        plt.plot(dates_test, pred, label=model_name, linestyle='--')
    
    plt.title('Comparaison des prédictions des modèles sur la 7e année')
    plt.xlabel('Date')
    plt.ylabel('Débit de pétrole (bbl/j)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('data/plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nMeilleur modèle pour la projection :", results_df.iloc[0]['Modèle'])
    print("Graphiques sauvegardés dans data/plots/")

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_prophet(df):
    prophet_df = df[['Date', 'Oil_rate']].rename(columns={'Date': 'ds', 'Oil_rate': 'y'})
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)
    return model

def predict_prophet(model, future_dates):
    future = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future)
    return forecast['yhat'].values

def train_arima(train_series, order=(5,1,0)):
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    return model_fit

def predict_arima(model_fit, steps):
    return model_fit.forecast(steps=steps)

def train_lstm(X_train, y_train):
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(X_train.values.reshape(-1, 1))
    X_train_lstm = scaled_train.reshape((scaled_train.shape[0], scaled_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=3)])
    return model

def predict_lstm(model, X_test):
    scaled_test = scaler.transform(X_test.values.reshape(-1, 1))
    X_test_lstm = scaled_test.reshape((scaled_test.shape[0], scaled_test.shape[1], 1))
    return model.predict(X_test_lstm, verbose=0).flatten()

def generate_prophet_forecast(df, forecast_years=5):
    """
    Generate a 5-year forecast using Prophet model.
    
    Args:
        df (pd.DataFrame): Historical data
        forecast_years (int): Number of years to forecast
        
    Returns:
        tuple: (historical_dates, historical_values, forecast_dates, forecast_values, forecast_uncertainty)
    """
    if not PROPHET_OK:
        raise ImportError("Prophet is not installed. Please install it using: pip install prophet")
    
    # Prepare data for Prophet
    prophet_df = df[['Date', 'Oil_rate']].rename(columns={'Date': 'ds', 'Oil_rate': 'y'})
    
    # Initialize and fit Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,  # More flexible trend
        seasonality_prior_scale=10.0,  # Stronger seasonality
        seasonality_mode='multiplicative'  # Better for oil production
    )
    
    # Add custom seasonality if needed
    model.add_seasonality(
        name='yearly',
        period=365.25,
        fourier_order=10
    )
    
    # Fit the model
    model.fit(prophet_df)
    
    # Generate future dates
    future_dates = model.make_future_dataframe(periods=forecast_years*365, freq='D')
    
    # Make predictions
    forecast = model.predict(future_dates)
    
    # Split into historical and forecast periods
    historical_mask = future_dates['ds'] <= df['Date'].max()
    forecast_mask = ~historical_mask
    
    return (
        df['Date'],
        df['Oil_rate'],
        forecast.loc[forecast_mask, 'ds'],
        forecast.loc[forecast_mask, 'yhat'],
        forecast.loc[forecast_mask, ['yhat_lower', 'yhat_upper']]
    )

def plot_prophet_forecast(historical_dates, historical_values, forecast_dates, forecast_values, forecast_uncertainty):
    """
    Plot the Prophet forecast with historical data and uncertainty intervals.
    """
    plt.figure(figsize=(15, 6))
    
    # Plot historical data
    plt.plot(historical_dates, historical_values, 
             label='Historical Data', color='black', alpha=0.7)
    
    # Plot forecast
    plt.plot(forecast_dates, forecast_values, 
             label='Prophet Forecast', color='red', linestyle='--')
    
    # Plot uncertainty intervals
    plt.fill_between(forecast_dates,
                     forecast_uncertainty['yhat_lower'],
                     forecast_uncertainty['yhat_upper'],
                     color='red', alpha=0.1, label='95% Confidence Interval')
    
    # Customize plot
    plt.title('5-Year Oil Production Forecast using Prophet', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Oil Rate (bbl/day)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig('data/plots/projection_5_ans.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load data
    df = pd.read_csv('data/synthetic_well_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Run model comparison for year 7
    run_all_models(df)
    
    # Generate and plot Prophet forecast
    try:
        hist_dates, hist_values, fcst_dates, fcst_values, fcst_uncertainty = generate_prophet_forecast(df)
        plot_prophet_forecast(hist_dates, hist_values, fcst_dates, fcst_values, fcst_uncertainty)
        print("\nProphet forecast generated and saved to 'data/plots/projection_5_ans.png'")
    except Exception as e:
        print(f"\nError generating Prophet forecast: {e}")

if __name__ == "__main__":
    main() 