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
    train_df, test_df = split_train_test(df)
    X_train, y_train = prepare_data(train_df)
    X_test, y_test = prepare_data(test_df)
    results = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(test_df['Date'], y_test, label='Production réelle', color='black')
    color_idx = 0

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    eval_and_plot(test_df, y_test, y_pred_lr, 'LinearRegression', colors[color_idx], results, ax)
    color_idx += 1

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    eval_and_plot(test_df, y_test, y_pred_rf, 'RandomForest', colors[color_idx], results, ax)
    color_idx += 1

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    eval_and_plot(test_df, y_test, y_pred_xgb, 'XGBoost', colors[color_idx], results, ax)
    color_idx += 1

    # Prophet (univarié)
    if PROPHET_OK:
        prophet_df = train_df[['Date', 'Oil_rate']].rename(columns={'Date': 'ds', 'Oil_rate': 'y'})
        prophet = Prophet(yearly_seasonality=True, daily_seasonality=False)
        prophet.fit(prophet_df)
        future = test_df[['Date']].rename(columns={'Date': 'ds'})
        forecast = prophet.predict(future)
        y_pred_prophet = forecast['yhat'].values
        eval_and_plot(test_df, y_test, y_pred_prophet, 'Prophet', colors[color_idx], results, ax)
        color_idx += 1

    # ARIMA (univarié)
    if ARIMA is not None:
        arima_fit = train_arima(train_df['Oil_rate'])
        y_pred_arima = predict_arima(arima_fit, len(test_df))
        eval_and_plot(test_df, y_test, y_pred_arima, 'ARIMA', colors[color_idx], results, ax)
        color_idx += 1

    # LSTM (univarié)
    try:
        lstm_model, lstm_scaler, lookback = train_lstm(train_df['Oil_rate'])
        y_pred_lstm = predict_lstm(lstm_model, lstm_scaler, lookback, train_df['Oil_rate'], len(test_df))
        eval_and_plot(test_df, y_test, y_pred_lstm, 'LSTM', colors[color_idx], results, ax)
        color_idx += 1
    except Exception as e:
        print(f"[AVERTISSEMENT] LSTM non testé : {e}")

    # Holt-Winters (univarié)
    if ExponentialSmoothing is not None:
        hw = ExponentialSmoothing(train_df['Oil_rate'], trend='add', seasonal='add', seasonal_periods=365)
        hw_fit = hw.fit()
        y_pred_hw = hw_fit.forecast(len(test_df))
        eval_and_plot(test_df, y_test, y_pred_hw, 'Holt-Winters', colors[color_idx], results, ax)
        color_idx += 1

    ax.set_title("Comparaison des modèles sur la 7e année")
    ax.set_xlabel("Date")
    ax.set_ylabel("Oil Rate (bbl/day)")
    ax.legend()
    plt.tight_layout()
    os.makedirs('data/plots', exist_ok=True)
    plt.savefig('data/plots/comparaison_tous_modeles_7e_annee.png')
    plt.close()

    # Affichage des résultats
    results_df = pd.DataFrame(results)
    print("\nComparaison des modèles sur la 7e année :")
    print(results_df.sort_values('MSE'))
    results_df.to_csv('data/plots/resultats_comparaison_modeles.csv', index=False)

    # Projection sur 5 ans (avec le meilleur modèle)
    best_model = results_df.sort_values('MSE').iloc[0]['Modèle']
    print(f"\nMeilleur modèle pour la projection : {best_model}")
    if best_model == 'LinearRegression':
        model = lr
    elif best_model == 'RandomForest':
        model = rf
    elif best_model == 'XGBoost':
        model = xgb
    else:
        model = lr  # fallback

    last_known = test_df.iloc[-1].copy()
    future_dates = pd.date_range(start=last_known['Date'] + pd.Timedelta(days=1), periods=5*365, freq='D')
    future_data = []
    for d in future_dates:
        row = last_known.copy()
        row['Date'] = d
        future_data.append(row)
    future_df = pd.DataFrame(future_data)
    X_future, _ = prepare_data(future_df)
    y_future_pred = model.predict(X_future)

    plt.figure(figsize=(15,6))
    plt.plot(df['Date'], df['Oil_rate'], label='Historique réel')
    plt.plot(future_dates, y_future_pred, label=f'Projection 5 ans ({best_model})', linestyle='--')
    plt.title(f"Projection de la production sur 5 ans ({best_model})")
    plt.xlabel("Date")
    plt.ylabel("Oil Rate (bbl/day)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/plots/projection_5_ans_best_model.png')
    plt.close()
    print("Graphiques sauvegardés dans data/plots/")

    # Sauvegarde des scores
    models = ['LinearRegression', 'RandomForest', 'XGBoost']
    mses = [mse_lr, mse_rf, mse_xgb]
    r2s = [r2_lr, r2_rf, r2_xgb]
    if PROPHET_OK:
        models.append('Prophet')
        mses.append(mse_prophet)
        r2s.append(r2_prophet)
    metrics = pd.DataFrame({
        'Model': models,
        'MSE': mses,
        'R2': r2s
    })
    metrics.to_csv('data/plots/model_metrics.csv', index=False)

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

def train_lstm(train_series, epochs=20, batch_size=32):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_series.values.reshape(-1,1))
    X, y = [], []
    lookback = 30
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(32, input_shape=(lookback,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[EarlyStopping(patience=3)])
    return model, scaler, lookback

def predict_lstm(model, scaler, lookback, train_series, test_len):
    full_series = np.concatenate([train_series.values, np.zeros(test_len)])
    preds = []
    for i in range(test_len):
        last_seq = full_series[i:lookback+i]
        last_seq_scaled = scaler.transform(last_seq.reshape(-1,1))
        X_input = last_seq_scaled.reshape((1, lookback, 1))
        pred_scaled = model.predict(X_input, verbose=0)[0,0]
        pred = scaler.inverse_transform([[pred_scaled]])[0,0]
        full_series[lookback+i] = pred
        preds.append(pred)
    return np.array(preds)

def main():
    # ... existing code ...
    # Préparation des données
    df = pd.read_csv('data/synthetic_well_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    features = ['Water_cut', 'Gas_rate', 'Tubing_Pressure', 'Casing_Pressure', 'Choke_size']
    X = df[features]
    y = df['Oil_rate']
    # Découpage : années 1-6 pour entraînement, année 7 pour test
    train_idx = df['Date'] < df['Date'].min() + pd.DateOffset(years=6)
    test_idx = (df['Date'] >= df['Date'].min() + pd.DateOffset(years=6)) & (df['Date'] < df['Date'].min() + pd.DateOffset(years=7))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    dates_test = df['Date'][test_idx]
    # Modèle 1 : Régression linéaire
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    # Modèle 2 : Random Forest
    rf = train_random_forest(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    # Modèle 3 : XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    # Modèle 4 : Prophet (univarié)
    if PROPHET_OK:
        prophet_model = train_prophet(df[train_idx | test_idx][['Date', 'Oil_rate']])
        future_dates = df['Date'][test_idx]
        y_pred_prophet = predict_prophet(prophet_model, future_dates)
        mse_prophet = mean_squared_error(y_test, y_pred_prophet)
        r2_prophet = r2_score(y_test, y_pred_prophet)
    # Affichage des scores
    print(f"\n--- Scores sur l'année 7 ---")
    print(f"Régression linéaire : MSE={mse_lr:.2f}, R²={r2_lr:.3f}")
    print(f"Random Forest : MSE={mse_rf:.2f}, R²={r2_rf:.3f}")
    print(f"XGBoost : MSE={mse_xgb:.2f}, R²={r2_xgb:.3f}")
    if PROPHET_OK:
        print(f"Prophet : MSE={mse_prophet:.2f}, R²={r2_prophet:.3f}")
    # Prévision sur 5 ans (tous modèles)
    future_steps = 5 * 365
    last_row = df.iloc[-1]
    X_future = pd.DataFrame({
        'Water_cut': np.linspace(last_row['Water_cut'], min(0.99, last_row['Water_cut']+0.1), future_steps),
        'Gas_rate': np.linspace(last_row['Gas_rate'], last_row['Gas_rate']*0.8, future_steps),
        'Tubing_Pressure': np.linspace(last_row['Tubing_Pressure'], last_row['Tubing_Pressure']*0.8, future_steps),
        'Casing_Pressure': np.linspace(last_row['Casing_Pressure'], last_row['Casing_Pressure']*0.8, future_steps),
        'Choke_size': np.full(future_steps, last_row['Choke_size'])
    })
    future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_steps)
    y_future_lr = lr.predict(X_future)
    y_future_rf = rf.predict(X_future)
    y_future_xgb = xgb_model.predict(X_future)
    # Prophet : prolongation
    if PROPHET_OK:
        prophet_future = pd.DataFrame({'ds': future_dates})
        prophet_forecast = prophet_model.predict(prophet_future)
        y_future_prophet = prophet_forecast['yhat'].values
    # Visualisation
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Oil_rate'], label='Production réelle (historique)', color='black')
    plt.plot(dates_test, y_pred_lr, label='Prédiction LR (Année 7)', linestyle='--')
    plt.plot(dates_test, y_pred_rf, label='Prédiction RF (Année 7)', linestyle='--')
    plt.plot(dates_test, y_pred_xgb, label='Prédiction XGB (Année 7)', linestyle='--')
    if PROPHET_OK:
        plt.plot(dates_test, y_pred_prophet, label='Prédiction Prophet (Année 7)', linestyle='--')
    plt.plot(future_dates, y_future_lr, label='Projection LR (5 ans)', color='blue', alpha=0.5)
    plt.plot(future_dates, y_future_rf, label='Projection RF (5 ans)', color='green', alpha=0.5)
    plt.plot(future_dates, y_future_xgb, label='Projection XGB (5 ans)', color='red', alpha=0.5)
    if PROPHET_OK:
        plt.plot(future_dates, y_future_prophet, label='Projection Prophet (5 ans)', color='orange', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('Débit de pétrole (bbl/j)')
    plt.title('Comparaison des modèles de prévision de la production pétrolière')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/plots/model_comparison.png')
    print("\nGraphique de comparaison sauvegardé dans data/plots/model_comparison.png")

if __name__ == "__main__":
    main() 