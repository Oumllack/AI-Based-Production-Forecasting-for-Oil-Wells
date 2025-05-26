import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

def create_visualization_directory():
    """Create the plots directory if it doesn't exist"""
    os.makedirs('data/plots', exist_ok=True)

def plot_oil_rate_trend(df):
    """Plot oil production rate trend"""
    plt.figure(figsize=(15, 6))
    plt.plot(df['Date'], df['Oil_rate'], label='Daily Oil Rate')
    plt.title('Oil Production Rate Trend Over Time', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Oil Rate (bbl/day)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/plots/oil_rate_trend.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_water_cut_trend(df):
    """Plot water cut trend"""
    plt.figure(figsize=(15, 6))
    plt.plot(df['Date'], df['Water_cut'] * 100, label='Water Cut')
    plt.title('Water Cut Evolution Over Time', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Water Cut (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/plots/water_cut_trend.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_pressure_trends(df):
    """Plot tubing and casing pressure trends"""
    plt.figure(figsize=(15, 6))
    plt.plot(df['Date'], df['Tubing_Pressure'], label='Tubing Pressure')
    plt.plot(df['Date'], df['Casing_Pressure'], label='Casing Pressure')
    plt.title('Pressure Trends Over Time', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Pressure (psi)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/plots/pressure_trends.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_oil_gas_correlation(df):
    """Plot oil rate vs gas rate correlation"""
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Oil_rate'], df['Gas_rate'], alpha=0.5)
    plt.title('Oil Rate vs Gas Rate Correlation', fontsize=14)
    plt.xlabel('Oil Rate (bbl/day)')
    plt.ylabel('Gas Rate (Mscf/day)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/plots/oil_gas_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(df):
    """Plot correlation matrix of numerical features"""
    numerical_cols = ['Oil_rate', 'Water_cut', 'Gas_rate', 'Tubing_Pressure', 'Casing_Pressure']
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('data/plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison(actual, predictions_dict):
    """Plot comparison of different model predictions"""
    plt.figure(figsize=(15, 6))
    plt.plot(actual.index, actual.values, label='Actual', color='black', alpha=0.7)
    
    colors = ['blue', 'red', 'green', 'purple']
    for (model_name, pred), color in zip(predictions_dict.items(), colors):
        plt.plot(pred.index, pred.values, label=model_name, alpha=0.7, color=color)
    
    plt.title('Model Predictions Comparison', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Oil Rate (bbl/day)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_actual_vs_predicted(actual, predicted, model_name):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(15, 6))
    plt.plot(actual.index, actual.values, label='Actual', color='black')
    plt.plot(predicted.index, predicted.values, label=f'{model_name} Prediction', color='red', alpha=0.7)
    plt.title(f'Actual vs {model_name} Predictions', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Oil Rate (bbl/day)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/plots/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_future_forecast(historical, forecast, model_name):
    """Plot historical data and future forecast"""
    plt.figure(figsize=(15, 6))
    plt.plot(historical.index, historical.values, label='Historical Data', color='black')
    plt.plot(forecast.index, forecast.values, label=f'{model_name} Forecast', color='red', alpha=0.7)
    plt.fill_between(forecast.index, 
                     forecast.values - 1.96 * np.std(historical.values),
                     forecast.values + 1.96 * np.std(historical.values),
                     color='red', alpha=0.1, label='95% Confidence Interval')
    plt.title(f'5-Year Production Forecast using {model_name}', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Oil Rate (bbl/day)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/plots/projection_5_ans.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create visualization directory
    create_visualization_directory()
    
    # Load data
    df = pd.read_csv('data/synthetic_well_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Generate all plots
    plot_oil_rate_trend(df)
    plot_water_cut_trend(df)
    plot_pressure_trends(df)
    plot_oil_gas_correlation(df)
    plot_correlation_matrix(df)
    
    print("All visualizations have been generated in the 'data/plots' directory") 