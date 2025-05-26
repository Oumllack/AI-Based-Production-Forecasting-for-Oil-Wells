import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_synthetic_well_data(start_date='2017-01-01', days=2555):
    """
    Génère des données simulées pour un puits de pétrole sur une période de 7 ans.
    
    Args:
        start_date (str): Date de début au format 'YYYY-MM-DD'
        days (int): Nombre de jours de données à générer (7 ans = 2555 jours)
        
    Returns:
        pd.DataFrame: DataFrame contenant les données simulées
    """
    # Génération des dates
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    # Paramètres de base
    base_oil_rate = 1200  # bbl/day - valeur initiale plus élevée pour 7 ans
    base_water_cut = 0.25  # 25% - départ plus bas pour montrer l'évolution
    base_gas_rate = 600   # Mscf/day
    base_tubing_pressure = 2200  # psi
    base_casing_pressure = 2700  # psi
    base_choke_size = 32  # 1/64 inch
    
    # Génération des données avec tendance à la baisse et variations saisonnières
    t = np.arange(days)
    
    # Tendance à la baisse exponentielle pour l'oil rate
    # Taux de déclin plus faible pour 7 ans
    decline_rate = 0.0003  # taux de déclin quotidien ajusté pour 7 ans
    oil_rate = base_oil_rate * np.exp(-decline_rate * t)
    
    # Ajout de variations saisonnières et bruit
    seasonal = 75 * np.sin(2 * np.pi * t / 365)  # variation saisonnière annuelle plus marquée
    noise = np.random.normal(0, 25, days)  # bruit aléatoire légèrement augmenté
    oil_rate = oil_rate + seasonal + noise
    oil_rate = np.maximum(oil_rate, 0)  # éviter les valeurs négatives
    
    # Water cut augmente avec le temps
    # Taux d'augmentation ajusté pour 7 ans
    water_cut = base_water_cut + 0.00008 * t + np.random.normal(0, 0.02, days)
    water_cut = np.minimum(np.maximum(water_cut, 0), 0.95)  # limiter entre 0 et 95%
    
    # Gas rate corrélé avec l'oil rate
    gas_rate = oil_rate * 0.5 + np.random.normal(0, 60, days)
    gas_rate = np.maximum(gas_rate, 0)
    
    # Pressions avec variations corrélées
    # Ajout d'une tendance à la baisse légère pour les pressions
    tubing_pressure = base_tubing_pressure * np.exp(-0.0001 * t) + 150 * np.sin(2 * np.pi * t / 180) + np.random.normal(0, 50, days)
    casing_pressure = base_casing_pressure * np.exp(-0.0001 * t) + 200 * np.sin(2 * np.pi * t / 180) + np.random.normal(0, 75, days)
    
    # Choke size avec changements occasionnels
    # Plus de changements sur 7 ans
    choke_size = np.ones(days) * base_choke_size
    change_days = np.random.choice(days, size=20, replace=False)  # plus de changements sur 7 ans
    for day in change_days:
        choke_size[day:] = np.random.choice([24, 28, 32, 36, 40, 44, 48])  # plus d'options de taille
    
    # Création du DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Oil_rate': np.round(oil_rate, 2),
        'Water_cut': np.round(water_cut, 4),
        'Gas_rate': np.round(gas_rate, 2),
        'Tubing_Pressure': np.round(tubing_pressure, 2),
        'Casing_Pressure': np.round(casing_pressure, 2),
        'Choke_size': choke_size.astype(int)
    })
    
    return df

if __name__ == "__main__":
    # Génération des données
    df = generate_synthetic_well_data()
    
    # Création du dossier data s'il n'existe pas
    import os
    os.makedirs('data', exist_ok=True)
    
    # Sauvegarde des données
    df.to_csv('data/synthetic_well_data.csv', index=False)
    print("Données générées et sauvegardées dans 'data/synthetic_well_data.csv'")
    
    # Affichage des statistiques de base
    print("\nStatistiques des données générées :")
    print(df.describe()) 