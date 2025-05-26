import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def plot_well_data(df, save_dir='data/plots'):
    """
    Génère des visualisations pour les données du puits.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données du puits
        save_dir (str): Répertoire où sauvegarder les graphiques
    """
    # Création du répertoire de sauvegarde
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Style des graphiques
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [15, 6]
    plt.rcParams['axes.grid'] = True
    
    # 1. Oil Rate au fil du temps
    plt.figure()
    plt.plot(df['Date'], df['Oil_rate'], linewidth=1, color='#1f77b4')
    plt.title('Évolution du taux de production pétrolière sur 7 ans')
    plt.xlabel('Date')
    plt.ylabel('Oil Rate (bbl/day)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/oil_rate_trend.png')
    plt.close()
    
    # 2. Water Cut au fil du temps
    plt.figure()
    plt.plot(df['Date'], df['Water_cut'] * 100, linewidth=1, color='#ff7f0e')
    plt.title('Évolution du Water Cut sur 7 ans')
    plt.xlabel('Date')
    plt.ylabel('Water Cut (%)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/water_cut_trend.png')
    plt.close()
    
    # 3. Matrice de corrélation
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Matrice de corrélation des variables numériques')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_matrix.png')
    plt.close()
    
    # 4. Distribution des pressions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    ax1.plot(df['Date'], df['Tubing_Pressure'], label='Tubing Pressure', linewidth=1, color='#2ca02c')
    ax1.set_title('Évolution de la pression tubing sur 7 ans')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Pression (psi)')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(df['Date'], df['Casing_Pressure'], label='Casing Pressure', color='#d62728', linewidth=1)
    ax2.set_title('Évolution de la pression casing sur 7 ans')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Pression (psi)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pressure_trends.png')
    plt.close()
    
    # 5. Scatter plot Oil Rate vs Gas Rate
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Oil_rate'], df['Gas_rate'], alpha=0.5, color='#9467bd')
    plt.title('Relation entre Oil Rate et Gas Rate')
    plt.xlabel('Oil Rate (bbl/day)')
    plt.ylabel('Gas Rate (Mscf/day)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/oil_gas_correlation.png')
    plt.close()

if __name__ == "__main__":
    # Chargement des données
    df = pd.read_csv('data/synthetic_well_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Génération des visualisations
    plot_well_data(df)
    print("Visualisations générées et sauvegardées dans le dossier 'data/plots'") 