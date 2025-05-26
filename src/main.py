import os
from data.generate_synthetic_data import generate_synthetic_well_data
from data.visualize_data import plot_well_data

def main():
    """
    Fonction principale pour générer et visualiser les données du puits.
    """
    print("Démarrage de la génération des données...")
    
    # Création des répertoires nécessaires
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/plots', exist_ok=True)
    
    # Génération des données
    df = generate_synthetic_well_data()
    
    # Sauvegarde des données
    df.to_csv('data/synthetic_well_data.csv', index=False)
    print("✓ Données générées et sauvegardées dans 'data/synthetic_well_data.csv'")
    
    # Affichage des statistiques de base
    print("\nStatistiques des données générées :")
    print(df.describe())
    
    # Génération des visualisations
    print("\nGénération des visualisations...")
    plot_well_data(df)
    print("✓ Visualisations générées et sauvegardées dans 'data/plots'")
    
    print("\nProcessus terminé avec succès !")

if __name__ == "__main__":
    main() 