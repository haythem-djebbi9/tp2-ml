import pandas as pd


df = pd.read_csv('seattle-weather.csv')

# Afficher les premières lignes du DataFrame
print(df.head())
# Structure du Dataset
# Afficher les colonnes du DataFrame
print("Colonnes du DataFrame :")
print(df.columns)

# Afficher la forme du DataFrame (nombre de lignes et de colonnes)
print("\nForme du DataFrame :")
print(df.shape)

# taille du data

print("\nForme du DataFrame :")
print(df.shape)
# valeur nulle


print("Valeurs manquantes par colonne :")
print(df.isnull().sum())# statistique data


print("Statistiques descriptives des variables numériques :")
print(df.describe())
# statistique data


print("Statistiques descriptives des variables numériques :")
print(df.describe())# statistique data object
print("Statistiques descriptives des variables de type objet :")
print(df.describe(include=['object']))


# Analyse de la Variable Cible

print("Répartition des valeurs de la variable 'weather' :")
print(df['weather'].value_counts())

# Supprimez la colonne date

# Supprimer la colonne 'date' du DataFrame
del df['date']

# Afficher les premières lignes du DataFrame après la suppression
print(df.head())

df

from sklearn.preprocessing import LabelEncoder

# Initialiser le LabelEncoder
label_encoder = LabelEncoder()

# Encoder la variable 'weather' et créer une nouvelle colonne 'weather_encode'
df['weather_encode'] = label_encoder.fit_transform(df['weather'])

# Afficher les premières lignes du DataFrame avec la colonne encodée
print(df.head())
df



import seaborn as sns
import matplotlib.pyplot as plt

# Supprimer la colonne 'weather' pour ne garder que les variables numériques
df_numeric = df.drop('weather', axis=1)

# Calculer la matrice de corrélation
correlation_matrix = df_numeric.corr()

# Afficher la heatmap de la matrice de corrélation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matrice de Corrélation')
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# Créer un pairplot avec la variable 'weather' comme hue
sns.pairplot(df, hue='weather', palette='Set1')

# Afficher la visualisation
plt.show()


# Supprimer les colonnes inutiles

X = df.drop('weather_encode', axis=1)  # Supprimer la colonne 'weather_encode' pour X
y = df['weather_encode']  # Assigner la colonne 'weather_encode' à y

# Afficher X et y pour vérifier


# affichage X

print(X.head())


# affichage y


print(y.head())


X.columns

from sklearn.model_selection import train_test_split

# Diviser les données en ensemble d'entraînement (70%) et ensemble de test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Afficher la taille des ensembles pour vérifier
print(f'Taille de l\'ensemble d\'entraînement X: {X_train.shape}')
print(f'Taille de l\'ensemble de test X: {X_test.shape}')
print(f'Taille de l\'ensemble d\'entraînement y: {y_train.shape}')
print(f'Taille de l\'ensemble de test y: {y_test.shape}')