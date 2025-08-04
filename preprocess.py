import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

def load_and_preprocess(filepath):
    """Load and preprocess Titanic dataset"""
    df = pd.read_csv(filepath)
    
    # 1. Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # 2. Feature engineering
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 
                                     'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 3. Drop unnecessary columns
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    
    return df

def visualize_outliers(df, features):
    """Generate boxplots for specified features"""
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x=df[feature])
        plt.title(f'Boxplot of {feature}')
    plt.tight_layout()
    plt.savefig('titanic_outliers.png')
    plt.close()
    print("Saved outlier visualizations to 'titanic_outliers.png'")

def preprocess_data(df):
    """Final preprocessing steps"""
    # One-hot encoding
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'Pclass'], drop_first=True)
    
    # Robust scaling for numerical features
    scaler = RobustScaler()
    numeric_features = ['Age', 'Fare', 'FamilySize']
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    return df

# Main execution
if __name__ == "__main__":
    # Load and preprocess
    df = load_and_preprocess('Titanic-Dataset.csv')
    
    # Visualize outliers before processing
    visualize_outliers(df, ['Age', 'Fare', 'FamilySize', 'SibSp'])
    
    # Final preprocessing
    processed_df = preprocess_data(df)
    
    # Save processed data
    processed_df.to_csv('titanic_processed_clean.csv', index=False)
    print("\nPreprocessing complete!")
    print(f"Final dataset shape: {processed_df.shape}")
    print("First 5 rows:")
    print(processed_df.head())