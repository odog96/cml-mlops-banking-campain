"""
Module 1 Utilities
==================

Helper functions for data processing, feature engineering, and model training
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def engineer_customer_engagement_score(df):
    """
    Create a customer engagement score feature
    
    This demonstrates feature engineering by combining multiple signals:
    - Contact frequency (campaign contacts)
    - Previous campaign responsiveness
    - Temporal engagement patterns
    
    Args:
        df: DataFrame with banking data
        
    Returns:
        DataFrame with new 'engagement_score' column
    """
    df = df.copy()
    
    # Normalize campaign contacts (0-1 scale)
    campaign_norm = (df['campaign'] - df['campaign'].min()) / (df['campaign'].max() - df['campaign'].min())
    
    # Previous campaign success (0 or 1)
    prev_success = (df['poutcome'] == 'success').astype(int)
    
    # Recent contact indicator (contacted in last 30 days)
    recent_contact = (df['pdays'] < 30).astype(int)
    recent_contact = recent_contact.replace({True: 1, False: 0})
    
    # Days since last contact (inverted and normalized)
    # pdays=999 means never contacted, we'll treat this as 0 engagement
    days_engagement = df['pdays'].apply(lambda x: 0 if x == 999 else 1 - (x / 999))
    
    # Weighted combination
    engagement_score = (
        0.3 * campaign_norm +           # Current campaign effort
        0.3 * prev_success +             # Historical responsiveness  
        0.2 * recent_contact +           # Recency of contact
        0.2 * days_engagement            # Overall contact history
    )
    
    df['engagement_score'] = engagement_score
    
    return df


def engineer_features(df):
    """
    Apply all feature engineering transformations
    
    Args:
        df: Raw DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Add customer engagement score
    df = engineer_customer_engagement_score(df)
    
    # Age groups (can help with interpretability)
    df['age_group'] = pd.cut(
        df['age'], 
        bins=[0, 30, 40, 50, 60, 100],
        labels=['<30', '30-40', '40-50', '50-60', '60+']
    )
    
    # Economic indicator categories (using employment variation rate)
    df['emp_var_category'] = pd.cut(
        df['emp.var.rate'],
        bins=[-np.inf, -1, 0, 1, np.inf],
        labels=['very_low', 'low', 'neutral', 'high']
    )
    
    # Contact duration categories (important predictor)
    df['duration_category'] = pd.cut(
        df['duration'],
        bins=[0, 180, 360, 600, np.inf],
        labels=['very_short', 'short', 'medium', 'long']
    )
    
    return df


def preprocess_for_training(df, target_column='y', include_engagement=True):
    """
    Prepare data for model training
    
    Args:
        df: DataFrame with features
        target_column: Name of target variable
        include_engagement: Whether to include the engineered engagement score
        
    Returns:
        X (features), y (target), feature_names, categorical_cols
    """
    df = df.copy()
    
    # Define feature sets
    numeric_features = [
        'age', 'duration', 'campaign', 'pdays', 'previous',
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
    ]

    categorical_features = [
        'job', 'marital', 'education', 'default',
        'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome'
    ]
    
    # Add engagement score if requested
    if include_engagement and 'engagement_score' in df.columns:
        numeric_features.append('engagement_score')
    
    # Encode target variable
    y = (df[target_column] == 'yes').astype(int)
    
    # Select features
    X = df[numeric_features + categorical_features].copy()
    
    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    return X_encoded, y, X_encoded.columns.tolist(), categorical_features


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def calculate_feature_importance_summary(feature_importance_dict, top_n=10):
    """
    Create a summary of feature importance
    
    Args:
        feature_importance_dict: Dictionary of {feature: importance}
        top_n: Number of top features to return
        
    Returns:
        DataFrame with top features
    """
    importance_df = pd.DataFrame({
        'feature': feature_importance_dict.keys(),
        'importance': feature_importance_dict.values()
    })
    
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    return importance_df
