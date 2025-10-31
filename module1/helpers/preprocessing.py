"""
Preprocessing Pipeline for Model Training and Inference
========================================================

Contains:
1. FeatureEngineer: Encapsulates all feature engineering logic
2. PreprocessingPipeline: Handles scaling and encoding for model input
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class FeatureEngineer:
    """
    Encapsulates all feature engineering logic for both training and inference.

    This class contains the logic to create engineered features that should be
    applied consistently to both training and new data at inference time.
    """

    def __init__(self):
        """Initialize the feature engineer."""
        pass

    def engineer_customer_engagement_score(self, df):
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

    def engineer_features(self, df):
        """
        Apply all feature engineering transformations

        This method should be called on raw data to create engineered features
        before preprocessing (scaling/encoding).

        Args:
            df: Raw DataFrame

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Add customer engagement score
        df = self.engineer_customer_engagement_score(df)

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

    def transform(self, df):
        """
        Apply feature engineering to data (alias for engineer_features for consistency).

        Args:
            df: Raw DataFrame

        Returns:
            DataFrame with engineered features
        """
        return self.engineer_features(df)


class PreprocessingPipeline:
    """
    Handles scaling of numeric features and encoding of categorical features.

    This pipeline should be fitted on training data and then applied to both
    training (after fitting) and inference data to ensure consistency.
    """

    def __init__(self, numeric_features=None, categorical_features=None,
                 include_engagement=True):
        """
        Initialize the preprocessing pipeline.

        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            include_engagement: Whether engagement_score is included
        """
        # Default features
        if numeric_features is None:
            numeric_features = [
                'age', 'duration', 'campaign', 'pdays', 'previous',
                'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
            ]

        if categorical_features is None:
            categorical_features = [
                'job', 'marital', 'education', 'default',
                'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome',
                'age_group', 'emp_var_category', 'duration_category'
            ]

        self.numeric_features = numeric_features.copy()
        self.categorical_features = categorical_features.copy()
        self.include_engagement = include_engagement

        # Add engagement_score to numeric features if requested
        if include_engagement and 'engagement_score' not in self.numeric_features:
            self.numeric_features.append('engagement_score')

        # Build the preprocessing pipeline
        self.pipeline = None
        self._build_pipeline()

    def _build_pipeline(self):
        """Build the sklearn preprocessing pipeline."""
        # Define transformations for each feature type
        numeric_transformer = StandardScaler()

        categorical_transformer = OneHotEncoder(
            handle_unknown='ignore',  # Handle unseen categories at inference
            sparse_output=False,
            drop='first'  # Drop first category to avoid multicollinearity
        )

        # Combine transformations using ColumnTransformer
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )

    def fit(self, X):
        """
        Fit the preprocessing pipeline on training data.

        Args:
            X: Training features (DataFrame or array)

        Returns:
            self
        """
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        """
        Transform data using the fitted pipeline.

        Args:
            X: Features to transform (DataFrame or array)

        Returns:
            Transformed features (numpy array)
        """
        return self.pipeline.transform(X)

    def fit_transform(self, X):
        """
        Fit and transform data in one step.

        Args:
            X: Features to fit and transform

        Returns:
            Transformed features (numpy array)
        """
        return self.pipeline.fit_transform(X)

    def get_feature_names(self):
        """
        Get the names of features after transformation.

        Returns:
            List of feature names after preprocessing
        """
        if self.pipeline is None:
            raise ValueError("Pipeline has not been fitted yet")

        feature_names = []

        # Get numeric feature names (scaled, no name change)
        feature_names.extend(self.numeric_features)

        # Get categorical feature names (one-hot encoded)
        cat_encoder = self.pipeline.named_transformers_['cat']
        cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_features)
        feature_names.extend(cat_feature_names)

        return feature_names


def preprocess_for_training(df, target_column='y', include_engagement=True):
    """
    Prepare data for model training using FeatureEngineer and PreprocessingPipeline.

    This function orchestrates the full preprocessing workflow:
    1. Apply feature engineering
    2. Extract target variable
    3. Create and fit preprocessing pipeline

    Args:
        df: DataFrame with raw features
        target_column: Name of target variable
        include_engagement: Whether to include engineered engagement score

    Returns:
        X_processed (transformed features), y (target), preprocessor, feature_engineer
    """
    df = df.copy()

    # Step 1: Feature Engineering
    feature_engineer = FeatureEngineer()
    df_engineered = feature_engineer.transform(df)

    # Step 2: Extract target
    y = (df_engineered[target_column] == 'yes').astype(int)

    # Step 3: Create and fit preprocessing pipeline
    preprocessor = PreprocessingPipeline(include_engagement=include_engagement)

    # Define features for preprocessing
    numeric_features = [
        'age', 'duration', 'campaign', 'pdays', 'previous',
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
    ]

    categorical_features = [
        'job', 'marital', 'education', 'default',
        'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome',
        'age_group', 'emp_var_category', 'duration_category'
    ]

    if include_engagement:
        numeric_features.append('engagement_score')

    preprocessor.numeric_features = numeric_features
    preprocessor.categorical_features = categorical_features
    preprocessor._build_pipeline()

    # Select features and fit preprocessor
    X = df_engineered[numeric_features + categorical_features].copy()
    X_processed = preprocessor.fit_transform(X)

    # Convert to DataFrame for better compatibility
    X_processed = pd.DataFrame(
        X_processed,
        columns=preprocessor.get_feature_names()
    )

    return X_processed, y, preprocessor, feature_engineer


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.

    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
