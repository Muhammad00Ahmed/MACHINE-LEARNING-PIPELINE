import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for machine learning
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = None
        
    def load_data(self, filepath, **kwargs):
        """
        Load data from various file formats
        """
        logger.info(f"Loading data from {filepath}")
        
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath, **kwargs)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath, **kwargs)
        elif filepath.endswith('.parquet'):
            return pd.read_parquet(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
    
    def handle_missing_values(self, df, strategy='mean', columns=None):
        """
        Handle missing values in the dataset
        
        Args:
            df: DataFrame
            strategy: 'mean', 'median', 'most_frequent', 'constant'
            columns: List of columns to impute (None for all)
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        if columns is None:
            columns = df.columns
        
        for col in columns:
            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64']:
                    imputer = SimpleImputer(strategy=strategy)
                    df[col] = imputer.fit_transform(df[[col]])
                    self.imputers[col] = imputer
                else:
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[col] = imputer.fit_transform(df[[col]])
                    self.imputers[col] = imputer
        
        return df
    
    def remove_outliers(self, df, columns=None, method='iqr', threshold=1.5):
        """
        Remove outliers from numerical columns
        
        Args:
            df: DataFrame
            columns: List of columns to check (None for all numerical)
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or z-score threshold
        """
        logger.info(f"Removing outliers using {method} method")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        df_clean = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & 
                                   (df_clean[col] <= upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df_clean = df_clean[z_scores < threshold]
        
        logger.info(f"Removed {len(df) - len(df_clean)} outliers")
        return df_clean
    
    def encode_categorical(self, df, columns=None, method='label'):
        """
        Encode categorical variables
        
        Args:
            df: DataFrame
            columns: List of columns to encode (None for all categorical)
            method: 'label' or 'onehot'
        """
        logger.info(f"Encoding categorical variables using {method} encoding")
        
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns
        
        df_encoded = df.copy()
        
        for col in columns:
            if method == 'label':
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder
            
            elif method == 'onehot':
                dummies = pd.get_dummies(df[col], prefix=col)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
        
        return df_encoded
    
    def scale_features(self, df, columns=None, method='standard'):
        """
        Scale numerical features
        
        Args:
            df: DataFrame
            columns: List of columns to scale (None for all numerical)
            method: 'standard', 'minmax', 'robust'
        """
        logger.info(f"Scaling features using {method} scaling")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        df_scaled = df.copy()
        
        if method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        
        df_scaled[columns] = scaler.fit_transform(df[columns])
        self.scalers['feature_scaler'] = scaler
        
        return df_scaled
    
    def feature_engineering(self, df):
        """
        Create new features from existing ones
        """
        logger.info("Performing feature engineering")
        
        df_engineered = df.copy()
        
        # Example: Create polynomial features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # Square features
            df_engineered[f'{col}_squared'] = df[col] ** 2
            
            # Log transform (for positive values)
            if (df[col] > 0).all():
                df_engineered[f'{col}_log'] = np.log1p(df[col])
        
        return df_engineered
    
    def split_data(self, df, target_column, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        """
        logger.info(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test")
        
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.feature_names = X.columns.tolist()
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance(self, model, top_n=10):
        """
        Get feature importance from trained model
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            feature_importance = pd.DataFrame({
                'feature': [self.feature_names[i] for i in indices],
                'importance': importances[indices]
            })
            
            return feature_importance
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return None
    
    def save_preprocessor(self, filepath):
        """
        Save preprocessor state
        """
        import joblib
        joblib.dump({
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers,
            'feature_names': self.feature_names
        }, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """
        Load preprocessor state
        """
        import joblib
        state = joblib.load(filepath)
        self.scalers = state['scalers']
        self.encoders = state['encoders']
        self.imputers = state['imputers']
        self.feature_names = state['feature_names']
        logger.info(f"Preprocessor loaded from {filepath}")