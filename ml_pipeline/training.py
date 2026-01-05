import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Unified model training interface for multiple ML frameworks
    """
    
    def __init__(self, experiment_name='default', tracking_uri=None):
        self.experiment_name = experiment_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        logger.info(f"Initialized trainer for experiment: {experiment_name}")
    
    def train_sklearn_model(self, model, X_train, y_train, X_val=None, y_val=None, 
                           params=None, log_model=True):
        """
        Train scikit-learn model with MLflow tracking
        """
        with mlflow.start_run():
            # Log parameters
            if params:
                mlflow.log_params(params)
            
            # Train model
            logger.info("Training scikit-learn model...")
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Log training time
            mlflow.log_metric("training_time", training_time)
            
            # Evaluate on training set
            train_pred = model.predict(X_train)
            train_metrics = self._calculate_metrics(y_train, train_pred)
            
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                val_pred = model.predict(X_val)
                val_metrics = self._calculate_metrics(y_val, val_pred)
                
                for metric_name, value in val_metrics.items():
                    mlflow.log_metric(f"val_{metric_name}", value)
            
            # Log model
            if log_model:
                mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            return model, train_metrics
    
    def train_tensorflow_model(self, model, X_train, y_train, X_val=None, y_val=None,
                               epochs=10, batch_size=32, callbacks=None):
        """
        Train TensorFlow/Keras model with MLflow tracking
        """
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            
            # Prepare validation data
            validation_data = (X_val, y_val) if X_val is not None else None
            
            # Train model
            logger.info("Training TensorFlow model...")
            start_time = time.time()
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            mlflow.log_metric("training_time", training_time)
            
            # Log metrics from history
            for metric_name in history.history.keys():
                for epoch, value in enumerate(history.history[metric_name]):
                    mlflow.log_metric(metric_name, value, step=epoch)
            
            # Log model
            mlflow.tensorflow.log_model(model, "model")
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            return model, history
    
    def train_pytorch_model(self, model, train_loader, val_loader=None, 
                           optimizer=None, criterion=None, epochs=10, device='cpu'):
        """
        Train PyTorch model with MLflow tracking
        """
        import torch
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("device", device)
            
            model = model.to(device)
            
            logger.info("Training PyTorch model...")
            start_time = time.time()
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                
                # Validation phase
                if val_loader:
                    model.eval()
                    val_loss = 0.0
                    
                    with torch.no_grad():
                        for data, target in val_loader:
                            data, target = data.to(device), target.to(device)
                            output = model(data)
                            loss = criterion(output, target)
                            val_loss += loss.item()
                    
                    avg_val_loss = val_loss / len(val_loader)
                    mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
            
            training_time = time.time() - start_time
            mlflow.log_metric("training_time", training_time)
            
            # Log model
            mlflow.pytorch.log_model(model, "model")
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            return model
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate appropriate metrics based on problem type
        """
        metrics = {}
        
        # Check if classification or regression
        if len(set(y_true)) < 20:  # Likely classification
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        else:  # Likely regression
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = mean_squared_error(y_true, y_pred, squared=False)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
        
        return metrics
    
    def hyperparameter_tuning(self, model_class, X_train, y_train, param_grid, 
                             cv=5, scoring='accuracy'):
        """
        Perform hyperparameter tuning using GridSearchCV
        """
        from sklearn.model_selection import GridSearchCV
        
        logger.info("Starting hyperparameter tuning...")
        
        with mlflow.start_run():
            grid_search = GridSearchCV(
                model_class(),
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Log best parameters
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric(f"best_{scoring}", grid_search.best_score_)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best {scoring}: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_, grid_search.best_params_