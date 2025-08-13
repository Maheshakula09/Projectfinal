# -*- coding: utf-8 -*-
"""
Fixed Base Model Trainer for Medical Abstract Bias Detection
Bug fixes for prediction handling and error management
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
import joblib
import warnings
import logging
from datetime import datetime
from typing import Dict, Tuple, List
import os
import json

# Set up logging and warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class FixedBaseModelTrainer:
    """
    Fixed trainer for traditional ML models with proper error handling
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the base model trainer"""
        self.random_state = random_state
        self.results = {}
        self.best_model = None
        self.X_test = None
        self.y_test = None
        
        # Set random seeds
        np.random.seed(random_state)
        
    def load_data(self, data_path: str, text_column: str = 'abstract', 
                  label_column: str = 'bias_label') -> Tuple[pd.Series, pd.Series]:
        """Load and validate the dataset"""
        logger.info(f"Loading data from {data_path}...")
        
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} samples")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        
        # Check required columns
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"Required columns {[text_column, label_column]} not found")
        
        # Remove empty abstracts
        original_size = len(df)
        df = df.dropna(subset=[text_column, label_column])
        df = df[df[text_column].str.strip() != '']
        
        logger.info(f"Removed {original_size - len(df)} empty/invalid rows")
        
        X = df[text_column].astype(str)
        y = df[label_column].astype(int)
        
        # Check class distribution
        class_counts = y.value_counts()
        logger.info(f"Class distribution:")
        for label, count in class_counts.items():
            logger.info(f"  Class {label}: {count} samples ({count/len(y)*100:.1f}%)")
        
        # Check text statistics
        avg_length = X.str.len().mean()
        avg_words = X.str.split().str.len().mean()
        logger.info(f"Text statistics:")
        logger.info(f"  Average length: {avg_length:.0f} characters")
        logger.info(f"  Average words: {avg_words:.0f} words")
        
        return X, y
    
    def create_vectorizers(self) -> Dict:
        """Create different text vectorizers"""
        vectorizers = {
            'tfidf_bigram': TfidfVectorizer(
                max_features=8000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                lowercase=True,
                strip_accents='unicode'
            ),
            'tfidf_unigram': TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 1),
                min_df=2,
                max_df=0.95,
                lowercase=True,
                strip_accents='unicode'
            ),
            'count_bigram': CountVectorizer(
                max_features=8000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                lowercase=True
            )
        }
        
        return vectorizers
    
    def create_models(self) -> Dict:
        """Create different ML models"""
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=2000,
                class_weight='balanced',
                solver='lbfgs'
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                class_weight='balanced',
                max_depth=20,
                min_samples_split=5,
                n_jobs=-1
            ),
            'svm': SVC(
                random_state=self.random_state,
                probability=True,
                class_weight='balanced',
                kernel='linear',
                C=1.0
            ),
            'naive_bayes': MultinomialNB(
                alpha=1.0
            )
        }
        
        return models
    
    def get_hyperparameter_grids(self) -> Dict:
        """Get hyperparameter grids for different models"""
        param_grids = {
            'logistic_regression': {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['liblinear', 'lbfgs']
            },
            'random_forest': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [10, 20, None]
            },
            'svm': {
                'classifier__C': [0.1, 1, 10]
            },
            'naive_bayes': {
                'classifier__alpha': [0.1, 1.0, 10.0]
            }
        }
        
        return param_grids
    
    def safe_predict_proba(self, pipeline, X_test):
        """Safely get prediction probabilities"""
        try:
            return pipeline.predict_proba(X_test)
        except Exception as e:
            logger.warning(f"Could not get probabilities: {e}")
            return None
    
    def calculate_metrics(self, y_true: np.array, y_pred: np.array, 
                         y_pred_proba: np.array = None) -> Dict:
        """Calculate comprehensive evaluation metrics with error handling"""
        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            # Per-class metrics
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'precision_per_class': precision_per_class.tolist() if precision_per_class is not None else [],
                'recall_per_class': recall_per_class.tolist() if recall_per_class is not None else [],
                'f1_per_class': f1_per_class.tolist() if f1_per_class is not None else [],
                'confusion_matrix': cm.tolist() if cm is not None else [],
                'support': support.tolist() if support is not None else []
            }
            
            # AUC metrics if probabilities provided
            if y_pred_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:  # Binary classification
                        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] >= 2:
                            y_scores = y_pred_proba[:, 1]
                        else:
                            y_scores = y_pred_proba
                        
                        auc_roc = roc_auc_score(y_true, y_scores)
                        auc_pr = average_precision_score(y_true, y_scores)
                        
                        metrics['auc_roc'] = float(auc_roc)
                        metrics['auc_pr'] = float(auc_pr)
                    else:
                        metrics['auc_roc'] = 0.0
                        metrics['auc_pr'] = 0.0
                except Exception as e:
                    logger.warning(f"Could not calculate AUC metrics: {e}")
                    metrics['auc_roc'] = 0.0
                    metrics['auc_pr'] = 0.0
            else:
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
            
            # Additional metrics for binary classification
            if len(np.unique(y_true)) == 2 and cm.size == 4:
                try:
                    tn, fp, fn, tp = cm.ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    
                    metrics['specificity'] = float(specificity)
                    metrics['sensitivity'] = float(sensitivity)
                except:
                    metrics['specificity'] = 0.0
                    metrics['sensitivity'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Return default metrics in case of error
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'auc_roc': 0.0,
                'auc_pr': 0.0,
                'specificity': 0.0,
                'sensitivity': 0.0,
                'precision_per_class': [],
                'recall_per_class': [],
                'f1_per_class': [],
                'confusion_matrix': [],
                'support': []
            }
    
    def train_single_model(self, model_name: str, vectorizer_name: str, 
                          X_train: pd.Series, y_train: pd.Series) -> Pipeline:
        """Train a single model-vectorizer combination with error handling"""
        logger.info(f"Training {model_name} + {vectorizer_name}...")
        
        try:
            vectorizers = self.create_vectorizers()
            models = self.create_models()
            
            # Create pipeline with SMOTE for class imbalance
            pipeline = ImbalancedPipeline([
                ('vectorizer', vectorizers[vectorizer_name]),
                ('smote', SMOTE(random_state=self.random_state, k_neighbors=min(3, len(y_train)//2))),
                ('classifier', models[model_name])
            ])
            
            # Get parameter grid
            param_grids = self.get_hyperparameter_grids()
            
            if model_name in param_grids:
                # Perform grid search
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)  # Reduced folds
                grid_search = GridSearchCV(
                    pipeline,
                    param_grids[model_name],
                    cv=cv,
                    scoring='f1_weighted',
                    n_jobs=1,  # Single job to avoid memory issues
                    verbose=0,
                    error_score='raise'
                )
                
                grid_search.fit(X_train, y_train)
                
                logger.info(f"Best parameters: {grid_search.best_params_}")
                logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")
                
                return grid_search.best_estimator_
            else:
                # Train without grid search
                pipeline.fit(X_train, y_train)
                return pipeline
                
        except Exception as e:
            logger.error(f"Error training {model_name}_{vectorizer_name}: {e}")
            raise
    
    def train_all_models(self, X: pd.Series, y: pd.Series, 
                        test_size: float = 0.2) -> Dict:
        """Train all model-vectorizer combinations with robust error handling"""
        logger.info("Starting comprehensive model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        # Store test data for later use
        self.X_test = X_test
        self.y_test = y_test
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Model-vectorizer combinations to try (reduced for reliability)
        combinations = [
            ('logistic_regression', 'tfidf_bigram'),
            ('logistic_regression', 'tfidf_unigram'),
            ('random_forest', 'tfidf_bigram'),
            ('random_forest', 'count_bigram'),
            ('svm', 'tfidf_bigram'),
            ('naive_bayes', 'count_bigram')
        ]
        
        results = {}
        best_f1 = 0
        best_combination = None
        successful_models = 0
        
        for model_name, vec_name in combinations:
            combination_name = f"{model_name}_{vec_name}"
            
            try:
                # Train model
                pipeline = self.train_single_model(model_name, vec_name, X_train, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                y_pred_proba = self.safe_predict_proba(pipeline, X_test)
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
                
                # Store results
                results[combination_name] = {
                    'pipeline': pipeline,
                    'metrics': metrics,
                    'predictions': y_pred.tolist() if y_pred is not None else [],
                    'model_name': model_name,
                    'vectorizer_name': vec_name
                }
                
                # Check if this is the best model
                current_f1 = metrics['f1_score']
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_combination = combination_name
                    self.best_model = pipeline
                
                successful_models += 1
                logger.info(f"✅ {combination_name}: F1={current_f1:.4f}, Accuracy={metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error training {combination_name}: {e}")
                continue
        
        if successful_models == 0:
            logger.error("❌ No models were successfully trained!")
            return {}
        
        logger.info(f"\n🏆 Best combination: {best_combination} (F1: {best_f1:.4f})")
        logger.info(f"✅ Successfully trained {successful_models}/{len(combinations)} models")
        
        self.results = results
        return results
    
    def evaluate_best_model(self) -> Dict:
        """Detailed evaluation of the best model"""
        if not self.results:
            raise ValueError("No models trained yet. Run train_all_models() first.")
        
        best_model_name = max(self.results.keys(), 
                             key=lambda k: self.results[k]['metrics']['f1_score'])
        best_result = self.results[best_model_name]
        metrics = best_result['metrics']
        
        logger.info(f"\n📊 Detailed evaluation for {best_model_name}:")
        logger.info("="*60)
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        
        if metrics['auc_roc'] > 0:
            logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
            logger.info(f"AUC-PR: {metrics['auc_pr']:.4f}")
        
        if 'specificity' in metrics:
            logger.info(f"Specificity: {metrics['specificity']:.4f}")
            logger.info(f"Sensitivity: {metrics['sensitivity']:.4f}")
        
        # Print classification report
        y_pred = best_result['predictions']
        if y_pred:
            print("\n📋 Classification Report:")
            print(classification_report(self.y_test, y_pred, target_names=['Non-Biased', 'Biased']))
        
        return best_result
    
    def create_visualizations(self, save_plots: bool = True) -> None:
        """Create visualizations with error handling"""
        logger.info("📊 Creating visualizations...")
        
        if not self.results:
            logger.warning("⚠️  No results available for visualization")
            return
        
        try:
            # Model Comparison Plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Base Model Performance Comparison', fontsize=16, fontweight='bold')
            
            model_names = list(self.results.keys())
            metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall']
            metric_labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
            
            for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
                ax = axes[idx//2, idx%2]
                values = [self.results[model]['metrics'][metric] for model in model_names]
                
                bars = ax.bar(range(len(model_names)), values, 
                             color=plt.cm.Set3(np.linspace(0, 1, len(model_names))))
                ax.set_title(f'{label} Comparison')
                ax.set_ylabel(label)
                ax.set_xticks(range(len(model_names)))
                ax.set_xticklabels([name.replace('_', '\n') for name in model_names], 
                                  rotation=45, ha='right', fontsize=9)
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('base_model_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def save_results(self, save_dir: str = 'base_model_outputs') -> None:
        """Save trained models and results"""
        try:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save best model
            if self.best_model:
                model_path = os.path.join(save_dir, f'best_base_model_{timestamp}.pkl')
                joblib.dump(self.best_model, model_path)
                logger.info(f"💾 Best model saved to {model_path}")
            
            # Save results (metrics only, not the pipeline objects)
            results_summary = {}
            for model_name, result in self.results.items():
                results_summary[model_name] = {
                    'metrics': result['metrics'],
                    'model_name': result['model_name'],
                    'vectorizer_name': result['vectorizer_name']
                }
            
            results_path = os.path.join(save_dir, f'base_model_results_{timestamp}.json')
            with open(results_path, 'w') as f:
                json.dump(results_summary, f, indent=2, default=str)
            logger.info(f"💾 Results saved to {results_path}")
            
            # Save report
            self._generate_report(save_dir, timestamp)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _generate_report(self, save_dir: str, timestamp: str) -> None:
        """Generate detailed text report"""
        try:
            report_path = os.path.join(save_dir, f'base_model_report_{timestamp}.txt')
            with open(report_path, 'w') as f:
                f.write("BASE MODEL TRAINING REPORT\n")
                f.write("="*60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if self.results:
                    # Best model info
                    best_model_name = max(self.results.keys(), 
                                         key=lambda k: self.results[k]['metrics']['f1_score'])
                    best_metrics = self.results[best_model_name]['metrics']
                    
                    f.write(f"BEST MODEL: {best_model_name}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"F1-Score: {best_metrics['f1_score']:.4f}\n")
                    f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")
                    f.write(f"Precision: {best_metrics['precision']:.4f}\n")
                    f.write(f"Recall: {best_metrics['recall']:.4f}\n")
                    
                    if best_metrics['auc_roc'] > 0:
                        f.write(f"AUC-ROC: {best_metrics['auc_roc']:.4f}\n")
                        f.write(f"AUC-PR: {best_metrics['auc_pr']:.4f}\n")
                    
                    f.write(f"\nALL MODEL RESULTS:\n")
                    f.write("-" * 40 + "\n")
                    
                    for model_name, result in self.results.items():
                        metrics = result['metrics']
                        f.write(f"\n{model_name}:\n")
                        f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
                        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                        f.write(f"  Precision: {metrics['precision']:.4f}\n")
                        f.write(f"  Recall: {metrics['recall']:.4f}\n")
            
            logger.info(f"💾 Report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")


def main():
    """Main function to train base models"""
    print("="*70)
    print("🤖 FIXED BASE MODEL TRAINER FOR BIAS DETECTION")
    print("="*70)
    
    # Configuration - UPDATE THIS PATH
    DATA_PATH = "/Users/satya/SyntheticBias/cleaned_abstracts_aggressive.csv"
    
    print(f"📁 Data file: {DATA_PATH}")
    print("🎯 Models: Logistic Regression, Random Forest, SVM, Naive Bayes")
    print("📊 Vectorizers: TF-IDF, Count Vectorizer")
    print("⚖️  Class balancing: SMOTE oversampling")
    print("🔧 Enhanced error handling and robustness")
    print("="*70)
    
    # Check if file exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: File {DATA_PATH} not found!")
        print("📁 Available CSV files:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"   📄 {file}")
        return None, None
    
    # Initialize trainer
    trainer = FixedBaseModelTrainer(random_state=42)
    
    try:
        # Load data
        X, y = trainer.load_data(DATA_PATH)
        
        # Train all models
        print("\n🚀 Starting model training...")
        results = trainer.train_all_models(X, y, test_size=0.2)
        
        if not results:
            print("❌ No models were successfully trained!")
            return None, None
        
        # Evaluate best model
        print("\n📊 Evaluating best model...")
        best_result = trainer.evaluate_best_model()
        
        # Create visualizations
        print("\n📈 Creating visualizations...")
        trainer.create_visualizations(save_plots=True)
        
        # Save results
        print("\n💾 Saving results...")
        trainer.save_results()
        
        # Print summary
        best_model_name = max(results.keys(), 
                             key=lambda k: results[k]['metrics']['f1_score'])
        best_metrics = results[best_model_name]['metrics']
        
        print("\n" + "="*70)
        print("🏆 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"✅ Best Model: {best_model_name}")
        print(f"🎯 F1-Score: {best_metrics['f1_score']:.4f}")
        print(f"🎯 Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"🎯 Precision: {best_metrics['precision']:.4f}")
        print(f"🎯 Recall: {best_metrics['recall']:.4f}")
        
        if best_metrics['auc_roc'] > 0:
            print(f"🎯 AUC-ROC: {best_metrics['auc_roc']:.4f}")
        
        print(f"\n📁 Generated Files:")
        print(f"   📊 base_model_comparison.png")
        print(f"   💾 base_model_outputs/ directory")
        print("="*70)
        
        return trainer, results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ ERROR: {str(e)}")
        return None, None


if __name__ == "__main__":
    print("📋 REQUIREMENTS:")
    print("pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn joblib")
    print("\n🔧 IMPROVEMENTS:")
    print("✅ Fixed prediction handling bugs")
    print("✅ Enhanced error handling and logging")
    print("✅ Reduced memory usage")
    print("✅ More robust hyperparameter tuning")
    print("✅ Better progress tracking")
    print("\n🚀 STARTING FIXED BASE MODEL TRAINING...")
    print()
    
    trainer, results = main()
    
    if trainer and results:
        print("\n🎉 SUCCESS! Base models trained successfully.")
        print("📈 Check the generated plots and saved models.")
        print("➡️  Next: Run the BioBERT trainer.")
    else:
        print("\n❌ Training failed. Please check the error messages above.")