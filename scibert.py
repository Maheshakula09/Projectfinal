#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SciBERT Trainer for Medical Abstract Bias Detection
CPU-Only version for maximum compatibility
"""

# DISABLE MPS BEFORE ANY IMPORTS
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Force CPU-only PyTorch
import torch
torch.backends.mps.is_available = lambda: False
torch.cuda.is_available = lambda: False

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
import warnings
import logging
from datetime import datetime
from typing import Dict, Tuple, List
import json
from tqdm.auto import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class BiasDataset(Dataset):
    """Custom Dataset class for bias detection with SciBERT"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class WeightedTrainer(Trainer):
    """Custom Trainer with class weights for SciBERT"""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with class weights"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                           labels.view(-1))
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


class SciBERTTrainer:
    """SciBERT trainer for medical abstract bias detection"""
    
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased", 
                 random_state: int = 42):
        """Initialize SciBERT trainer"""
        self.model_name = model_name
        self.random_state = random_state
        self.device = torch.device('cpu')  # FORCE CPU for stability
        
        logger.info(f"🔬 SCIBERT TRAINER - CPU MODE")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {model_name}")
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.results = {}
        self.test_dataset = None
        self.test_labels = None
        
    def load_data(self, data_path: str, text_column: str = 'abstract', 
                  label_column: str = 'bias_label') -> Tuple[List[str], List[int]]:
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
            available_cols = df.columns.tolist()
            raise ValueError(f"Required columns {[text_column, label_column]} not found. Available columns: {available_cols}")
        
        # Clean data
        original_size = len(df)
        df = df.dropna(subset=[text_column, label_column])
        df = df[df[text_column].str.strip() != '']
        
        logger.info(f"Removed {original_size - len(df)} empty/invalid rows")
        
        # Convert to lists
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].astype(int).tolist()
        
        # Check class distribution
        unique_labels = np.unique(labels)
        logger.info(f"Classes found: {unique_labels}")
        for label in unique_labels:
            count = labels.count(label)
            logger.info(f"  Class {label}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        # Check text statistics
        text_lengths = [len(text) for text in texts]
        avg_length = np.mean(text_lengths)
        max_length = max(text_lengths)
        
        logger.info(f"Text statistics:")
        logger.info(f"  Average length: {avg_length:.0f} characters")
        logger.info(f"  Maximum length: {max_length} characters")
        
        return texts, labels
    
    def split_data(self, texts: List[str], labels: List[int], 
                   val_size: float = 0.15, test_size: float = 0.15) -> Tuple:
        """Split data into train/validation/test sets"""
        # Split data: train/temp, then temp -> val/test
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=(val_size + test_size), 
            random_state=self.random_state, stratify=labels
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, 
            test_size=(test_size / (val_size + test_size)),
            random_state=self.random_state, stratify=temp_labels
        )
        
        logger.info(f"Data split:")
        logger.info(f"  Training set: {len(train_texts)} samples")
        logger.info(f"  Validation set: {len(val_texts)} samples") 
        logger.info(f"  Test set: {len(test_texts)} samples")
        
        return (train_texts, val_texts, test_texts, 
                train_labels, val_labels, test_labels)
    
    def initialize_model_and_tokenizer(self, num_labels: int = 2) -> None:
        """Initialize SciBERT tokenizer and model"""
        logger.info("Initializing SciBERT tokenizer and model...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("✅ SciBERT tokenizer loaded successfully")
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                output_attentions=False,
                output_hidden_states=False,
            )
            
            # Move model to CPU
            self.model.to(self.device)
            logger.info("✅ SciBERT model loaded and moved to CPU")
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def create_datasets(self, train_texts: List[str], val_texts: List[str], 
                       test_texts: List[str], train_labels: List[int], 
                       val_labels: List[int], test_labels: List[int],
                       max_length: int = 256) -> Tuple:
        """Create PyTorch datasets"""
        logger.info("Creating datasets...")
        
        # Analyze text lengths for SciBERT optimization
        all_texts = train_texts[:50] + val_texts[:25] + test_texts[:25]  # Sample
        text_lengths = [len(self.tokenizer.tokenize(text)) for text in all_texts]
        avg_tokens = np.mean(text_lengths)
        max_tokens = max(text_lengths)
        
        logger.info(f"Token statistics for SciBERT (sample):")
        logger.info(f"  Average tokens: {avg_tokens:.0f}")
        logger.info(f"  Maximum tokens: {max_tokens}")
        
        train_dataset = BiasDataset(train_texts, train_labels, self.tokenizer, max_length)
        val_dataset = BiasDataset(val_texts, val_labels, self.tokenizer, max_length)
        test_dataset = BiasDataset(test_texts, test_labels, self.tokenizer, max_length)
        
        # Store test data for later evaluation
        self.test_dataset = test_dataset
        self.test_labels = test_labels
        
        logger.info(f"✅ Datasets created:")
        logger.info(f"  Train: {len(train_dataset)} samples")
        logger.info(f"  Validation: {len(val_dataset)} samples")
        logger.info(f"  Test: {len(test_dataset)} samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation during training"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def setup_training_arguments(self, output_dir: str = './scibert_results',
                               num_epochs: int = 3, batch_size: int = 16,
                               learning_rate: float = 2e-5) -> TrainingArguments:
        """Setup training arguments optimized for SciBERT"""
        
        # CPU-optimized settings
        batch_size = 2  # Conservative for CPU
        logger.info(f"🔧 Using batch size {batch_size} for CPU training")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=50,
            weight_decay=0.01,
            learning_rate=learning_rate,  # SciBERT-optimized LR
            logging_dir=f'{output_dir}/logs',
            logging_steps=25,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to=None,
            dataloader_pin_memory=False,
            fp16=False,
            gradient_accumulation_steps=4,  # Effective batch size = 8
            remove_unused_columns=False,
            dataloader_num_workers=0,
            push_to_hub=False,
            seed=42,
            data_seed=42,
            skip_memory_metrics=True,
        )
        
        return training_args
    
    def train_model(self, train_dataset, val_dataset, use_class_weights: bool = True) -> None:
        """Train the SciBERT model"""
        logger.info("Starting SciBERT fine-tuning...")
        
        try:
            # Setup training arguments
            training_args = self.setup_training_arguments()
            
            if use_class_weights:
                # Calculate class weights for imbalanced data
                train_labels = [train_dataset[i]['labels'].item() for i in range(len(train_dataset))]
                class_weights = compute_class_weight(
                    'balanced', 
                    classes=np.unique(train_labels), 
                    y=train_labels
                )
                class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
                logger.info(f"🎯 Class weights: {class_weights.cpu().numpy()}")
                
                # Try weighted trainer first
                try:
                    self.trainer = WeightedTrainer(
                        class_weights=class_weights,
                        model=self.model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        compute_metrics=self.compute_metrics,
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
                        data_collator=DataCollatorWithPadding(self.tokenizer),
                    )
                    logger.info("✅ Using WeightedTrainer with class balancing")
                except Exception as weighted_error:
                    logger.warning(f"WeightedTrainer failed: {weighted_error}")
                    use_class_weights = False
            
            if not use_class_weights:
                logger.info("🔄 Using standard Trainer without class weights")
                self.trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=self.compute_metrics,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
                    data_collator=DataCollatorWithPadding(self.tokenizer),
                )
            
            # Train the model
            logger.info("🚀 Training started...")
            train_result = self.trainer.train()
            
            # Store training results
            self.results['training_results'] = train_result.metrics
            logger.info("✅ SciBERT training completed successfully!")
            logger.info(f"📊 Final training loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
            
            # Get training history
            if hasattr(self.trainer, 'state') and self.trainer.state.log_history:
                self.results['training_history'] = self.trainer.state.log_history
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate_model(self) -> Dict:
        """Comprehensive evaluation of the trained SciBERT model"""
        logger.info("📊 Evaluating SciBERT model on test set...")
        
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation")
        
        if self.test_dataset is None:
            raise ValueError("Test dataset not available")
        
        try:
            # Get predictions on test set
            predictions = self.trainer.predict(self.test_dataset)
            y_pred_logits = predictions.predictions
            y_pred_proba = torch.softmax(torch.tensor(y_pred_logits), dim=1).numpy()
            y_pred = np.argmax(y_pred_logits, axis=1)
            y_true = np.array(self.test_labels)
            
            # Calculate comprehensive metrics with proper handling
            accuracy = accuracy_score(y_true, y_pred)
            
            # Handle potential None returns from precision_recall_fscore_support
            try:
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true, y_pred, average='weighted', zero_division=0
                )
                # Ensure no None values
                precision = float(precision) if precision is not None else 0.0
                recall = float(recall) if recall is not None else 0.0
                f1 = float(f1) if f1 is not None else 0.0
                
                # Per-class metrics with safety checks
                precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                    y_true, y_pred, average=None, zero_division=0
                )
                
                # Convert to lists with None handling
                precision_per_class = [float(x) if x is not None else 0.0 for x in precision_per_class]
                recall_per_class = [float(x) if x is not None else 0.0 for x in recall_per_class]
                f1_per_class = [float(x) if x is not None else 0.0 for x in f1_per_class]
                support = support.tolist() if hasattr(support, 'tolist') else [int(support)] if support is not None else [0]
                
            except Exception as metric_error:
                logger.warning(f"Error calculating detailed metrics: {metric_error}")
                precision = recall = f1 = 0.0
                precision_per_class = recall_per_class = f1_per_class = [0.0, 0.0]
                support = [0, 0]
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # AUC metrics with error handling
            try:
                if len(np.unique(y_true)) == 2 and y_pred_proba.shape[1] >= 2:  # Binary classification
                    auc_roc = roc_auc_score(y_true, y_pred_proba[:, 1])
                    auc_pr = average_precision_score(y_true, y_pred_proba[:, 1])
                else:
                    try:
                        auc_roc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                    except:
                        auc_roc = 0.0
                    auc_pr = 0.0
            except Exception as auc_error:
                logger.warning(f"Could not calculate AUC metrics: {auc_error}")
                auc_roc = 0.0
                auc_pr = 0.0
            
            # Additional metrics for binary classification
            if len(np.unique(y_true)) == 2 and cm.size == 4:
                try:
                    tn, fp, fn, tp = cm.ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                except:
                    specificity = sensitivity = 0.0
            else:
                specificity = sensitivity = 0.0
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_roc': float(auc_roc),
                'auc_pr': float(auc_pr),
                'specificity': float(specificity),
                'sensitivity': float(sensitivity),
                'precision_per_class': precision_per_class,
                'recall_per_class': recall_per_class,
                'f1_per_class': f1_per_class,
                'confusion_matrix': cm.tolist(),
                'support': support,
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist(),
                'true_labels': y_true.tolist()
            }
            
            # Store results
            self.results['test_metrics'] = metrics
            
            # Print detailed results
            logger.info("="*60)
            logger.info("📊 SCIBERT TEST SET EVALUATION RESULTS")
            logger.info("="*60)
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            logger.info(f"AUC-ROC: {auc_roc:.4f}")
            logger.info(f"AUC-PR: {auc_pr:.4f}")
            logger.info(f"Specificity: {specificity:.4f}")
            logger.info(f"Sensitivity: {sensitivity:.4f}")
            
            print("\n📋 Classification Report:")
            try:
                print(classification_report(y_true, y_pred, target_names=['Non-Biased', 'Biased'], zero_division=0))
            except Exception as report_error:
                print(f"Could not generate classification report: {report_error}")
                print(f"Predictions summary: {np.bincount(y_pred)}")
                print(f"True labels summary: {np.bincount(y_true)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def create_visualizations(self, save_plots: bool = True) -> None:
        """Create comprehensive visualizations for SciBERT results"""
        logger.info("📈 Creating SciBERT visualizations...")
        
        if 'test_metrics' not in self.results:
            logger.warning("No test results available for visualization")
            return
        
        try:
            # Create analysis plot
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('SciBERT Model Performance Analysis', fontsize=16, fontweight='bold')
            
            metrics = self.results['test_metrics']
            y_true = np.array(metrics['true_labels'])
            y_pred = np.array(metrics['predictions'])
            y_pred_proba = np.array(metrics['probabilities'])
            
            # 1. Confusion Matrix
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                       xticklabels=['Non-Biased', 'Biased'],
                       yticklabels=['Non-Biased', 'Biased'])
            axes[0,0].set_title('Confusion Matrix')
            axes[0,0].set_ylabel('True Label')
            axes[0,0].set_xlabel('Predicted Label')
            
            # 2. Performance Metrics
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']
            metric_values = [
                metrics['accuracy'], metrics['precision'], metrics['recall'],
                metrics['f1_score'], metrics['auc_roc'], metrics['auc_pr']
            ]
            
            bars = axes[0,1].bar(metric_names, metric_values, 
                                color=plt.cm.plasma(np.linspace(0, 1, len(metric_names))))
            axes[0,1].set_title('Performance Metrics')
            axes[0,1].set_ylabel('Score')
            axes[0,1].set_ylim(0, 1)
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            # 3. ROC Curve (if binary classification)
            if len(np.unique(y_true)) == 2 and metrics['auc_roc'] > 0:
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                    axes[1,0].plot(fpr, tpr, label=f'SciBERT ROC (AUC = {metrics["auc_roc"]:.3f})', 
                                  linewidth=2, color='darkblue')
                    axes[1,0].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
                    axes[1,0].set_xlabel('False Positive Rate')
                    axes[1,0].set_ylabel('True Positive Rate')
                    axes[1,0].set_title('ROC Curve')
                    axes[1,0].legend()
                    axes[1,0].grid(True, alpha=0.3)
                except:
                    axes[1,0].text(0.5, 0.5, 'ROC Curve\nNot Available', 
                                  ha='center', va='center', transform=axes[1,0].transAxes)
            else:
                axes[1,0].text(0.5, 0.5, 'ROC Curve\nNot Available', 
                              ha='center', va='center', transform=axes[1,0].transAxes)
            
            # 4. Training History (if available)
            if 'training_history' in self.results:
                history = self.results['training_history']
                eval_losses = []
                epochs = []
                
                for entry in history:
                    if 'eval_loss' in entry and 'epoch' in entry:
                        eval_losses.append(entry['eval_loss'])
                        epochs.append(entry['epoch'])
                
                if eval_losses:
                    axes[1,1].plot(epochs, eval_losses, 'g-', linewidth=2, marker='o')
                    axes[1,1].set_xlabel('Epoch')
                    axes[1,1].set_ylabel('Validation Loss')
                    axes[1,1].set_title('SciBERT Training Progress')
                    axes[1,1].grid(True, alpha=0.3)
                else:
                    axes[1,1].text(0.5, 0.5, 'Training History\nNot Available', 
                                  ha='center', va='center', transform=axes[1,1].transAxes)
            else:
                axes[1,1].text(0.5, 0.5, 'Training History\nNot Available', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('scibert_performance_analysis.png', dpi=300, bbox_inches='tight')
                logger.info("📈 Visualization saved as 'scibert_performance_analysis.png'")
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def save_model_and_results(self, save_dir: str = 'scibert_outputs') -> Tuple[str, str]:
        """Save trained SciBERT model and results"""
        try:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model and tokenizer
            model_path = os.path.join(save_dir, f'scibert_model_{timestamp}')
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            logger.info(f"💾 SciBERT model and tokenizer saved to {model_path}")
            
            # Save results
            results_path = os.path.join(save_dir, f'scibert_results_{timestamp}.json')
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"💾 Results saved to {results_path}")
            
            # Save report
            self._generate_report(save_dir, timestamp)
            
            return model_path, results_path
            
        except Exception as e:
            logger.error(f"Error saving model and results: {e}")
            raise
    
    def _generate_report(self, save_dir: str, timestamp: str) -> None:
        """Generate detailed text report"""
        try:
            report_path = os.path.join(save_dir, f'scibert_report_{timestamp}.txt')
            with open(report_path, 'w') as f:
                f.write("SCIBERT FINE-TUNING REPORT\n")
                f.write("="*60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Device: {self.device}\n\n")
                
                # Test results
                if 'test_metrics' in self.results:
                    test_metrics = self.results['test_metrics']
                    f.write("TEST SET RESULTS:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
                    f.write(f"Precision: {test_metrics['precision']:.4f}\n")
                    f.write(f"Recall: {test_metrics['recall']:.4f}\n")
                    f.write(f"F1-Score: {test_metrics['f1_score']:.4f}\n")
                    f.write(f"AUC-ROC: {test_metrics['auc_roc']:.4f}\n")
                    f.write(f"AUC-PR: {test_metrics['auc_pr']:.4f}\n")
                    f.write(f"Specificity: {test_metrics['specificity']:.4f}\n")
                    f.write(f"Sensitivity: {test_metrics['sensitivity']:.4f}\n")
            
            logger.info(f"💾 Report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")

    def predict_single_text(self, text: str) -> Dict:
        """Predict bias for a single text using SciBERT"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be trained before prediction")
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        return {
            'prediction': prediction.item(),
            'probabilities': probabilities.cpu().numpy().tolist()[0],
            'prediction_label': 'Biased' if prediction.item() == 1 else 'Non-Biased'
        }


def load_trained_scibert_model(model_path: str) -> SciBERTTrainer:
    """Load a previously trained SciBERT model"""
    trainer = SciBERTTrainer()
    
    try:
        # Load tokenizer and model
        trainer.tokenizer = AutoTokenizer.from_pretrained(model_path)
        trainer.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        trainer.model.to(trainer.device)
        
        logger.info(f"✅ SciBERT model loaded from {model_path}")
        return trainer
        
    except Exception as e:
        logger.error(f"Failed to load SciBERT model: {e}")
        raise


def batch_predict_scibert(trainer: SciBERTTrainer, texts: List[str]) -> List[Dict]:
    """Predict bias for multiple texts using SciBERT"""
    if trainer.model is None or trainer.tokenizer is None:
        raise ValueError("Model must be trained or loaded before prediction")
    
    predictions = []
    trainer.model.eval()
    
    with torch.no_grad():
        for text in tqdm(texts, desc="SciBERT Predicting"):
            try:
                result = trainer.predict_single_text(text)
                predictions.append(result)
            except Exception as e:
                logger.warning(f"Failed to predict for text: {str(text)[:50]}... Error: {e}")
                predictions.append({
                    'prediction': -1,
                    'probabilities': [0.0, 0.0],
                    'prediction_label': 'Error'
                })
    
    return predictions


def compare_models(texts: List[str], scibert_predictions: List[Dict], 
                  biobert_predictions: List[Dict] = None) -> pd.DataFrame:
    """Compare SciBERT with BioBERT predictions if available"""
    results_df = pd.DataFrame({
        'text': texts,
        'scibert_prediction': [p['prediction'] for p in scibert_predictions],
        'scibert_label': [p['prediction_label'] for p in scibert_predictions],
        'scibert_confidence': [max(p['probabilities']) for p in scibert_predictions],
        'scibert_bias_prob': [p['probabilities'][1] if len(p['probabilities']) > 1 else 0.0 for p in scibert_predictions]
    })
    
    if biobert_predictions:
        results_df['biobert_prediction'] = [p['prediction'] for p in biobert_predictions]
        results_df['biobert_label'] = [p['prediction_label'] for p in biobert_predictions]
        results_df['biobert_confidence'] = [max(p['probabilities']) for p in biobert_predictions]
        results_df['agreement'] = results_df['scibert_prediction'] == results_df['biobert_prediction']
        
        print("🔬 Model Comparison Summary:")
        print(f"Agreement rate: {results_df['agreement'].mean():.3f}")
        print(f"SciBERT avg confidence: {results_df['scibert_confidence'].mean():.3f}")
        print(f"BioBERT avg confidence: {results_df['biobert_confidence'].mean():.3f}")
    
    print("🔬 SciBERT Prediction Summary:")
    print(f"Total texts: {len(results_df)}")
    print(f"Predicted as biased: {sum(results_df['scibert_prediction'] == 1)}")
    print(f"Predicted as non-biased: {sum(results_df['scibert_prediction'] == 0)}")
    print(f"Average confidence: {results_df['scibert_confidence'].mean():.4f}")
    
    return results_df


def main():
    """Main function to train SciBERT model"""
    print("="*70)
    print("🔬 SCIBERT TRAINER FOR BIAS DETECTION")
    print("="*70)
    print("🧬 Scientific literature pre-trained model")
    print("🔧 CPU-only for maximum compatibility")
    print("📊 Optimized for medical abstracts")
    print("="*70)
    
    # Configuration - UPDATE THIS PATH
    DATA_PATH = "/Users/satya/SyntheticBias/cleaned_abstracts_aggressive.csv"
    
    # Available SciBERT models (try in order of preference)
    SCIBERT_MODELS = [
        "allenai/scibert_scivocab_uncased",     # Updated model name
        "allenai/scibert_scivocab_cased",       # Cased version
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",  # Alternative biomedical
        "dmis-lab/biobert-base-cased-v1.2",     # Fallback to BioBERT
    ]
    
    MAX_LENGTH = 256
    
    print(f"📁 Data file: {DATA_PATH}")
    print(f"📏 Max sequence length: {MAX_LENGTH}")
    print("⚖️  Class balancing: Weighted loss function")
    print("🔧 FORCED CPU usage for stability")
    print("="*70)
    
    # Check if file exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: File {DATA_PATH} not found!")
        print("Please update the DATA_PATH variable with the correct path to your CSV file.")
        return None, None
    
    # Try different SciBERT models until one works
    trainer = None
    for model_name in SCIBERT_MODELS:
        print(f"🔬 Trying model: {model_name}")
        try:
            trainer = SciBERTTrainer(model_name=model_name, random_state=42)
            print(f"✅ Successfully initialized with {model_name}")
            break
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {str(e)[:100]}...")
            continue
    
    if trainer is None:
        print("❌ ERROR: Could not load any SciBERT models!")
        print("Please check your internet connection or try the BioBERT trainer instead.")
        return None, None
    
    try:
        # Load and prepare data
        print("📊 Loading and preparing data...")
        texts, labels = trainer.load_data(DATA_PATH)
        
        # Split data
        print("🔀 Splitting data...")
        (train_texts, val_texts, test_texts, 
         train_labels, val_labels, test_labels) = trainer.split_data(texts, labels)
        
        # Initialize model and tokenizer
        print(f"🔬 Initializing {trainer.model_name}...")
        trainer.initialize_model_and_tokenizer(num_labels=2)
        
        # Create datasets
        print("📦 Creating datasets...")
        train_dataset, val_dataset, test_dataset = trainer.create_datasets(
            train_texts, val_texts, test_texts,
            train_labels, val_labels, test_labels,
            max_length=MAX_LENGTH
        )
        
        # Train model
        print(f"🚀 Starting {trainer.model_name} fine-tuning...")
        trainer.train_model(train_dataset, val_dataset, use_class_weights=True)
        
        # Evaluate model
        print(f"📊 Evaluating {trainer.model_name} on test set...")
        metrics = trainer.evaluate_model()
        
        # Create visualizations
        print("📈 Creating visualizations...")
        trainer.create_visualizations(save_plots=True)
        
        # Save model and results
        print("💾 Saving model and results...")
        model_path, results_path = trainer.save_model_and_results()
        
        # Determine model type for display
        model_type = "SciBERT" if "scibert" in trainer.model_name.lower() else "Alternative Scientific Model"
        
        # Print final summary
        print("\n" + "="*70)
        print(f"🎉 {model_type.upper()} TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📊 Final {model_type} Results:")
        if 'test_metrics' in trainer.results:
            test_metrics = trainer.results['test_metrics']
            print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"   Precision: {test_metrics['precision']:.4f}")
            print(f"   Recall: {test_metrics['recall']:.4f}")
            print(f"   F1-Score: {test_metrics['f1_score']:.4f}")
            print(f"   AUC-ROC: {test_metrics['auc_roc']:.4f}")
            print(f"   Specificity: {test_metrics['specificity']:.4f}")
            print(f"   Sensitivity: {test_metrics['sensitivity']:.4f}")
        
        print(f"\n💾 {model_type} outputs saved:")
        print(f"   Model: {model_path}")
        print(f"   Results: {results_path}")
        print(f"   Visualization: scibert_performance_analysis.png")
        
        # Demo prediction
        print(f"\n🔍 Testing {model_type} with sample predictions...")
        sample_texts = [
            "This study demonstrates significant improvements in treatment outcomes with the new therapy.",
            "The results show no statistically significant difference between groups.",
            "Our findings reveal promising therapeutic potential for this novel intervention.",
            "The data indicates potential bias in participant selection methodology."
        ]
        
        for i, sample_text in enumerate(sample_texts, 1):
            try:
                prediction = trainer.predict_single_text(sample_text)
                print(f"\nSample {i}: '{sample_text[:60]}...'")
                print(f"  Prediction: {prediction['prediction_label']}")
                print(f"  Confidence: {max(prediction['probabilities']):.4f}")
                print(f"  Bias probability: {prediction['probabilities'][1]:.4f}")
            except Exception as e:
                print(f"Sample {i} prediction failed: {e}")
        
        # Model-specific notes
        if "scibert" in trainer.model_name.lower():
            print(f"\n📊 SciBERT Performance Notes:")
            print("  ✅ Pre-trained on scientific literature")
            print("  ✅ Good for research methodology terminology")
            print("  ✅ Excellent generalization to scientific domains")
        elif "pubmed" in trainer.model_name.lower():
            print(f"\n📊 PubMedBERT Performance Notes:")
            print("  ✅ Pre-trained on PubMed abstracts")
            print("  ✅ Optimized for biomedical literature")
            print("  ✅ Strong performance on medical abstracts")
        else:
            print(f"\n📊 Model Performance Notes:")
            print("  ✅ Scientific domain optimization")
            print("  ✅ Good performance on abstract classification")
        
        print(f"\n✅ {model_type} training pipeline completed successfully!")
        return trainer, metrics
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Check internet connection for model download")
        print("2. Ensure you have enough RAM (requires ~2-4GB)")
        print("3. Verify CSV has 'abstract' and 'bias_label' columns")
        print("4. Try the BioBERT trainer as an alternative")
        return None, None


def run_both_models_comparison():
    """Run both SciBERT and BioBERT for comparison"""
    print("="*70)
    print("🔬 DUAL MODEL COMPARISON: SCIBERT vs BIOBERT")
    print("="*70)
    
    # This function can be used to compare both models
    # You would need to modify this to load both models and compare results
    
    print("To compare both models:")
    print("1. Run SciBERT training first: python scibert_trainer.py")
    print("2. Run BioBERT training: python biobert_trainer_fixed.py")
    print("3. Load both trained models and compare predictions")
    
    return None


if __name__ == "__main__":
    # Run the main SciBERT training pipeline
    trainer, metrics = main()
    
    # Optional: Additional analysis or predictions can be added here
    if trainer is not None:
        print("\n🔧 SciBERT model is ready for additional predictions!")
        print("Use trainer.predict_single_text(text) for single predictions")
        print("Use batch_predict_scibert(trainer, texts) for multiple predictions")
        print("Use compare_models(texts, scibert_preds, biobert_preds) for model comparison")