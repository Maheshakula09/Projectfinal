# -*- coding: utf-8 -*-
"""
Medical Abstract Text Cleaning System
Comprehensive text preprocessing for medical research abstracts
"""

import pandas as pd
import numpy as np
import re
import string
import warnings
from typing import List, Dict, Optional, Tuple
import logging

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import unicodedata

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class MedicalAbstractCleaner:
    """
    Comprehensive text cleaning system for medical research abstracts
    """
    
    def __init__(self, preserve_medical_terms: bool = True, preserve_numbers: bool = True):
        """
        Initialize the abstract cleaner
        
        Args:
            preserve_medical_terms: Keep important medical terminology
            preserve_numbers: Keep numerical values and statistical measures
        """
        self.preserve_medical_terms = preserve_medical_terms
        self.preserve_numbers = preserve_numbers
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize NLTK tools
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Load English stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
        # Medical-specific stopwords to remove
        self.medical_stopwords = {
            'study', 'studies', 'research', 'patient', 'patients', 'group', 'groups',
            'method', 'methods', 'result', 'results', 'conclusion', 'conclusions',
            'background', 'objective', 'objectives', 'purpose', 'aim', 'aims',
            'design', 'setting', 'participants', 'intervention', 'interventions',
            'outcome', 'outcomes', 'measure', 'measures', 'analysis', 'analyses'
        }
        
        # Important medical terms to preserve (even if they might be considered stopwords)
        self.preserve_terms = {
            'cancer', 'tumor', 'malignant', 'benign', 'diagnosis', 'treatment',
            'therapy', 'surgery', 'medication', 'drug', 'clinical', 'trial',
            'randomized', 'controlled', 'double-blind', 'placebo', 'efficacy',
            'safety', 'adverse', 'effect', 'side', 'dose', 'dosage', 'mg', 'ml',
            'significant', 'statistical', 'confidence', 'interval', 'p-value',
            'odds', 'ratio', 'risk', 'hazard', 'survival', 'mortality', 'morbidity'
        }
        
        # Statistical and numerical patterns to preserve
        self.numerical_patterns = [
            r'\d+\.?\d*\s*%',  # Percentages
            r'p\s*[<>=]\s*\d+\.?\d*',  # P-values
            r'\d+\.?\d*\s*mg',  # Dosages
            r'\d+\.?\d*\s*ml',  # Volumes
            r'\d+\.?\d*\s*years?',  # Ages/time
            r'\d+\.?\d*\s*months?',  # Time periods
            r'\d+\.?\d*\s*days?',  # Time periods
            r'n\s*=\s*\d+',  # Sample sizes
            r'\d+\.?\d*\s*±\s*\d+\.?\d*',  # Mean ± SD
            r'\d+\.?\d*\s*\(\d+\.?\d*-\d+\.?\d*\)',  # Confidence intervals
            r'OR\s*=?\s*\d+\.?\d*',  # Odds ratios
            r'HR\s*=?\s*\d+\.?\d*',  # Hazard ratios
            r'RR\s*=?\s*\d+\.?\d*'   # Relative risk
        ]
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        required_data = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
        
        for data_name in required_data:
            try:
                if data_name in ['punkt', 'punkt_tab']:
                    nltk.data.find(f'tokenizers/{data_name}')
                elif data_name == 'stopwords':
                    nltk.data.find(f'corpora/{data_name}')
                elif data_name == 'wordnet':
                    nltk.data.find(f'corpora/{data_name}')
            except LookupError:
                logger.info(f"Downloading NLTK data: {data_name}")
                nltk.download(data_name, quiet=True)
    
    def preserve_special_terms(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Temporarily replace special terms and patterns with placeholders
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (modified_text, replacement_dict)
        """
        replacements = {}
        modified_text = text
        placeholder_counter = 0
        
        # Preserve numerical patterns if requested
        if self.preserve_numbers:
            for pattern in self.numerical_patterns:
                matches = re.finditer(pattern, modified_text, re.IGNORECASE)
                for match in matches:
                    placeholder = f"NUMERICPATTERN{placeholder_counter}"
                    replacements[placeholder] = match.group()
                    modified_text = modified_text.replace(match.group(), placeholder, 1)
                    placeholder_counter += 1
        
        # Preserve important medical terms
        if self.preserve_medical_terms:
            for term in self.preserve_terms:
                # Use word boundaries to match whole words only
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = re.finditer(pattern, modified_text, re.IGNORECASE)
                for match in matches:
                    placeholder = f"MEDTERM{placeholder_counter}"
                    replacements[placeholder] = match.group()
                    modified_text = modified_text.replace(match.group(), placeholder, 1)
                    placeholder_counter += 1
        
        return modified_text, replacements
    
    def restore_special_terms(self, text: str, replacements: Dict[str, str]) -> str:
        """
        Restore previously preserved terms
        
        Args:
            text: Text with placeholders
            replacements: Dictionary mapping placeholders to original terms
            
        Returns:
            Text with restored terms
        """
        restored_text = text
        for placeholder, original in replacements.items():
            restored_text = restored_text.replace(placeholder, original)
        return restored_text
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Normalize Unicode to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Replace common problematic characters
        replacements = {
            ''': "'", ''': "'", '"': '"', '"': '"',
            '–': '-', '—': '-', '…': '...',
            'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
            '≤': '<=', '≥': '>=', '±': '+/-'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def remove_structured_labels(self, text: str) -> str:
        """
        Remove common abstract section labels
        
        Args:
            text: Input abstract text
            
        Returns:
            Text without section labels
        """
        # Common abstract section patterns
        section_patterns = [
            r'^(background|objective|objectives|purpose|aims?):?\s*',
            r'^(methods?|methodology|design|study design):?\s*',
            r'^(results?|findings?|outcomes?):?\s*',
            r'^(conclusion|conclusions?|interpretation):?\s*',
            r'^(discussion|implications?):?\s*',
            r'^(introduction|summary):?\s*',
            r'^(materials? and methods?):?\s*',
            r'^(patients? and methods?):?\s*'
        ]
        
        cleaned_text = text
        for pattern in section_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        
        return cleaned_text.strip()
    
    def clean_html_xml(self, text: str) -> str:
        """
        Remove HTML/XML tags and entities
        
        Args:
            text: Input text
            
        Returns:
            Text without HTML/XML
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove XML tags
        text = re.sub(r'</?[^>]+/?>', '', text)
        
        # Replace common HTML entities
        html_entities = {
            '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
            '&quot;': '"', '&#39;': "'", '&apos;': "'",
            '&mdash;': '-', '&ndash;': '-', '&hellip;': '...'
        }
        
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace and remove extra spaces
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.;:!?])([^\s])', r'\1 \2', text)  # Add space after punctuation
        
        return text
    
    def clean_punctuation(self, text: str, remove_all: bool = False) -> str:
        """
        Clean and normalize punctuation
        
        Args:
            text: Input text
            remove_all: If True, remove all punctuation; if False, clean selectively
            
        Returns:
            Text with cleaned punctuation
        """
        if remove_all:
            # Remove all punctuation except those in preserved terms
            text = text.translate(str.maketrans('', '', string.punctuation))
        else:
            # Clean specific punctuation issues
            # Remove excessive punctuation
            text = re.sub(r'[.]{2,}', '.', text)  # Multiple periods
            text = re.sub(r'[!]{2,}', '!', text)  # Multiple exclamations
            text = re.sub(r'[?]{2,}', '?', text)  # Multiple questions
            
            # Remove brackets and parentheses content that's not important
            # (but preserve statistical information)
            if not self.preserve_numbers:
                text = re.sub(r'\([^)]*\)', '', text)
                text = re.sub(r'\[[^\]]*\]', '', text)
        
        return text
    
    def tokenize_and_clean_words(self, text: str, 
                                remove_stopwords: bool = True,
                                min_word_length: int = 2,
                                lemmatize: bool = True) -> List[str]:
        """
        Tokenize text and clean individual words
        
        Args:
            text: Input text
            remove_stopwords: Remove English stopwords
            min_word_length: Minimum word length to keep
            lemmatize: Apply lemmatization
            
        Returns:
            List of cleaned tokens
        """
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        cleaned_tokens = []
        for token in tokens:
            # Skip if too short
            if len(token) < min_word_length:
                continue
            
            # Skip if it's only punctuation
            if token in string.punctuation:
                continue
            
            # Skip if it's a number (unless preserving numbers)
            if token.isdigit() and not self.preserve_numbers:
                continue
            
            # Skip stopwords (but preserve medical terms)
            if remove_stopwords:
                if token in self.stop_words and token not in self.preserve_terms:
                    continue
                if token in self.medical_stopwords and token not in self.preserve_terms:
                    continue
            
            # Apply lemmatization
            if lemmatize:
                try:
                    token = self.lemmatizer.lemmatize(token)
                except:
                    pass  # Keep original token if lemmatization fails
            
            cleaned_tokens.append(token)
        
        return cleaned_tokens
    
    def clean_single_abstract(self, abstract: str, 
                            cleaning_level: str = 'moderate') -> str:
        """
        Clean a single abstract with specified cleaning level
        
        Args:
            abstract: Raw abstract text
            cleaning_level: 'light', 'moderate', or 'aggressive'
            
        Returns:
            Cleaned abstract text
        """
        if not isinstance(abstract, str) or not abstract.strip():
            return ""
        
        # Step 1: Preserve special terms
        text, replacements = self.preserve_special_terms(abstract)
        
        # Step 2: Basic cleaning (all levels)
        text = self.normalize_unicode(text)
        text = self.clean_html_xml(text)
        text = self.remove_structured_labels(text)
        
        if cleaning_level in ['moderate', 'aggressive']:
            # Step 3: Moderate cleaning
            text = self.clean_punctuation(text, remove_all=False)
            text = self.normalize_whitespace(text)
        
        if cleaning_level == 'aggressive':
            # Step 4: Aggressive cleaning
            tokens = self.tokenize_and_clean_words(
                text, 
                remove_stopwords=True,
                min_word_length=3,
                lemmatize=True
            )
            text = ' '.join(tokens)
        
        # Step 5: Restore special terms
        text = self.restore_special_terms(text, replacements)
        
        # Step 6: Final cleanup
        text = self.normalize_whitespace(text)
        
        return text.strip()
    
    def clean_dataset(self, df: pd.DataFrame, 
                     text_column: str = 'abstract',
                     cleaning_level: str = 'moderate',
                     create_backup: bool = True) -> pd.DataFrame:
        """
        Clean all abstracts in a dataset
        
        Args:
            df: DataFrame containing abstracts
            text_column: Name of column containing abstract text
            cleaning_level: 'light', 'moderate', or 'aggressive'
            create_backup: Create backup of original text
            
        Returns:
            DataFrame with cleaned abstracts
        """
        logger.info(f"Starting abstract cleaning with '{cleaning_level}' level...")
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Create backup column if requested
        if create_backup:
            cleaned_df[f'{text_column}_original'] = cleaned_df[text_column].copy()
        
        # Clean abstracts
        total_abstracts = len(cleaned_df)
        cleaned_abstracts = []
        
        for idx, abstract in enumerate(cleaned_df[text_column]):
            if idx % 100 == 0:
                logger.info(f"Cleaning progress: {idx}/{total_abstracts} ({idx/total_abstracts*100:.1f}%)")
            
            cleaned_abstract = self.clean_single_abstract(abstract, cleaning_level)
            cleaned_abstracts.append(cleaned_abstract)
        
        cleaned_df[text_column] = cleaned_abstracts
        
        # Remove empty abstracts
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df[cleaned_df[text_column].str.strip() != '']
        final_count = len(cleaned_df)
        
        logger.info(f"Cleaning complete!")
        logger.info(f"Abstracts processed: {total_abstracts}")
        logger.info(f"Abstracts retained: {final_count}")
        logger.info(f"Empty abstracts removed: {initial_count - final_count}")
        
        return cleaned_df.reset_index(drop=True)
    
    def analyze_cleaning_impact(self, original_df: pd.DataFrame, 
                              cleaned_df: pd.DataFrame,
                              text_column: str = 'abstract') -> Dict:
        """
        Analyze the impact of cleaning on the dataset
        
        Args:
            original_df: Original DataFrame
            cleaned_df: Cleaned DataFrame
            text_column: Name of text column
            
        Returns:
            Dictionary with cleaning statistics
        """
        original_texts = original_df[text_column].tolist()
        cleaned_texts = cleaned_df[text_column].tolist()
        
        stats = {
            'original_count': len(original_texts),
            'cleaned_count': len(cleaned_texts),
            'removed_count': len(original_texts) - len(cleaned_texts),
            'original_avg_length': np.mean([len(text) for text in original_texts]),
            'cleaned_avg_length': np.mean([len(text) for text in cleaned_texts]),
            'original_avg_words': np.mean([len(text.split()) for text in original_texts]),
            'cleaned_avg_words': np.mean([len(text.split()) for text in cleaned_texts])
        }
        
        # Calculate length reduction
        stats['length_reduction_pct'] = (1 - stats['cleaned_avg_length'] / stats['original_avg_length']) * 100
        stats['words_reduction_pct'] = (1 - stats['cleaned_avg_words'] / stats['original_avg_words']) * 100
        
        return stats


def clean_medical_abstracts(input_file: str, 
                          output_file: str = None,
                          text_column: str = 'abstract',
                          cleaning_level: str = 'moderate',
                          preserve_medical_terms: bool = True,
                          preserve_numbers: bool = True) -> pd.DataFrame:
    """
    Main function to clean medical abstracts from a CSV file
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional)
        text_column: Name of column containing abstracts
        cleaning_level: 'light', 'moderate', or 'aggressive'
        preserve_medical_terms: Keep important medical terminology
        preserve_numbers: Keep numerical values
        
    Returns:
        Cleaned DataFrame
    """
    
    # Load data
    logger.info(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        return None
    
    logger.info(f"Loaded {len(df)} abstracts")
    
    # Initialize cleaner
    cleaner = MedicalAbstractCleaner(
        preserve_medical_terms=preserve_medical_terms,
        preserve_numbers=preserve_numbers
    )
    
    # Clean dataset
    cleaned_df = cleaner.clean_dataset(df, text_column, cleaning_level)
    
    # Analyze cleaning impact
    stats = cleaner.analyze_cleaning_impact(df, cleaned_df, text_column)
    
    # Print statistics
    print("\n" + "="*60)
    print("📊 CLEANING STATISTICS")
    print("="*60)
    print(f"Original abstracts: {stats['original_count']}")
    print(f"Cleaned abstracts: {stats['cleaned_count']}")
    print(f"Removed abstracts: {stats['removed_count']}")
    print(f"Average length: {stats['original_avg_length']:.0f} → {stats['cleaned_avg_length']:.0f} chars ({stats['length_reduction_pct']:.1f}% reduction)")
    print(f"Average words: {stats['original_avg_words']:.0f} → {stats['cleaned_avg_words']:.0f} words ({stats['words_reduction_pct']:.1f}% reduction)")
    print("="*60)
    
    # Save cleaned data
    if output_file:
        cleaned_df.to_csv(output_file, index=False)
        logger.info(f"Cleaned data saved to {output_file}")
    
    return cleaned_df


# Example usage and testing
def main():
    """Example usage of the medical abstract cleaner"""
    
    print("🏥 MEDICAL ABSTRACT CLEANING SYSTEM")
    print("="*50)
    
    # Example: Clean a dataset
    input_file = "balanced_medical_abstracts_final_20250721_180109.csv"  # Your dataset file
    output_file = "cleaned_medical_abstracts.csv"
    
    print("📋 CLEANING OPTIONS:")
    print("  • light: Basic cleaning (HTML, Unicode, whitespace)")
    print("  • moderate: + punctuation, section labels (RECOMMENDED)")
    print("  • aggressive: + stopwords, lemmatization, tokenization")
    print()
    
    # Clean with different levels
    cleaning_levels = ['light', 'moderate', 'aggressive']
    
    for level in cleaning_levels:
        print(f"🧹 Testing {level} cleaning...")
        output_name = f"cleaned_abstracts_{level}.csv"
        
        try:
            cleaned_df = clean_medical_abstracts(
                input_file=input_file,
                output_file=output_name,
                cleaning_level=level,
                preserve_medical_terms=True,
                preserve_numbers=True
            )
            
            if cleaned_df is not None:
                print(f"✅ {level} cleaning completed: {len(cleaned_df)} abstracts")
            else:
                print(f"❌ {level} cleaning failed")
                
        except Exception as e:
            print(f"❌ Error with {level} cleaning: {e}")
        
        print("-" * 30)
    
    print("\n🎯 RECOMMENDED: Use 'moderate' cleaning level")
    print("📁 Check the cleaned CSV files in your directory")


if __name__ == "__main__":
    print("📦 Required packages: pandas, numpy, nltk")
    print("💡 Usage: Modify input_file path in main() function")
    print("🚀 Run main() to start cleaning")
    print()
    
    # Uncomment to run example
    main()