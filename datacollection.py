# -*- coding: utf-8 -*-
"""
Balanced Bias Detection System - Ensures mix of biased and non-biased abstracts
"""

import pandas as pd
import numpy as np
import re
import xml.etree.ElementTree as ET
import time
import warnings
import logging
import json
import os
import ssl
import urllib.request
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from Bio import Entrez
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
from sklearn.model_selection import train_test_split

# SSL Certificate fix
def setup_ssl_fix():
    """Setup SSL fix for PubMed API"""
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context)))
        
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except ImportError:
            pass
    except Exception as e:
        logging.warning(f"Could not setup SSL fix: {e}")

setup_ssl_fix()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class BalancedBiasDetector:
    def __init__(self, email: str, api_key: str):
        """
        Initialize bias detector with balanced detection patterns
        """
        Entrez.email = email
        Entrez.api_key = api_key
        
        self._download_nltk_data()
        
        # Load sentiment analyzer
        try:
            logger.info("Loading sentiment analyzer...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
        except Exception as e:
            logger.warning(f"Could not load sentiment analyzer: {e}")
            self.sentiment_analyzer = None
        
        # Enhanced search queries targeting different bias types
        self.search_queries = {
            'high_impact_studies': [
                "breakthrough AND (cancer OR cardiovascular) AND 2020:2025[Date - Publication]",
                "revolutionary AND treatment AND 2020:2025[Date - Publication]", 
                "unprecedented AND therapy AND 2020:2025[Date - Publication]"
            ],
            'clinical_trials': [
                "clinical trial[Publication Type] AND (significant OR effective) AND 2020:2025[Date - Publication]",
                "randomized controlled trial[Publication Type] AND outcomes AND 2020:2025[Date - Publication]",
                "phase 3[Title] AND results AND 2020:2025[Date - Publication]"
            ],
            'statistical_studies': [
                "(p < 0.05 OR p = 0.04) AND biomedical AND 2020:2025[Date - Publication]",
                "statistical significance AND medical AND 2020:2025[Date - Publication]",
                "post hoc analysis AND clinical AND 2020:2025[Date - Publication]"
            ],
            'general_medical': [
                "biomedical research[MeSH Terms] AND 2020:2025[Date - Publication]",
                "medical treatment AND outcomes AND 2020:2025[Date - Publication]"
            ]
        }
        
        # Comprehensive bias patterns with lower thresholds
        self.bias_patterns = {
            'hype_language': {
                'extreme_hype': [
                    'breakthrough', 'revolutionary', 'groundbreaking', 'unprecedented',
                    'miraculous', 'game-changing', 'transformative', 'paradigm-shifting',
                    'extraordinary', 'phenomenal', 'spectacular', 'stunning',
                    'amazing', 'incredible', 'remarkable breakthrough', 'historic'
                ],
                'moderate_hype': [
                    'novel', 'innovative', 'superior', 'exceptional', 'remarkable',
                    'outstanding', 'dramatic', 'substantial', 'impressive',
                    'promising', 'encouraging', 'exciting', 'significant advance',
                    'major improvement', 'highly effective', 'optimal'
                ],
                'superlatives': [
                    'best', 'most effective', 'highest', 'lowest', 'fastest',
                    'safest', 'strongest', 'perfect', 'ultimate', 'ideal',
                    'maximum', 'minimum', 'greatest', 'largest', 'smallest'
                ],
                'certainty_language': [
                    'definitely', 'certainly', 'undoubtedly', 'clearly shows',
                    'proves that', 'demonstrates conclusively', 'establishes that',
                    'confirms that', 'eliminates doubt', 'without question'
                ]
            },
            'statistical_bias': {
                'p_hacking_patterns': [
                    r'p\s*=\s*0\.04[5-9]',  # p = 0.045-0.049
                    r'p\s*=\s*0\.050',      # exactly 0.050
                    r'p\s*=\s*0\.051',      # just over significance
                    r'marginally\s+significant',
                    r'borderline\s+significant',
                    r'trending\s+toward\s+significance',
                    r'approached\s+significance',
                    r'nearly\s+significant',
                    r'almost\s+significant',
                    r'p\s*=\s*0\.052',
                    r'p\s*=\s*0\.053'
                ],
                'multiple_testing_issues': [
                    r'multiple\s+comparisons?\s+not\s+corrected',
                    r'no\s+correction\s+for\s+multiple\s+testing',
                    r'post\s+hoc\s+analysis',
                    r'exploratory\s+analysis',
                    r'subgroup\s+analysis.*significant',
                    r'data\s+mining',
                    r'secondary\s+endpoint.*significant'
                ],
                'missing_information': [
                    r'significant\s+improvement(?!.*p\s*[<>=])',
                    r'substantial\s+difference(?!.*p\s*[<>=])',
                    r'marked\s+change(?!.*p\s*[<>=])',
                    r'clear\s+benefit(?!.*p\s*[<>=])',
                    r'obvious\s+effect(?!.*p\s*[<>=])'
                ]
            },
            'selective_reporting': [
                r'primary\s+endpoint.*not\s+met.*but.*secondary',
                r'although.*primary.*endpoint.*failed',
                r'despite.*negative.*primary.*outcome',
                r'secondary\s+analysis.*showed.*significant',
                r'post\s+hoc.*analysis.*revealed',
                r'exploratory\s+analysis.*found',
                r'when\s+we\s+looked\s+at.*subgroup',
                r'interesting.*trend.*observed'
            ],
            'spin_language': [
                r'trend\s+toward\s+improvement',
                r'numerical\s+trend',
                r'clinical\s+relevance.*despite.*statistical',
                r'meaningful\s+difference.*not\s+significant',
                r'important\s+findings.*borderline'
            ]
        }
        
        # Bias scoring weights (adjusted for more sensitivity)
        self.bias_weights = {
            'extreme_hype': 0.8,
            'moderate_hype': 0.4,
            'superlatives': 0.5,
            'certainty_language': 0.6,
            'p_hacking': 0.9,
            'multiple_testing': 0.7,
            'missing_stats': 0.5,
            'selective_reporting': 0.8,
            'spin_language': 0.6
        }
        
        # Lower bias threshold for more sensitivity
        self.bias_threshold = 0.2  # Lowered from 0.3
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        for item in ['punkt', 'stopwords', 'punkt_tab']:
            try:
                if 'punkt' in item:
                    nltk.data.find(f'tokenizers/{item}')
                else:
                    nltk.data.find(f'corpora/{item}')
            except LookupError:
                nltk.download(item, quiet=True)
    
    def fetch_pubmed_ids(self, query: str, max_results: int = 300) -> List[str]:
        """Fetch PubMed IDs with retry logic"""
        for attempt in range(3):
            try:
                if attempt > 0:
                    time.sleep(2)
                
                handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
                record = Entrez.read(handle)
                handle.close()
                
                ids = record.get("IdList", [])
                logger.info(f"Found {len(ids)} articles for query: {query[:60]}...")
                return ids
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:
                    logger.error(f"All attempts failed for query")
                    return []
        return []
    
    def fetch_detailed_abstracts(self, ids: List[str]) -> List[Dict]:
        """Fetch detailed abstracts"""
        abstracts_data = []
        batch_size = 50
        
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            logger.info(f"Fetching batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")
            
            for attempt in range(3):
                try:
                    if attempt > 0:
                        time.sleep(3)
                    
                    handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="medline", retmode="xml")
                    xml_data = handle.read()
                    handle.close()
                    
                    root = ET.fromstring(xml_data)
                    
                    for article in root.findall(".//MedlineCitation"):
                        abstract_data = self._parse_article_xml(article)
                        if abstract_data and len(abstract_data['abstract']) > 200:
                            abstracts_data.append(abstract_data)
                    break
                    
                except Exception as e:
                    logger.warning(f"Batch {i//batch_size + 1}, attempt {attempt + 1} failed: {str(e)}")
                    if attempt == 2:
                        logger.error(f"Failed to fetch batch after 3 attempts")
            
            time.sleep(1)  # Rate limiting
        
        return abstracts_data
    
    def _parse_article_xml(self, article) -> Optional[Dict]:
        """Parse article XML"""
        try:
            pmid_element = article.find(".//PMID")
            pmid = pmid_element.text if pmid_element is not None else None
            
            title_element = article.find(".//ArticleTitle")
            title = title_element.text if title_element is not None else ""
            
            abstract_elements = article.findall(".//AbstractText")
            if not abstract_elements:
                return None
            
            abstract_parts = []
            for abs_elem in abstract_elements:
                text = abs_elem.text if abs_elem.text else ""
                if text:
                    abstract_parts.append(text)
            
            abstract = " ".join(abstract_parts).strip()
            if len(abstract) < 200:
                return None
            
            return {
                'pmid': pmid,
                'title': title,
                'abstract': abstract
            }
            
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
            return None
    
    def enhanced_bias_detection(self, abstract: str, title: str = "") -> Dict:
        """Enhanced bias detection with multiple approaches"""
        text_to_analyze = f"{title} {abstract}".lower()
        sentences = sent_tokenize(abstract)
        
        bias_scores = {}
        detected_patterns = []
        
        # 1. Hype Language Detection
        for category, words in self.bias_patterns['hype_language'].items():
            found_words = [word for word in words if word in text_to_analyze]
            if found_words:
                score = len(found_words) * self.bias_weights.get(category, 0.3)
                bias_scores[category] = score
                detected_patterns.extend([(word, category) for word in found_words])
        
        # 2. Statistical Bias Detection
        for category, patterns in self.bias_patterns['statistical_bias'].items():
            category_score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_to_analyze, re.IGNORECASE)
                if matches:
                    pattern_score = len(matches) * self.bias_weights.get(category.split('_')[0], 0.5)
                    category_score += pattern_score
                    detected_patterns.append((pattern, category, matches))
            
            if category_score > 0:
                bias_scores[category] = category_score
        
        # 3. Selective Reporting Detection
        selective_score = 0
        for pattern in self.bias_patterns['selective_reporting']:
            matches = re.findall(pattern, text_to_analyze, re.IGNORECASE)
            if matches:
                selective_score += len(matches) * self.bias_weights['selective_reporting']
                detected_patterns.append((pattern, 'selective_reporting', matches))
        
        if selective_score > 0:
            bias_scores['selective_reporting'] = selective_score
        
        # 4. Spin Language Detection
        spin_score = 0
        for pattern in self.bias_patterns['spin_language']:
            matches = re.findall(pattern, text_to_analyze, re.IGNORECASE)
            if matches:
                spin_score += len(matches) * self.bias_weights['spin_language']
                detected_patterns.append((pattern, 'spin_language', matches))
        
        if spin_score > 0:
            bias_scores['spin_language'] = spin_score
        
        # 5. Contextual Analysis
        conclusion_sentences = [s for s in sentences if any(word in s.lower() 
                               for word in ['conclusion', 'conclude', 'results show', 'demonstrated'])]
        
        contextual_score = 0
        if conclusion_sentences:
            conclusion_text = " ".join(conclusion_sentences).lower()
            # Check for overconfident conclusions
            overconfident_phrases = ['proves', 'demonstrates conclusively', 'clearly shows', 
                                   'without doubt', 'definitively', 'establishes that']
            for phrase in overconfident_phrases:
                if phrase in conclusion_text:
                    contextual_score += 0.4
                    detected_patterns.append((phrase, 'overconfident_conclusion'))
        
        if contextual_score > 0:
            bias_scores['overconfident_conclusions'] = contextual_score
        
        # 6. Length and Complexity Bias (shorter abstracts with strong claims)
        if len(abstract) < 800 and len(detected_patterns) > 0:
            length_bias = 0.2
            bias_scores['length_complexity'] = length_bias
        
        # Calculate total bias score
        total_score = sum(bias_scores.values()) if bias_scores else 0.0
        
        # Apply different thresholds based on content
        is_biased = total_score > self.bias_threshold
        
        # Additional checks for edge cases
        if not is_biased and len(detected_patterns) >= 3:  # Multiple weak signals
            is_biased = True
            total_score = max(total_score, self.bias_threshold + 0.1)
        
        return {
            'bias_label': 1 if is_biased else 0,
            'confidence_score': min(total_score, 1.0),
            'bias_scores': bias_scores,
            'detected_patterns': detected_patterns[:10],  # Limit for storage
            'bias_types': list(bias_scores.keys())
        }
    
    def collect_balanced_dataset(self, target_size: int = 1500, target_bias_rate: float = 0.3) -> pd.DataFrame:
        """Collect dataset with target bias rate"""
        all_abstracts = []
        collected_count = 0
        bias_count = 0
        
        target_biased = int(target_size * target_bias_rate)
        target_non_biased = target_size - target_biased
        
        logger.info(f"🎯 Target: {target_size} abstracts ({target_biased} biased, {target_non_biased} non-biased)")
        
        # Process each query type
        for query_type, queries in self.search_queries.items():
            logger.info(f"\n=== Processing {query_type.upper()} queries ===")
            
            for query in queries:
                if collected_count >= target_size:
                    break
                
                # Adjust collection based on current bias rate
                current_bias_rate = bias_count / max(collected_count, 1)
                need_more_biased = current_bias_rate < target_bias_rate and bias_count < target_biased
                need_more_non_biased = current_bias_rate > target_bias_rate or bias_count >= target_biased
                
                ids = self.fetch_pubmed_ids(query, 200)
                if not ids:
                    continue
                
                abstracts_batch = self.fetch_detailed_abstracts(ids)
                
                for abstract_data in abstracts_batch:
                    if collected_count >= target_size:
                        break
                    
                    try:
                        # Perform bias detection
                        bias_results = self.enhanced_bias_detection(
                            abstract_data['abstract'], 
                            abstract_data.get('title', '')
                        )
                        
                        abstract_data.update(bias_results)
                        
                        # Selective inclusion based on bias needs
                        is_biased = bias_results['bias_label'] == 1
                        
                        should_include = False
                        if need_more_biased and is_biased:
                            should_include = True
                        elif need_more_non_biased and not is_biased:
                            should_include = True
                        elif collected_count < target_size // 2:  # Early collection phase
                            should_include = True
                        
                        if should_include:
                            all_abstracts.append(abstract_data)
                            collected_count += 1
                            if is_biased:
                                bias_count += 1
                            
                            if collected_count % 100 == 0:
                                current_bias_rate = bias_count / collected_count * 100
                                logger.info(f"✅ Collected: {collected_count} total, {bias_count} biased ({current_bias_rate:.1f}%)")
                    
                    except Exception as e:
                        logger.error(f"Error processing abstract: {e}")
                        continue
        
        # If we still don't have enough biased examples, lower the threshold
        if bias_count < target_biased * 0.5:  # Less than half target biased
            logger.warning(f"Low bias count ({bias_count}), applying secondary detection...")
            self.bias_threshold = 0.1  # Lower threshold
            
            # Re-analyze some non-biased abstracts with lower threshold
            non_biased_abstracts = [a for a in all_abstracts if a['bias_label'] == 0]
            for abstract_data in non_biased_abstracts[:200]:  # Re-analyze first 200
                bias_results = self.enhanced_bias_detection(
                    abstract_data['abstract'], 
                    abstract_data.get('title', '')
                )
                if bias_results['bias_label'] == 1:
                    abstract_data.update(bias_results)
                    bias_count += 1
                    if bias_count >= target_biased:
                        break
        
        df = pd.DataFrame(all_abstracts)
        
        final_bias_count = df['bias_label'].sum()
        final_bias_rate = final_bias_count / len(df) * 100
        
        logger.info(f"\n🎉 Final Dataset: {len(df)} abstracts")
        logger.info(f"📊 Biased: {final_bias_count} ({final_bias_rate:.1f}%)")
        logger.info(f"📊 Non-biased: {len(df) - final_bias_count} ({100-final_bias_rate:.1f}%)")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, base_filename: str = "balanced_medical_abstracts") -> Dict[str, str]:
        """Save the balanced dataset"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files_created = {}
        
        # Create final clean dataset
        final_dataset = df[['abstract', 'bias_label']].copy()
        
        # Main dataset
        main_file = f"{base_filename}_final_{timestamp}.csv"
        final_dataset.to_csv(main_file, index=False)
        files_created['final_dataset'] = main_file
        
        # Create splits ensuring balance in each split
        train_data, temp_data = train_test_split(
            final_dataset, test_size=0.3, random_state=42, 
            stratify=final_dataset['bias_label']
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=42,
            stratify=temp_data['bias_label']
        )
        
        # Save splits
        train_file = f"{base_filename}_train_{timestamp}.csv"
        train_data.to_csv(train_file, index=False)
        files_created['train_split'] = train_file
        
        val_file = f"{base_filename}_validation_{timestamp}.csv"
        val_data.to_csv(val_file, index=False)
        files_created['validation_split'] = val_file
        
        test_file = f"{base_filename}_test_{timestamp}.csv"
        test_data.to_csv(test_file, index=False)
        files_created['test_split'] = test_file
        
        # Detailed analysis dataset (with bias detection details)
        detailed_file = f"{base_filename}_detailed_{timestamp}.csv"
        df.to_csv(detailed_file, index=False)
        files_created['detailed_dataset'] = detailed_file
        
        # Summary
        biased_count = final_dataset['bias_label'].sum()
        non_biased_count = (final_dataset['bias_label'] == 0).sum()
        bias_rate = final_dataset['bias_label'].mean() * 100
        
        summary = f"""=== BALANCED MEDICAL ABSTRACTS BIAS DATASET ===
Total Abstracts: {len(final_dataset)}
Biased (1): {biased_count} ({bias_rate:.1f}%)
Non-Biased (0): {non_biased_count} ({100-bias_rate:.1f}%)
Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BIAS DETECTION IMPROVEMENTS:
- Lower bias threshold: {self.bias_threshold}
- Enhanced pattern detection
- Multiple bias type detection
- Contextual analysis included

FILES CREATED:
- Main Dataset: {main_file}
- Training Set: {train_file} 
- Validation Set: {val_file}
- Test Set: {test_file}
- Detailed Analysis: {detailed_file}

READY FOR BALANCED ML TRAINING!
"""
        
        summary_file = f"{base_filename}_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        files_created['summary'] = summary_file
        
        return files_created


def main():
    """Main function for balanced bias detection"""
    
    # Configuration
    EMAIL = "phanipulagala619@gmail.com"  # ← UPDATE THIS
    API_KEY = "a40c28e45d89c0f8d3b429e2245f9f755c08"  # ← UPDATE THIS
    TARGET_SIZE = 1500
    TARGET_BIAS_RATE = 0.25  # Target 25% bias rate
    
    print("="*70)
    print("🏥 BALANCED MEDICAL ABSTRACTS BIAS DETECTOR")
    print("="*70)
    print(f"📧 Email: {EMAIL}")
    print(f"🔑 API Key: {'*' * (len(API_KEY) - 4) + API_KEY[-4:] if len(API_KEY) > 4 else 'NOT SET'}")
    print(f"🎯 Target: {TARGET_SIZE} abstracts")
    print(f"⚖️  Target Bias Rate: {TARGET_BIAS_RATE*100:.1f}%")
    print(f"🔧 Enhanced Detection: Multiple bias patterns, lower threshold")
    print("="*70)
    
    if EMAIL == "your_email@gmail.com" or API_KEY == "your_pubmed_api_key":
        print("❌ ERROR: Please update EMAIL and API_KEY!")
        return None, None
    
    # Initialize detector
    logger.info("🚀 Initializing Balanced Bias Detector...")
    try:
        detector = BalancedBiasDetector(EMAIL, API_KEY)
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        return None, None
    
    # Collect balanced dataset
    logger.info(f"🔍 Starting balanced collection...")
    try:
        dataset = detector.collect_balanced_dataset(TARGET_SIZE, TARGET_BIAS_RATE)
    except Exception as e:
        logger.error(f"❌ Collection failed: {e}")
        return None, None
    
    if len(dataset) == 0:
        logger.error("❌ No data collected")
        return None, None
    
    # Save dataset
    logger.info("💾 Saving balanced dataset...")
    try:
        files_created = detector.save_dataset(dataset)
    except Exception as e:
        logger.error(f"❌ Save failed: {e}")
        return dataset, None
    
    # Final summary
    final_dataset = dataset[['abstract', 'bias_label']].copy()
    biased_count = final_dataset['bias_label'].sum()
    bias_rate = final_dataset['bias_label'].mean() * 100
    
    print("\n" + "="*70)
    print("🎉 BALANCED COLLECTION COMPLETE!")
    print(f"📊 Total: {len(final_dataset)} abstracts")
    print(f"📈 Biased: {biased_count} ({bias_rate:.1f}%)")
    print(f"📉 Non-Biased: {len(final_dataset) - biased_count} ({100-bias_rate:.1f}%)")
    print(f"✅ Bias Rate Target: {TARGET_BIAS_RATE*100:.1f}% → Achieved: {bias_rate:.1f}%")
    print(f"📁 Main file: {files_created['final_dataset']}")
    print("="*70)
    
    return final_dataset, files_created


if __name__ == "__main__":
    print("🔧 BALANCED BIAS DETECTION SYSTEM")
    print("✅ Enhanced patterns for better bias detection")
    print("✅ Lower threshold for more sensitivity") 
    print("✅ Target bias rate balancing")
    print("✅ Multiple bias type detection")
    print("\n🚀 STARTING BALANCED COLLECTION...")
    
    final_dataset, files_created = main()
    
    if final_dataset is not None:
        bias_rate = final_dataset['bias_label'].mean() * 100
        print(f"\n🎯 SUCCESS! Achieved {bias_rate:.1f}% bias rate")
        print(f"📈 Now you have both biased and non-biased abstracts!")
    else:
        print("\n❌ FAILED - Check error messages above")