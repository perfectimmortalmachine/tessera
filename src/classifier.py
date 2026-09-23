"""
HYBRID APPROACH:
  - Classification: Filename-prefix parsing (auditable, deterministic)
  - Keywords: TF-IDF extraction (semantic importance scoring)
  - Confidence: 1.0 for deterministic classification

Usage:
    python src/classifier.py                         # Classify all documents
    python src/classifier.py --query "orthopedic"    # Search for relevant guidance

Output:
    - data/processed/classified_documents.csv (risk class + keywords)
"""

import csv
csv.field_size_limit(int(1e8))

from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import math
import argparse


class FAERSGuidanceClassifier:
    def __init__(self, csv_file: str = "data/processed/documents.csv"):
        self.csv_file = csv_file
        self.documents = []
        self.classified_documents = []
        self.idf_cache = {}
        
        self.load_documents()
        self.compute_idf()
    
    def load_documents(self):
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.documents = list(reader)
        print(f"Loaded {len(self.documents)} documents\n")
    
    def compute_idf(self):
        """
        Compute Inverse Document Frequency across all documents.
        Words common to ALL docs get low IDF (less important).
        Words specific to FEW docs get high IDF (more important).
        """
        word_doc_count = defaultdict(int)
        total_docs = len(self.documents)
        
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'is', 'are', 'was', 'be', 'been', 'being', 'have', 'has', 'had',
            'from', 'with', 'by', 'as', 'of', 'all', 'each', 'every', 'both',
            'that', 'this', 'which', 'what', 'who', 'when', 'where', 'why', 'how'
        }
        
        for document in self.documents:
            text = document.get('text', '').lower()
            words = text.split()
            unique_words = set()
            
            for word in words:
                clean_word = ''.join(c for c in word if c.isalnum())
                if len(clean_word) > 4 and clean_word not in stopwords and not clean_word.isdigit():
                    unique_words.add(clean_word)
            
            for word in unique_words:
                word_doc_count[word] += 1
        
        # IDF = log(total_docs / docs_containing_word)
        for word, doc_count in word_doc_count.items():
            self.idf_cache[word] = math.log(total_docs / doc_count) if doc_count > 0 else 0
        
        print(f"Computed IDF for {len(self.idf_cache)} unique terms\n")
    
    def extract_class_from_filename(self, filename: str) -> str:
        """
        Extract device risk class from filename prefix.
        Deterministic, auditable, ground truth.
        """
        if filename.startswith('class_I_') and not filename.startswith('class_II_') and not filename.startswith('class_III_'):
            return 'Class I'
        elif filename.startswith('class_II_'):
            return 'Class II'
        elif filename.startswith('class_III_'):
            return 'Class III'
        else:
            return 'General'
    
    def extract_keywords_by_tfidf(self, text: str, limit: int = 10) -> List[str]:
        """
        Extract top keywords by TF-IDF score.
        High TF-IDF = important to this doc, rare elsewhere.
        """
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'is', 'are', 'was', 'be', 'been', 'being', 'have', 'has', 'had',
            'from', 'with', 'by', 'as', 'of', 'all', 'each', 'every', 'both'
        }
        
        words = text.lower().split()
        word_freq = {}
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) > 4 and clean_word not in stopwords and not clean_word.isdigit():
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
        
        # Calculate TF-IDF for each word
        tfidf_scores = {}
        doc_length = len(words)
        
        for word, freq in word_freq.items():
            tf = freq / doc_length  # Term Frequency
            idf = self.idf_cache.get(word, 0)  # Inverse Document Frequency
            tfidf_scores[word] = tf * idf
        
        # Sort by TF-IDF, take top N
        top_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [word for word, score in top_words] if top_words else ['regulatory', 'guidance', 'device']
    
    def classify_document(self, document: Dict) -> Dict:
        filename = document.get('filename', '')
        text = document.get('text', '')
        
        # Filename-based classification (deterministic)
        risk_class = self.extract_class_from_filename(filename)
        
        # TF-IDF keyword extraction (semantic)
        keywords = self.extract_keywords_by_tfidf(text)
        
        return {
            'filename': filename,
            'risk_class': risk_class,
            'keywords': ', '.join(keywords),
            'text_length': document.get('text_length', 0)
        }
    
    def classify_all(self) -> List[Dict]:
        print("="*70)
        print("CLASSIFYING DOCUMENTS: FILENAME PREFIX + TF-IDF KEYWORDS")
        print("="*70 + "\n")
        
        for i, document in enumerate(self.documents, 1):
            filename = document.get('filename', '')
            print(f"[{i}/{len(self.documents)}] {filename:45s}", end=' → ', flush=True)
            
            classified = self.classify_document(document)
            self.classified_documents.append(classified)
            
            print(f"{classified['risk_class']:12s}")
        
        print(f"\nClassified {len(self.classified_documents)} documents\n")
        return self.classified_documents
    
    def export_classifications(self, output_file: str = "data/processed/classified_documents.csv"):
        if not self.classified_documents:
            print("No documents to export")
            return
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'risk_class', 'keywords', 'text_length'])
            writer.writeheader()
            writer.writerows(self.classified_documents)
        
        print(f"Exported classifications to {output_file}\n")
    
    def search_guidance(self, query: str) -> List[Dict]:
        query_lower = query.lower()
        results = []
        
        for doc in self.classified_documents:
            filename = doc.get('filename', '').lower()
            keywords = doc.get('keywords', '').lower()
            
            if query_lower in filename or query_lower in keywords:
                results.append(doc)
        
        return results
    
    def generate_summary(self) -> str:
        class_counts = {}
        
        for doc in self.classified_documents:
            risk_class = doc.get('risk_class', 'Unknown')
            class_counts[risk_class] = class_counts.get(risk_class, 0) + 1
        
        summary = """
╔════════════════════════════════════════════════════════════╗
║       TESSERA: GUIDANCE DOCUMENT CLASSIFICATION            ║
╚════════════════════════════════════════════════════════════╝

CLASSIFICATION SUMMARY
──────────────────────
"""
        
        for risk_class in sorted(class_counts.keys()):
            count = class_counts[risk_class]
            percentage = (count / len(self.classified_documents)) * 100
            summary += f"{risk_class:20s}: {count:>3} documents ({percentage:>5.1f}%)\n"
        
        summary += f"\nTotal Documents: {len(self.classified_documents)}\n"
        summary += "="*60 + "\n"
        summary += f"\nAPPROACH:\n"
        summary += f"  Classification: Filename prefix parsing (deterministic, auditable)\n"
        summary += f"  Keywords: TF-IDF scoring (inverse document frequency)\n"
        summary += f"  Confidence: 1.0 (ground truth from filenames)\n"
        
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify FDA guidance documents")
    parser.add_argument('--csv', type=str, default='data/processed/documents.csv', help='Input CSV file path')
    parser.add_argument('--output', type=str, default='data/processed/classified_documents.csv', help='Output CSV file path')
    parser.add_argument('--query', type=str, default=None, help='Search for guidance documents')
    
    args = parser.parse_args()
    
    classifier = FAERSGuidanceClassifier(csv_file=args.csv)
    classifier.classify_all()
    classifier.export_classifications(output_file=args.output)
    
    summary = classifier.generate_summary()
    print(summary)
    
    with open('data/processed/classification_summary.txt', 'w') as f:
        f.write(summary)
    print("Classification summary saved to data/processed/classification_summary.txt\n")
    
    if args.query:
        print(f"=== SEARCH RESULTS FOR '{args.query}' ===\n")
        results = classifier.search_guidance(args.query)
        
        if results:
            for result in results:
                print(f"   {result['filename']}")
                print(f"   Class: {result['risk_class']}")
                print(f"   Keywords: {result['keywords']}\n")
        else:
            print(f"No guidance documents found for '{args.query}'")