"""
Usage:
    python src/processor.py                          # Extract text from all PDFs
    python src/processor.py --pdf filename.pdf       # Extract specific PDF

Output:
    - data/processed/documents.csv (extracted text indexed by filename)
"""

import os
import csv
from pathlib import Path
from typing import List, Dict
import PyPDF2
import argparse

class PDFProcessor:
    def __init__(self, pdf_dir: str = "data/pdfs"):
        self.pdf_dir = Path(pdf_dir)
        self.documents = []
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            return text.strip()
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            return ""
    
    def process_all_pdfs(self, specific_file: str = None) -> List[Dict]:
        if specific_file:
            pdf_files = [self.pdf_dir / specific_file]
            if not pdf_files[0].exists():
                print(f"Error: {specific_file} not found in {self.pdf_dir}")
                return []
        else:
            pdf_files = sorted(self.pdf_dir.glob('*.pdf'))
        
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            print(f"Extracting {pdf_file.name}...")
            text = self.extract_text_from_pdf(pdf_file)
            
            if text:
                self.documents.append({
                    'filename': pdf_file.name,
                    'text': text,
                    'text_length': len(text)
                })
                print(f"  Extracted {len(text)} characters")
            else:
                print(f"  Failed to extract text")
        
        print(f"\nTotal documents processed: {len(self.documents)}")
        return self.documents
    
    def export_to_csv(self, output_file: str = "data/processed/documents.csv"):
        if not self.documents:
            print("No documents to export")
            return
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'text', 'text_length'])
            writer.writeheader()
            writer.writerows(self.documents)
        
        print(f"Exported {len(self.documents)} documents to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from FDA guidance PDFs")
    parser.add_argument('--pdf', type=str, default=None, help='Extract specific PDF file')
    parser.add_argument('--output', type=str, default='data/processed/documents.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    processor = PDFProcessor()
    documents = processor.process_all_pdfs(specific_file=args.pdf)
    processor.export_to_csv(output_file=args.output)