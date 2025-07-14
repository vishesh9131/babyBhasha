import pandas as pd
import numpy as np
import os
import re
from typing import Union, Tuple, Dict
import math

class DatasetAnalyzer:
    """
    Analyzes text datasets and estimates optimal model parameters based on dataset size.
    
    Rules of thumb used:
    - Conservative: 100 tokens per parameter (very safe)
    - Moderate: 50 tokens per parameter (recommended)
    - Aggressive: 20 tokens per parameter (minimum viable)
    - Very Aggressive: 10 tokens per parameter (risky, may overfit)
    """
    
    def __init__(self):
        self.ratios = {
            'ultra_conservative': 200,  # 200 tokens per param (very very safe)
            'conservative': 100,        # 100 tokens per param (very safe)
            'moderate': 50,             # 50 tokens per param (recommended)
            'aggressive': 20,           # 20 tokens per param (minimum viable)
            'very_aggressive': 10,      # 10 tokens per param (risky, may overfit)
            'experimental': 5           # 5 tokens per param (for experimentation)
        }
    
    def load_dataset(self, file_path: str, text_column: str = None) -> str:
        """
        Load text data from CSV or TXT file.
        
        Args:
            file_path: Path to the dataset file
            text_column: For CSV files, specify which column contains text
            
        Returns:
            Combined text string
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif file_ext == '.csv':
            df = pd.read_csv(file_path)
            
            # If text_column not specified, try to find text column
            if text_column is None:
                text_columns = []
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['text', 'content', 'message', 'body', 'description']):
                        text_columns.append(col)
                
                if len(text_columns) == 0:
                    # Use the column with longest average text
                    text_column = max(df.columns, key=lambda x: df[x].astype(str).str.len().mean())
                    print(f"Auto-detected text column: '{text_column}'")
                else:
                    text_column = text_columns[0]
                    print(f"Auto-detected text column: '{text_column}'")
            
            # Combine all text
            text_data = df[text_column].fillna('').astype(str)
            return ' '.join(text_data.tolist())
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use .txt or .csv")
    
    def count_tokens(self, text: str, method: str = 'words') -> int:
        """
        Count tokens in text using different methods.
        
        Args:
            text: Input text
            method: 'words', 'chars', or 'subwords'
            
        Returns:
            Number of tokens
        """
        if method == 'words':
            # Split by whitespace and punctuation
            tokens = re.findall(r'\b\w+\b', text.lower())
            return len(tokens)
        
        elif method == 'chars':
            # Character-level tokenization
            return len(text)
        
        elif method == 'subwords':
            # Rough estimate: ~0.75 words per subword token (like BPE)
            words = len(re.findall(r'\b\w+\b', text.lower()))
            return int(words / 0.75)
        
        else:
            raise ValueError("Method must be 'words', 'chars', or 'subwords'")
    
    def estimate_vocab_size(self, text: str, method: str = 'words') -> int:
        """Estimate vocabulary size for the dataset."""
        if method == 'words':
            unique_words = set(re.findall(r'\b\w+\b', text.lower()))
            return len(unique_words)
        
        elif method == 'chars':
            unique_chars = set(text)
            return len(unique_chars)
        
        elif method == 'subwords':
            # Rough estimate for subword vocabulary
            unique_words = set(re.findall(r'\b\w+\b', text.lower()))
            # Subword vocabs are typically 2-10x smaller than word vocabs
            return max(1000, len(unique_words) // 5)
        
        return len(set(text.split()))
    
    def calculate_model_params(self, vocab_size: int, d_model: int, n_layers: int, expand: int = 2) -> int:
        """
        Calculate total parameters for a Mamba-like model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_layers: Number of layers
            expand: Expansion factor for inner dimension
            
        Returns:
            Total number of parameters
        """
        # Embedding layer (tied with output)
        embedding_params = vocab_size * d_model
        
        # Per layer params (simplified Mamba calculation)
        d_inner = d_model * expand
        d_state = 16  # Default state dimension
        dt_rank = math.ceil(d_model / 16)  # Default dt_rank calculation
        
        per_layer_params = (
            # Input projection
            d_model * d_inner * 2 +
            # Conv1d
            d_inner * 1 * 4 +  # kernel_size=4, groups=d_inner
            # x_proj
            d_inner * (dt_rank + d_state * 2) +
            # dt_proj  
            dt_rank * d_inner +
            # A_log and D parameters
            d_inner * d_state + d_inner +
            # Output projection
            d_inner * d_model +
            # RMS norm
            d_model
        )
        
        total_layer_params = per_layer_params * n_layers
        
        # Final norm
        final_norm_params = d_model
        
        return embedding_params + total_layer_params + final_norm_params
    
    def suggest_model_configs(self, num_tokens: int, vocab_size: int) -> Dict:
        """
        Suggest model configurations based on dataset size.
        
        Args:
            num_tokens: Total number of tokens in dataset
            vocab_size: Vocabulary size
            
        Returns:
            Dictionary with suggested configurations
        """
        configs = {}
        
        for ratio_name, tokens_per_param in self.ratios.items():
            max_params = num_tokens // tokens_per_param
            
            # Search for good d_model and n_layers combination
            best_config = None
            best_param_count = 0
            
            # Try different model configurations (expanded range for small datasets)
            for d_model in [64, 128, 192, 256, 384, 512, 768, 1024]:
                for n_layers in range(1, 25):  # 1 to 24 layers
                    for expand in [1, 2, 4]:  # Include expand=1 for smaller models
                        param_count = self.calculate_model_params(vocab_size, d_model, n_layers, expand)
                        
                        if param_count <= max_params and param_count > best_param_count:
                            best_param_count = param_count
                            best_config = {
                                'd_model': d_model,
                                'n_layers': n_layers,
                                'vocab_size': vocab_size,
                                'expand': expand,
                                'total_params': param_count,
                                'tokens_per_param': num_tokens / param_count if param_count > 0 else 0
                            }
            
            # If no config found, try to find the smallest possible model
            if best_config is None and ratio_name == 'experimental':
                min_config = None
                min_param_count = float('inf')
                
                for d_model in [32, 64, 96, 128]:
                    for n_layers in range(1, 5):
                        for expand in [1, 2]:
                            param_count = self.calculate_model_params(vocab_size, d_model, n_layers, expand)
                            if param_count < min_param_count:
                                min_param_count = param_count
                                min_config = {
                                    'd_model': d_model,
                                    'n_layers': n_layers,
                                    'vocab_size': vocab_size,
                                    'expand': expand,
                                    'total_params': param_count,
                                    'tokens_per_param': num_tokens / param_count if param_count > 0 else 0
                                }
                
                best_config = min_config
            
            configs[ratio_name] = best_config
        
        return configs
    
    def analyze_dataset(self, file_path: str, text_column: str = None, 
                       tokenization: str = 'words') -> Dict:
        """
        Complete analysis of a dataset file.
        
        Args:
            file_path: Path to dataset file
            text_column: Column name for CSV files
            tokenization: 'words', 'chars', or 'subwords'
            
        Returns:
            Complete analysis results
        """
        print(f"Analyzing dataset: {file_path}")
        print(f"Tokenization method: {tokenization}")
        print("-" * 50)
        
        # Load data
        text = self.load_dataset(file_path, text_column)
        
        # Count tokens and estimate vocab
        num_tokens = self.count_tokens(text, tokenization)
        vocab_size = self.estimate_vocab_size(text, tokenization)
        
        # Basic statistics
        num_chars = len(text)
        num_lines = text.count('\n') + 1
        avg_line_length = num_chars / num_lines if num_lines > 0 else 0
        
        print(f"Dataset Statistics:")
        print(f"  Total characters: {num_chars:,}")
        print(f"  Total lines: {num_lines:,}")
        print(f"  Average line length: {avg_line_length:.1f} chars")
        print(f"  Total tokens ({tokenization}): {num_tokens:,}")
        print(f"  Estimated vocab size: {vocab_size:,}")
        print(f"  Tokens per vocab item: {num_tokens/vocab_size:.1f}")
        print()
        
        # Get model suggestions
        configs = self.suggest_model_configs(num_tokens, vocab_size)
        
        print("Suggested Model Configurations:")
        print("=" * 50)
        
        for ratio_name, config in configs.items():
            if config:
                print(f"\n{ratio_name.upper()} ({self.ratios[ratio_name]} tokens/param):")
                print(f"  d_model: {config['d_model']}")
                print(f"  n_layers: {config['n_layers']}")
                print(f"  expand: {config['expand']}")
                print(f"  vocab_size: {config['vocab_size']:,}")
                print(f"  Total parameters: {config['total_params']:,} ({config['total_params']/1e6:.2f}M)")
                print(f"  Actual tokens/param: {config['tokens_per_param']:.1f}")
            else:
                print(f"\n{ratio_name.upper()}: Dataset too small for this ratio")
        
        return {
            'file_path': file_path,
            'dataset_stats': {
                'num_chars': num_chars,
                'num_lines': num_lines,
                'num_tokens': num_tokens,
                'vocab_size': vocab_size,
                'tokenization': tokenization
            },
            'model_configs': configs
        }

def main():
    """Example usage of the DatasetAnalyzer"""
    analyzer = DatasetAnalyzer()
    
    # Example: analyze a text file
    # result = analyzer.analyze_dataset('shakespeare.txt', tokenization='chars')
    
    # Example: analyze a CSV file
    # result = analyzer.analyze_dataset('tweets.csv', text_column='text', tokenization='subwords')
    
    print("Dataset Analyzer - Usage Examples:")
    print("=" * 50)
    print("1. For text files:")
    print("   analyzer.analyze_dataset('data.txt', tokenization='chars')")
    print("\n2. For CSV files:")
    print("   analyzer.analyze_dataset('data.csv', text_column='text', tokenization='words')")
    print("\n3. Tokenization options: 'words', 'chars', 'subwords'")
    print("\nThe analyzer will suggest optimal model configurations based on your dataset size!")

if __name__ == '__main__':
    main() 