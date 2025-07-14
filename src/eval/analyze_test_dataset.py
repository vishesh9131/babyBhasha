#!/usr/bin/env python3





ENTER_YOUR_PATH = 'test.txt'





# #################################

import sys
sys.path.append('.')
from dataset_analyzer import DatasetAnalyzer

def analyze_test_dataset():
    """Analyze the test.txt dataset with different tokenization methods"""
    
    analyzer = DatasetAnalyzer()
    
    print("🔍 ANALYZING test.txt DATASET")
    print("=" * 60)
    
    # Analyze with different tokenization methods
    tokenization_methods = ['chars', 'words', 'subwords']
    
    results = {}
    
    for method in tokenization_methods:
        print(f"\n{'🔤' if method == 'chars' else '📝' if method == 'words' else '🧩'} ANALYSIS WITH {method.upper()} TOKENIZATION")
        print("=" * 60)
        
        try:
            result = analyzer.analyze_dataset(ENTER_YOUR_PATH, tokenization=method)
            results[method] = result
            
            # Print summary for quick comparison
            stats = result['dataset_stats']
            print(f"\n📊 QUICK SUMMARY ({method}):")
            print(f"  Tokens: {stats['num_tokens']:,}")
            print(f"  Vocab: {stats['vocab_size']:,}")
            print(f"  Ratio: {stats['num_tokens']/stats['vocab_size']:.1f} tokens/vocab")
            
        except Exception as e:
            print(f"❌ Error analyzing with {method}: {e}")
    
    # Compare tokenization methods
    print(f"\n📈 TOKENIZATION METHOD COMPARISON")
    print("=" * 60)
    print(f"{'Method':<12} {'Tokens':<12} {'Vocab':<8} {'Best Config (Moderate)'}")
    print("-" * 60)
    
    for method in tokenization_methods:
        if method in results:
            stats = results[method]['dataset_stats']
            configs = results[method]['model_configs']
            
            # Get moderate config
            moderate_config = configs.get('moderate')
            if moderate_config:
                config_str = f"{moderate_config['total_params']/1e6:.1f}M params"
            else:
                config_str = "Too small"
            
            print(f"{method:<12} {stats['num_tokens']:<12,} {stats['vocab_size']:<8,} {config_str}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 60)
    
    if 'chars' in results:
        char_stats = results['chars']['dataset_stats']
        char_configs = results['chars']['model_configs']
        
        print(f"🎯 RECOMMENDED APPROACH: Character-level tokenization")
        print(f"   Reason: Smallest vocabulary ({char_stats['vocab_size']} chars) = more params for model layers")
        
        moderate_config = char_configs.get('moderate')
        if moderate_config:
            print(f"\n🏗️  SUGGESTED MODEL CONFIGURATION:")
            print(f"   d_model: {moderate_config['d_model']}")
            print(f"   n_layers: {moderate_config['n_layers']}")
            print(f"   expand: {moderate_config['expand']}")
            print(f"   vocab_size: {moderate_config['vocab_size']}")
            print(f"   Total parameters: {moderate_config['total_params']:,} ({moderate_config['total_params']/1e6:.2f}M)")
            print(f"   Training data ratio: {moderate_config['tokens_per_param']:.1f} tokens/param")
            
            print(f"\n💻 CODE TO USE THIS CONFIG:")
            print(f"   model_args = ModelArgs(")
            print(f"       d_model={moderate_config['d_model']},")
            print(f"       n_layer={moderate_config['n_layers']},")
            print(f"       vocab_size={moderate_config['vocab_size']},")
            print(f"       expand={moderate_config['expand']}")
            print(f"   )")
    
    return results

if __name__ == '__main__':
    results = analyze_test_dataset() 