"""
Data Processing Script for Prediction Resource Dataset

This script helps convert the CSV data from the Andrea-de-Varda/prediction-resource
repository into the JSON format needed by the frontend visualization.

Usage:
1. Download all_measures.csv from the prediction-resource repository
2. Place it in the same directory as this script
3. Run: python process_data.py
4. The script will generate data.js with the processed data

Requirements:
- pandas
- json
"""

import pandas as pd
import json
import numpy as np
from collections import defaultdict

def load_data(csv_file='all_measures.csv'):
    """Load the CSV data from the prediction-resource dataset."""
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} rows from {csv_file}")
        return df
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Please download it from:")
        print("https://github.com/Andrea-de-Varda/prediction-resource")
        return None

def process_sentences(df):
    """Group words by sentence and create the structure needed for visualization."""
    
    # Metrics available in the dataset
    metrics = [
        'cloze_p_smoothed', 'rating_mean', 'rating_sd', 'cloze_s', 'competition', 'entropy',
        's_GPT2', 's_GPT2_medium', 's_GPT2_large', 's_GPT2_xl',
        's_GPTNeo_125M', 's_GPTNeo', 's_GPTNeo_2.7B',
        'rnn', 'psg', 'bigram', 'trigram', 'tetragram',
        'RTfirstfix', 'RTfirstpass', 'RTgopast', 'RTrightbound', 'self_paced_reading_time',
        'ELAN', 'LAN', 'N400', 'EPNP', 'P600', 'PNP'
    ]
    
    sentences_data = []
    
    # Group by sentence
    sentence_groups = df.groupby('sentence')
    
    for sentence_text, group in sentence_groups:
        # Sort by context_length to maintain word order
        group_sorted = group.sort_values('context_length')
        
        words_data = []
        for _, row in group_sorted.iterrows():
            word_data = {'word': row['word']}
            
            # Add all available metrics
            for metric in metrics:
                if metric in row and pd.notna(row[metric]):
                    word_data[metric] = float(row[metric])
            
            words_data.append(word_data)
        
        sentence_data = {
            'sentence': sentence_text,
            'words': words_data
        }
        
        sentences_data.append(sentence_data)
    
    return sentences_data

def filter_complete_sentences(sentences_data, min_words=3, max_words=15):
    """Filter sentences based on length and data completeness."""
    filtered = []
    
    for sentence_data in sentences_data:
        words = sentence_data['words']
        
        # Check length
        if len(words) < min_words or len(words) > max_words:
            continue
        
        # Check if sentence has reasonable data coverage
        metrics_with_data = 0
        key_metrics = ['cloze_p_smoothed', 'rating_mean', 's_GPT2', 'N400']
        
        for word in words:
            for metric in key_metrics:
                if metric in word:
                    metrics_with_data += 1
                    break
        
        # Keep sentences where most words have at least one key metric
        if metrics_with_data >= len(words) * 0.7:
            filtered.append(sentence_data)
    
    return filtered

def generate_sample_sentences(sentences_data, num_samples=20):
    """Select a representative sample of sentences for the frontend."""
    
    # Sort by various criteria to get diverse examples
    sentences_with_stats = []
    
    for sentence_data in sentences_data:
        words = sentence_data['words']
        
        # Calculate some basic statistics
        cloze_probs = [w.get('cloze_p_smoothed', 0) for w in words if 'cloze_p_smoothed' in w]
        surprisals = [w.get('s_GPT2', 0) for w in words if 's_GPT2' in w]
        
        stats = {
            'sentence_data': sentence_data,
            'length': len(words),
            'avg_cloze_prob': np.mean(cloze_probs) if cloze_probs else 0,
            'avg_surprisal': np.mean(surprisals) if surprisals else 0,
            'cloze_variance': np.var(cloze_probs) if len(cloze_probs) > 1 else 0
        }
        
        sentences_with_stats.append(stats)
    
    # Sort by different criteria and pick diverse examples
    selected = []
    
    # High surprisal sentences (unexpected/difficult)
    high_surprisal = sorted(sentences_with_stats, key=lambda x: x['avg_surprisal'], reverse=True)[:5]
    selected.extend([s['sentence_data'] for s in high_surprisal])
    
    # High predictability sentences (easy)
    high_predictable = sorted(sentences_with_stats, key=lambda x: x['avg_cloze_prob'], reverse=True)[:5]
    selected.extend([s['sentence_data'] for s in high_predictable])
    
    # High variance sentences (mixed difficulty)
    high_variance = sorted(sentences_with_stats, key=lambda x: x['cloze_variance'], reverse=True)[:5]
    selected.extend([s['sentence_data'] for s in high_variance])
    
    # Different lengths
    short_sentences = [s for s in sentences_with_stats if s['length'] <= 6][:2]
    long_sentences = [s for s in sentences_with_stats if s['length'] >= 10][:3]
    selected.extend([s['sentence_data'] for s in short_sentences])
    selected.extend([s['sentence_data'] for s in long_sentences])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_selected = []
    for sentence_data in selected:
        sentence_text = sentence_data['sentence']
        if sentence_text not in seen:
            seen.add(sentence_text)
            unique_selected.append(sentence_data)
    
    return unique_selected[:num_samples]

def export_to_javascript(sentences_data, output_file='data.js'):
    """Export the processed data as a JavaScript file."""
    
    js_content = f"""// Auto-generated data from prediction-resource dataset
// Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
// Source: https://github.com/Andrea-de-Varda/prediction-resource

const realData = {json.dumps(sentences_data, indent=2)};

// Replace the sampleData in script.js with realData to use actual dataset
// Example: change 'const sampleData = [...];' to 'const sampleData = realData;'

console.log(`Loaded ${{realData.length}} sentences from prediction-resource dataset`);
"""
    
    with open(output_file, 'w') as f:
        f.write(js_content)
    
    print(f"Data exported to {output_file}")
    print(f"Total sentences: {len(sentences_data)}")
    print(f"To use this data, replace the sampleData array in script.js with realData")

def print_data_summary(sentences_data):
    """Print a summary of the processed data."""
    
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    
    total_sentences = len(sentences_data)
    total_words = sum(len(s['words']) for s in sentences_data)
    
    print(f"Total sentences: {total_sentences}")
    print(f"Total words: {total_words}")
    print(f"Average words per sentence: {total_words/total_sentences:.1f}")
    
    # Check metric coverage
    metrics_coverage = defaultdict(int)
    for sentence_data in sentences_data:
        for word in sentence_data['words']:
            for metric in word.keys():
                if metric != 'word':
                    metrics_coverage[metric] += 1
    
    print(f"\nMetric coverage (number of words with data):")
    for metric, count in sorted(metrics_coverage.items()):
        percentage = (count / total_words) * 100
        print(f"  {metric}: {count} ({percentage:.1f}%)")
    
    # Show some example sentences
    print(f"\nExample sentences:")
    for i, sentence_data in enumerate(sentences_data[:5]):
        print(f"  {i+1}. {sentence_data['sentence'][:60]}...")

def main():
    """Main processing pipeline."""
    
    print("Processing Prediction Resource Dataset for Frontend Visualization")
    print("="*60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Process sentences
    print("\nProcessing sentences...")
    sentences_data = process_sentences(df)
    print(f"Found {len(sentences_data)} unique sentences")
    
    # Export all sentences, not just a sample or filtered subset
    print("\nExporting all sentences to data.js...")
    export_to_javascript(sentences_data)
    print_data_summary(sentences_data)
    print(f"\n✅ Processing complete!")
    print(f"📁 Output file: data.js")
    print(f"🔧 Next steps:")
    print(f"   1. Copy the contents of data.js")
    print(f"   2. Replace the sampleData array in script.js")
    print(f"   3. Refresh your browser to see the real data")

if __name__ == "__main__":
    main()
