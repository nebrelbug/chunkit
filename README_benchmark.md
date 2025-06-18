# Tokenizer Benchmark System

A comprehensive system for benchmarking and comparing tokenizers across multiple languages and domains.

## 🎯 Overview

This benchmark system allows you to:

- **Compare 12+ tokenizers** (GPT-2/4, LLaMA-3.3, Mixtral, Gemma, Qwen2, DeepSeek-V3, Yi, Phi-3, BLOOM, CodeT5, StarCoder, + auto-discovered custom tokenizers)
- **Test across languages** (English, Chinese, Russian, German, Japanese, Spanish, French)
- **Test across domains** (Web content, Programming languages)
- **Measure key metrics** (compression ratio, encoding speed, vocabulary utilization)
- **Generate visualizations** (charts, heatmaps, comparison tables)

## 🚀 Quick Start

### 1. Run a Basic Benchmark

```bash
# Small test with 10 samples per dataset
python tokenizer_benchmark.py --config configs/benchmark.yaml --samples 10 --output ./test_benchmark

# Larger benchmark with 100 samples per dataset
python tokenizer_benchmark.py --config configs/benchmark.yaml --samples 100 --output ./full_benchmark
```

### 2. View Results

```bash
# Print summary to console
python benchmark_visualizer.py --results ./test_benchmark --summary-only

# Generate all visualizations
python benchmark_visualizer.py --results ./test_benchmark
```

## 📊 What Gets Measured

### Core Metrics

- **Compression Ratio**: Characters per token (higher = better compression)
- **Tokens per Character**: Inverse of compression ratio
- **Encoding Speed**: Time to tokenize text (lower = faster)
- **Vocabulary Size**: Size of tokenizer vocabulary

### Comparisons

- **Overall Performance**: Best tokenizer across all languages/domains
- **Language-Specific**: Best tokenizer for each language
- **Domain-Specific**: Performance on web content vs. code
- **Statistical Analysis**: Means, standard deviations, min/max values

## 🔧 Configuration

### Benchmark Configuration (`configs/benchmark.yaml`)

```yaml
training:
  total_samples: 10_000 # Total samples across all datasets
  streaming_enabled: true
  output_dir: "./benchmark_test"
  temperature: 0.3
  min_samples_per_lang: 100
  max_samples_per_lang: 1000

datasets:
  # English baseline
  - path: "HuggingFaceFW/fineweb"
    description: "English web data"
    subsets:
      - name: "sample-10BT"
        priority: 5

  # Multilingual content
  - path: "HuggingFaceFW/fineweb-2"
    description: "Multilingual web data"
    subsets:
      - name: "rus_Cyrl" # Russian
        priority: 4
      - name: "cmn_Hani" # Chinese
        priority: 4
      # ... more languages

  # Programming languages
  - path: "bigcode/starcoderdata"
    description: "Programming languages"
    subsets:
      - name: "python"
        priority: 4
        text_column: "content"
      - name: "javascript"
        priority: 3
        text_column: "content"
      # ... more languages
```

### Tokenizers Tested

The benchmark automatically includes:

**Legacy Tokenizers:**

- `GPT-2` - OpenAI's GPT-2 tokenizer
- `GPT-4` - OpenAI's GPT-4 tokenizer (via Xenova)
- `CodeT5` - Salesforce's CodeT5 tokenizer
- `StarCoder` - BigCode's StarCoder tokenizer (requires authentication)

**Modern State-of-the-Art Tokenizers:**

- `LLaMA-3.3` - Meta's latest LLaMA 3.3 70B tokenizer
- `Mixtral` - Mistral AI's Mixtral 8x22B tokenizer (requires sentencepiece)
- `Gemma` - Google's Gemma 7B tokenizer
- `Qwen2` - Alibaba's Qwen2 72B tokenizer
- `DeepSeek-V3` - DeepSeek's latest V3 tokenizer
- `Yi` - 01.AI's Yi 34B tokenizer
- `Phi-3` - Microsoft's Phi-3 Small tokenizer (auto-handles trust_remote_code)
- `BLOOM` - BigScience's BLOOM tokenizer

**Custom Tokenizers:**

- **Automatic Discovery**: Scans `./tokenizers/` directory recursively for any `tokenizer.json` files
- **Smart Naming**: Generates display names from folder structure (e.g., `train-simple-500k` → `Custom-Simple-500K`)
- **Zero Configuration**: Just train your tokenizers and they'll be automatically included in benchmarks

## 📈 Output Files

### Results Directory Structure

```
benchmark_results/
├── benchmark_results.csv          # Detailed results for every test
├── benchmark_summary.json         # Summary statistics and winners
├── benchmark_aggregated.csv       # Aggregated stats by tokenizer/language
└── visualizations/                # Generated charts and tables
    ├── compression_by_tokenizer.png
    ├── compression_by_language.png
    ├── compression_by_domain.png
    ├── performance_heatmap.png
    ├── encoding_speed.png
    └── comparison_table.html
```

### Key Files Explained

**`benchmark_results.csv`** - Raw data with every test result:

- `tokenizer_name`, `language`, `domain`, `sample_id`
- `original_chars`, `token_count`, `compression_ratio`
- `encoding_time_ms`, `vocab_size`

**`benchmark_summary.json`** - High-level insights:

- Best tokenizer overall and by language
- Average compression ratios
- Performance summaries

**`comparison_table.html`** - Interactive HTML table with:

- Sortable columns
- Statistical summaries
- Best performer highlighting

## 🎨 Visualizations

### 1. Overall Compression Comparison

Bar chart showing average compression ratio by tokenizer with error bars.

### 2. Language-Specific Performance

Box plots showing compression distribution for each language across tokenizers.

### 3. Domain Performance

Violin plots comparing web content vs. code performance.

### 4. Performance Heatmap

Color-coded matrix showing tokenizer performance across all languages.

### 5. Encoding Speed

Bar chart comparing tokenization speed (lower = better).

## 🔍 Example Results

From our test benchmark:

```
🏆 BEST TOKENIZER OVERALL: GPT-4

📈 AVERAGE COMPRESSION BY TOKENIZER:
  GPT-4          : 3.430 chars/token
  LLaMA          : 2.773 chars/token
  GPT-2          : 2.396 chars/token
  CodeT5         : 2.348 chars/token

🌍 BEST TOKENIZER BY LANGUAGE:
  Chinese     : GPT-4
  German      : GPT-4
  English     : GPT-4
  Python      : GPT-4
  JavaScript  : GPT-4
  Russian     : LLaMA  # Interesting exception!
```

## ⚙️ Advanced Usage

### Custom Dataset Configuration

Create your own benchmark configuration:

```yaml
# my_benchmark.yaml
training:
  total_samples: 5000

datasets:
  - path: "my-org/my-dataset"
    description: "Custom dataset"
    subsets:
      - name: "my-subset"
        priority: 5
        text_column: "text" # or "content", "code", etc.
```

### Adding Custom Tokenizers

The benchmark automatically discovers all tokenizers in your `./tokenizers/` directory:

```bash
# Your trained tokenizers (any structure works)
./tokenizers/train-simple-500k/tokenizer.json     → "Custom-Simple-500K"
./tokenizers/train-simple-50M/tokenizer.json      → "Custom-Simple-50M"
./tokenizers/my-multilingual/tokenizer.json       → "Custom-My-Multilingual"
./tokenizers/experimental/v1/tokenizer.json       → "Custom-V1"
```

**No configuration needed** - just train your tokenizers and run the benchmark!

### Filtering Results

You can filter the CSV results for specific analysis:

```python
import pandas as pd

# Load results
df = pd.read_csv('benchmark_results/benchmark_results.csv')

# Filter for specific language
python_results = df[df['language'] == 'Python']

# Compare only specific tokenizers
gpt_comparison = df[df['tokenizer_name'].isin(['GPT-2', 'GPT-4'])]

# Focus on compression ratios > 3.0
high_compression = df[df['compression_ratio'] > 3.0]
```

## 🛠️ Dependencies

```bash
pip install datasets transformers tokenizers pandas matplotlib seaborn omegaconf

# Optional: For Mixtral tokenizer support
pip install sentencepiece
```

## 🎯 Use Cases

### 1. **Tokenizer Selection**

Compare different tokenizers to choose the best one for your multilingual project.

### 2. **Training Validation**

Benchmark your custom-trained tokenizers against established baselines.

### 3. **Language Coverage Analysis**

Identify which tokenizers perform best for specific languages or domains.

### 4. **Research & Development**

Generate data for papers, blog posts, or technical reports on tokenizer performance.

### 5. **Production Optimization**

Balance compression efficiency vs. encoding speed for your specific use case.

## 📝 Tips & Best Practices

### Sample Size Recommendations

- **Quick test**: 10-50 samples per dataset
- **Development**: 100-500 samples per dataset
- **Production benchmark**: 1000+ samples per dataset

### Performance Considerations

- Use `--samples 10` for quick tests
- Larger samples give more reliable statistics but take longer
- Consider running overnight for comprehensive benchmarks

### Interpreting Results

- **Higher compression ratio = better** (more characters per token)
- **Lower encoding time = better** (faster tokenization)
- Look for consistency across languages, not just averages
- Consider domain-specific performance (web vs. code)

## 🔮 Future Enhancements

Potential improvements:

- Support for more tokenizer formats
- Memory usage benchmarking
- Token distribution analysis
- Subword fertility metrics
- Custom evaluation datasets
- Integration with MLflow/Weights & Biases

---

## 🤝 Contributing

Feel free to extend the benchmark system with:

- Additional tokenizers
- New languages/domains
- More sophisticated metrics
- Better visualizations

Happy benchmarking! 🚀
