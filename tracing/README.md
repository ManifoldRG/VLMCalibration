# SimpleQA Analysis with Infini-gram

This script analyzes SimpleQA dataset by extracting questions and plotting Expected Calibration Error (ECE) vs frequency of text in OLMO training data across different training stages (pretraining and post-training).

## Features

- Extracts clean question text from SimpleQA JSON format
- Queries infini-gram API for frequency counts across OLMO training stages:
  - **Pretraining**: OLMO 2 13B full pretraining data
  - **Post-training**: OLMO 2 13B post-training data (excluding pretraining)
- Calculates Expected Calibration Error (ECE) for frequency bins
- Generates publication-ready plots

## Installation

```bash
pip install requests matplotlib numpy
```

## Usage

```bash
python simpleqa_analysis.py --input zs_exp_records_test_OLMo-2-1124-13B-Instruct.json --output ece_vs_frequency.png
```

### Options

- `--input`: Input SimpleQA JSON file (required)
- `--output`: Output plot file (default: `ece_vs_frequency.png`)
- `--save-results`: Save detailed analysis results to JSON file

### Example with all options

```bash
python simpleqa_analysis.py \
    --input zs_exp_records_test_OLMo-2-1124-13B-Instruct.json \
    --output olmo_calibration_analysis.png \
    --save-results detailed_results.json
```

## Input Format

The script expects JSON data in SimpleQA format with the following fields:
- `question`: Full question prompt (will be parsed to extract clean question text)
- `p_true`: Model confidence score (0-1)
- `correct`: Boolean indicating if the answer was correct

## Output

- **Plot**: ECE vs frequency scatter plot with trend lines for each training stage
- **JSON** (optional): Detailed analysis results including frequency bins and raw data

## How it Works

1. **Question Extraction**: Uses regex to extract clean question text from full prompts
2. **Frequency Counting**: Queries infini-gram API for each question across training stages
3. **Binning**: Groups questions by frequency (log scale bins)
4. **ECE Calculation**: Computes Expected Calibration Error within each frequency bin
5. **Visualization**: Creates scatter plot with point sizes indicating sample counts

## Rate Limiting

The script includes appropriate delays between API calls to respect infini-gram rate limits. Processing time depends on dataset size and API response times.

## Expected Behavior

The analysis should reveal how model calibration (ECE) relates to training data frequency:
- **High frequency text**: May show different calibration patterns due to memorization
- **Low frequency text**: May show higher uncertainty or miscalibration
- **Training stage differences**: Different patterns across pretraining vs post-training data 