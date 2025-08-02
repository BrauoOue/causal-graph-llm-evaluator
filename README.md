# Causal Graph LLM Evaluator

A comprehensive evaluation framework for testing and benchmarking the causal reasoning capabilities of large language models (LLMs). This project provides tools for generating, evaluating, and analyzing causal explanations across multiple datasets and model configurations.

## Overview

The Causal Graph LLM Evaluator helps researchers and practitioners assess how well language models understand and explain causal relationships across different domains (code, mathematics, and natural language). The framework includes:

- Dataset standardization from various formats
- Configurable prediction pipeline with multiple LLMs
- Explanation generation and evaluation
- Detailed metrics and result analysis
- Comprehensive logging and visualization

## Features

- **Multiple Dataset Support**: Process datasets from different domains (code, mathematics, text)
- **Flexible Model Configuration**: Use different LLM models for predictions and evaluations
- **Parallel Processing**: Efficient batch processing with configurable threading options
- **Custom Prompting**: Override default prompts with domain-specific instructions
- **Comprehensive Logging**: Detailed logs with configurable verbosity levels
- **Evaluation Metrics**: Analyze both prediction accuracy and explanation quality
- **Cost Tracking**: Monitor API costs for predictions and evaluations

## Project Structure

```
causal-graph-llm-evaluator/
├── pipeline.py                 # End-to-end execution pipeline
├── eda.ipynb                   # Jupyter notebook containing exploratory data analysis of precomputed results
├── modules/                    # Core modules
│   ├── __init__.py
│   ├── agent.py                # Main prediction agent
│   ├── conversion.py           # Dataset standardization utilities
│   ├── evaluate.py             # Evaluation logic
│   ├── explanation_evaluation_agent.py  # Explanation quality assessment
│   ├── logger.py               # Custom logging functionality
│   └── prompt_builder.py       # Prompt generation utilities
├── data/                       # Dataset storage
│   ├── datasets/               # Processed datasets
│   ├── other/                  # Additional dataset resources
│   └── raw/                    # Raw input datasets
├── logs/                       # Application logs
├── output/                     # Results storage
│   ├── explanations/           # Explanation results
│   ├── metadata/               # Execution metadata
│   │   ├── explanations/       # Explanation metadata
│   │   └── predictions/        # Prediction metadata
│   ├── predictions/            # Prediction results
│   └── results/                # Final evaluation results
└── prompts/                    # Prompt templates
    ├── explanation/            # Explanation evaluation prompts
    └── prediction/             # Prediction generation prompts
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/causal-graph-llm-evaluator.git
   cd causal-graph-llm-evaluator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   # Create .env file with your API keys
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Basic Usage

Run the evaluation pipeline with default settings:

```bash
python pipeline.py
```

### Advanced Usage

Run with custom settings:

```bash
python pipeline.py --limit 5 --use_manual_prompt --model_predictions gpt-4o --model_explanations gpt-4o \
    --max_tokens_predictions 2000 --max_tokens_explanations 5000 --log_level DEBUG
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--limit` | Maximum number of rows to process from each dataset | `None` (all rows) |
| `--use_manual_prompt` | Enable manual prompt input | `False` |
| `--model_predictions` | Model name for predictions | `gpt-4o-mini` |
| `--model_explanations` | Model name for explanation evaluation | `gpt-4o-mini` |
| `--max_tokens_predictions` | Maximum tokens for prediction responses | `2000` |
| `--max_tokens_explanations` | Maximum tokens for explanation evaluation responses | `5000` |
| `--log_level` | Logging verbosity level | `INFO` |

## Datasets

The framework works with multiple dataset formats:

1. **Code datasets**: Causal relationships in programming contexts
2. **Mathematics datasets**: Causal relationships in mathematical scenarios
3. **Text datasets**: General causal reasoning in natural language
4. **e-Care datasets**: Cause-effect relationship identification

When running the pipeline, you will be prompted to select which datasets to process.

## Output

Results are stored in the `output` directory:

- `predictions/`: Raw model predictions
- `explanations/`: Generated explanations and evaluations
- `metadata/`: Execution metrics (time, cost)
- `results/`: Final evaluation results and analysis

## Logging

Logs are stored in the `logs` directory with color-coded console output for easy monitoring:

- `app_YYYY-MM-DD.log`: General application logs
- `errors_YYYY-MM-DD.log`: Error-specific logs

## Contributors

This project was developed for the course **Intro To Data Science** (Вовед во науката за податоци - ВНП) at the Faculty of Computer Science and Engineering (ФИНКИ), Ss. Cyril and Methodius University, Skopje, Macedonia.

### Development Team
- **Itonkdong** (Viktor Kostadinoski)
- **GogoPro27** (Gorazd Filipovski)
- **Andrea-44** (Andrea Stevanoska)
- **IamMistake** (Nikola Jagurinoski)
- **borismoko** (Boris)

### Instructors
- Prof. Sonja Gievska
- Prof. Slobodan Kalajdziski
- Assistant Martina Toshevska

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- OpenAI for providing the API used in model evaluations
- e-Care dataset creators for causal reasoning benchmarks
- CausalBench for additional testing materials
