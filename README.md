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
   git clone https://github.com/BrauoOue/causal-graph-llm-evaluator.git
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

## Disclaimer - Pre-computed Results

**Important Note**: The `output/` folder contains 100 pre-computed results from a sample run of the evaluation pipeline. These are provided as examples so users can immediately explore the output format and structure without running the full pipeline. The `eda.ipynb` notebook contains exploratory data analysis (EDA) of these pre-computed results along with basic visualizations and insights.

### Output Folder Structure Explained

The output directory contains four main types of files for each dataset:

#### 1. Predictions (`output/predictions/`)
Contains the raw predictions made by the causal reasoning agent for each dataset. Each file includes:

- **id**: Unique identifier for each question/sample
- **predicted_answer**: The answer the agent predicted
- **correct_answer**: The actual correct answer
- **is_correct**: Boolean indicating whether the prediction was correct
- **is_valid_choice**: Boolean indicating whether the answer was among the valid choices
- **explanation**: The agent's reasoning behind the prediction
- **correct_explanation**: The reference/correct explanation
- **confidence**: The agent's confidence level in the prediction (0-1)
- **question_type**: Whether the question asked for a "cause" or "effect"
- **cost**: API cost for this specific prediction

#### 2. Explanations (`output/explanations/`)
Contains the evaluation of explanation quality for each dataset:

- **id**: Unique identifier matching the predictions
- **correct_explanation**: Boolean indicating whether the evaluation agent determined the given explanation and reference explanation are semantically and logically equivalent
- **confidence**: Confidence level of the explanation evaluation (0-1)
- **predicted_explanation**: The original explanation from the prediction agent
- **reference_explanation**: The ground truth explanation
- **cost**: API cost for this explanation evaluation

#### 3. Results (`output/results/`)
Contains aggregated metrics and summary statistics for each dataset:

- **dataset**: Name of the dataset being evaluated (e.g., "code", "math", "text", "e")
- **predictions_model**: The LLM model used for generating predictions (e.g., "gpt-4o-mini")
- **total_predictions**: Total number of predictions made for this dataset
- **correct_predictions**: Number of predictions that matched the correct answer
- **valid_choice_predictions**: Number of predictions that were valid choices (within the given options)
- **prediction_accuracy**: Ratio of correct predictions to total predictions (0.0 to 1.0)
- **predictions_execution_time**: Total time taken to generate all predictions (in seconds)
- **predictions_total_cost**: Total API cost for generating predictions (in USD)
- **explanations_model**: The LLM model used for evaluating explanations
- **total_explanations**: Total number of explanations evaluated
- **correct_explanations**: Number of explanations deemed semantically/logically correct
- **explanations_accuracy**: Ratio of correct explanations to total explanations (0.0 to 1.0)
- **average_explanation_confidence**: Mean confidence score across all explanation evaluations (0.0 to 1.0)
- **explanations_execution_time**: Total time taken to evaluate all explanations (in seconds)
- **explanations_total_cost**: Total API cost for explanation evaluations (in USD)

#### 4. Metadata (`output/metadata/`)
Contains execution metadata for both predictions and explanations:

**Predictions metadata fields:**
- **name**: Dataset name identifier
- **cost**: Total API cost incurred for prediction generation (in USD)
- **time**: Total execution time for prediction generation (in seconds)
- **model**: LLM model used for predictions

**Explanations metadata fields:**
- **name**: Dataset name identifier  
- **cost**: Total API cost incurred for explanation evaluation (in USD)
- **time**: Total execution time for explanation evaluation (in seconds)
- **model**: LLM model used for explanation evaluation

These pre-computed results demonstrate the framework's capabilities across different domains (code, mathematics, text, and e-care datasets) and provide immediate insights into model performance without requiring API calls.


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
- **borismoko** (Boris Smokovski)

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
