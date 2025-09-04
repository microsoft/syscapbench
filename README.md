# VanSys: Benchmarking AI for System Engineering

VanSys is a comprehensive benchmarking framework designed to evaluate the performance of Large Language Models (LLMs) and AI systems on various system engineering tasks. The project includes multiple specialized benchmarks for caching systems, course examinations, project assignments, and provides both CLI tools and SDK for easy integration.

## Project Structure

### Benchmarks

- **Cache Benchmark** (`benchmarks/cache_bench/`) - Evaluates AI performance on cache system optimization and simulation tasks
- **Course Exam Benchmark** (`benchmarks/course_exam_bench/`) - Tests LLM understanding of system engineering concepts through MIT course exams (54 questions across 4 exams)
- **Course Project Benchmark** (`benchmarks/course_project_bench/`) - Assesses AI capability on practical system engineering lab projects
- **Example Benchmark** (`benchmarks/example_bench/`) - Template and reference implementation for creating new benchmarks

### Core Components

- **CLI Tools** (`cli/`) - Command-line interface for running benchmarks and managing evaluations
- **SDK** (`sdk/`) - Software development kit providing evaluators, LLM interfaces, and utility functions
- **Docker Support** - Each benchmark includes Dockerfiles for containerized execution

## Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized execution)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/vansys.git
   cd vansys
   ```

2. Install dependencies for a specific benchmark:

   ```bash
   cd cli
   ./install.sh
   ```
3. Each benchmark includes an `env.toml` file for configuration. You should add your own llm endpoint url and key there.

### Running Benchmarks

#### Using CLI

```bash
cd cli
./run.sh
```
#### Output Format

Benchmarks generate standardized outputs:

- `result.jsonl`: Detailed evaluation results
- `summary.json`: Aggregated performance metrics
- Test-specific breakdowns and comparisons


## Available Benchmarks

### Course Exam Benchmark

Evaluates LLM performance on system engineering course materials from MIT, including:

- 6.5840 Distributed System Engineering (Spring 2024/2025)
- 6.1810 Operating System Engineering (Fall 2024)
- Coverage: 54 questions across operating systems and distributed systems
- Question types: Single-choice, multiple-choice, and short-answer

**Sample Results:**

| Model | Score Range | Student Average |
|-------|-------------|-----------------|
| GPT-4 | 24-54/95 | 77/95 |
| GPT-4o | 24-55/95 | 77/95 |
| O3-mini | 25-52/95 | 77/95 |

### Cache Benchmark

Tests AI systems on cache optimization tasks including:

- Cache policy implementation and evaluation
- Trace-based simulation analysis  
- Performance optimization strategies
- Support for multiple cache algorithms (LRU, LFU, FIFO, etc.)

### Course Project Benchmark

Assesses practical system engineering skills through lab exercises and project implementations.

## Development

### Creating New Benchmarks

Use the `example_bench` as a template:

1. Copy the example benchmark structure
2. Implement your specific evaluation logic in `src/main.py`
3. Update the README with benchmark-specific details
4. Add test cases in the `tests/` directory

### Code Quality

The project uses Ruff for code formatting and linting:

- Line length: 120 characters
- Target Python version: 3.9+
- Google-style docstrings
- Comprehensive linting rules for code quality


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
