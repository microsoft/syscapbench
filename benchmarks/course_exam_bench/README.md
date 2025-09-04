# Sytsem Course Exam Benchmark

## Introduction

This benchmark evaluates the performance of Large Language Models (LLMs) on system course exams. Currently, this benchmark includes 4 exams from MIT, in total 54 questions, covering various topics such as operating system and distributed system. It contains single-choice questions, multiple-choice questions, and short-answer questions. The questions are designed to test the understanding of system concepts and problem-solving skills.

## Task Details

- **Input**: The questions in the system course exams. It include single-choice questions, multiple-choice questions, and short-answer questions.
- **Output**: The answers to the questions. The output can be in the form of selected options for single-choice and multiple-choice questions, and detailed explanations for short-answer questions. For single-choice and multiple-choice questions, the output should be the selected option(s) (e.g., "A", "B", "C", etc.). For short-answer questions, the output should be a detailed explanation or answer to the question.

- **Evaluation**: For single-choice and multiple-choice questions, the evaluation is to compare the selected option(s) with the ground truth answers provided in the exam papers. The evaluation is binary: correct or incorrect. For multiple-choice questions, partial credit can be given if some of the selected options are correct. For short-answer questions, the evaluation is based on the correctness and completeness of the answer, which can be subjective and may require human evaluation or a predefined rubric. We use LLM combined with human defined rubric to evaluate the short-answer questions.

## Eval Results

You can see the detailed information of each exam in the table below.

| Course                                                                 | # of questions | Score (gpt-4 1) (score/total) | Score (gpt-4o) (score/total) | Score (o3-mini) (score/total) | Student Score (max/average/medium) |
|------------------------------------------------------------------------|----------------|-------------------------------|------------------------------|-------------------------------|-------------------------------------|
| 6.5840 Distributed System Engineering: Spring 2025 Exam I              | 11             | 29/65                         | 27/65                        | 25/65                         | 65/ **51.8** /52                    |
| 6.5840 Distributed System Engineering: Spring 2024 Exam I              | 15             | 54/95                         | 55/95                        | 42/95                         | 95/ **77** /78                      |
| 6.5840 Distributed System Engineering: Spring 2024 Exam II             | 14             | 24/71                         | 24/71                        | 36/71                         | 72/ **56.6** /57                    |
| 6.1810 Fall 2024 MIT 6.1810 Operating System Engineering               | 14             | 35/70                         | 40/70                        | 52/70                         | 65/ **49.8** /49                    |

## Benchmark Setup

### Test in Docker

To test your benchmark in a Docker container, follow these steps:

1. Build the Docker image using the provided Dockerfile. You can do this by running the following command in the terminal:

   ```sh
   docker build -t your_benchmark_image .
   ```

2. Once the image is built, you can run it using the following command:

   ```sh
   docker run -it --rm your_benchmark_image
   # docker run --rm your_benchmark_image
   ```

3. Inside the container, navigate to the appropriate directory and execute the benchmark script to start the testing process.

   ```sh
   ./run.sh
   ```

### Maunaly Test

To manually test your benchmark, follow these steps:

#### Install Dependencies

To install and configure your benchmark, follow these steps:

1. configure `env.toml` to set LLM API endpoint
2. install dependencies

```bash
./install.sh
```

#### Run

To run your benchmark and obtain results for a specific task and model, follow these steps:

1. Review the `run.sh` script to understand the expected commands and parameters.
2. Execute the `run.sh` script to start the benchmark. The script will guide you through the process and generate the results.

```bash
./run.sh "gpt-4o" 
```

or

```bash
python3 src/main.py --model_name $MODEL_NAME # default output: ./outputs/system_course)bench___${MODEL_NAME}___$(date +"%Y-%m-%d_%H-%M-%S")

# or specify the save path
python3 src/main.py --model_name $MODEL_NAME --save_path ./outputs/BAISysEducation___${MODEL_NAME}___$(date +"%Y-%m-%d_%H-%M-%S")
```

### Output Description

- `result.jsonl`: Detailed output information
- `summary.json`: Summary of model results
  - `reference`: Original test scores (ground truth student performance)
  - `score`: Test scores
  - `score_by_test_paper`: Test score by test paper
