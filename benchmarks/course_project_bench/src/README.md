##

## Setup

```bash
git clone https://github.com/SWE-agent/SWE-agent.git
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip && pip install --editable .
sweagent --help
```

## Run

````bash
export AZURE_API_BASE="XXX"
export AZURE_OPENAI_API_KEY="XXX"



sweagent run --config config_aoi.yaml --env.repo.path=test-repo --problem_statement.path=test-repo/problem_statements/go.md --output_dir="test"

sweagent run --config config_aoi.yaml --env.repo.github_url=git://g.csail.mit.edu/6.5840-golabs-2024 --problem_statement.path=6.5840/problem_statements/simple.md --output_dir="test"

sweagent run --config config_aoi.yaml --env.repo.path=6.5840 --problem_statement.path=6.5840/problem_statements/1.md --output_dir="test" --env.deployment.image=golang:1.22

sweagent run --config config_aoi.yaml --env.repo.path=projects/test-repo --problem_statement.path=problems/test-repo-problems/1.md --output_dir="example_patch/" --env.deployment.image=xuafeng/swe-go-python:latest --env.post_startup_commands='["export PATH=/usr/local/go/bin:${PATH} && export HOME=/tmp"]'

```

```

INFO     Trajectory saved to trajectories/fuchur/azure-gpt4__problem__coding_challenge__t-0.00__p-0.95__c-3.00__install-1/26d111.traj
INFO     Saved predictions to
         trajectories/fuchur/azure-gpt4__problem__coding_challenge__t-0.00__p-0.95__c-3.00__install-1/all_preds.jsonl
╭──────────────────────────── 🎉 Submission successful 🎉 ────────────────────────────╮
│ SWE-agent has produced a patch that it believes will solve the issue you submitted! │
│ Use the code snippet below to inspect or apply it!                                  │
╰─────────────────────────────────────────────────────────────────────────────────────╯

 # The patch has been saved to your local filesystem at:
 PATCH_FILE_PATH='/Users/fuchur/Documents/24/git_sync/SWE-agent/trajectories/fuchur/azure-gpt4__problem__coding_challenge__t-0.00__p-0
 5__c-3.00__install-1/patches/26d111.patch'
 # Inspect it:
 cat "${PATCH_FILE_PATH}"
 # Apply it to a local repository:
 cd <your local repo root>
 git apply "${PATCH_FILE_PATH}"

 ```

 ## Evaluation

 ```bash
python eval.py --task_id "example_0" --project_path "projects/test-repo" --patch "example_patch/1c2844/1c2844.patch" --test_method "pip install -e . && pytest tests/test_tribonaccy.py"

 ```


 ## One-click run

 ```bash
 ./run_example.sh example_0 projects/test-repo problems/test-repo-problems/1.md "pip install -e . && pytest tests/test_tribonaccy.py" "xuafeng/swe-go-python:latest"
 ```
