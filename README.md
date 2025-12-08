# llm-mcts

This repository contains the code for NeurIPS'23 paper: [Large language models as commonsense knowledge for large-scale task planning](https://arxiv.org/abs/2305.14078). 

We use Large Language Models as both the commonsense world model and the heuristic policy within Monte Carlo Tree Search. LLM's world model provides with MCTS a commonsense prior belief of states for reasoned decision-making. The LLM's heuristic policy guides the search to relevant parts of the tree, substantially reducing the search complexity.

![Figure](media/llm-mcts.jpeg)

## Updates

* [25 Feb 2024] We have updated the code to use the latest version of the OpenAI API. 

## Cite

```
@inproceedings{
  zhao2023large,
  title={Large Language Models as Commonsense Knowledge for Large-Scale Task Planning},
  author={Zirui Zhao and Wee Sun Lee and David Hsu},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=Wjp1AYB8lH}
}
```

## Install

Install the repo: 
```
git clone --recurse-submodules https://github.com/1989Ryan/llm-mcts.git
```

You need to first install virtual home. Please follow with the link at [here](./vh/vh_sim/README_Download.md) as well as the official repository at [here](https://github.com/xavierpuigf/virtualhome) to install.

To intall the dependencies in our method, run
```
pip install -r requirement.txt
```

## Generate Data

We use the code from [here](https://github.com/xavierpuigf/watch_and_help) to generate the data. You can also use the script at [here](./scripts/gene_data.sh) to generate the data. 

To generate data, you need to generate the goal of a domain first, using the command
```
python vh/data_gene/gen_data/vh_init.py \
    --port "{Port Number}" \
    --task {choose your task} \
    --mode {choose one difficulty} \
    --usage {training or testing} \
    --num-per-apartment {a number} 
```

Then, to generate expert data, you need to use
```
python vh/data_gene/testing_agents/gene_data.py \
    --mode {difficulty} \
    --dataset_path {the path to the file generated in the previous step}\
    --base-port {port number}
```

After that, we need to pre-process the expert data
```
python mcts/virtualhome/expert_data.py
```

## Run

Add your openai api key in both `./mcts/virtualhome/llm_model.py` and `./mcts/virtualhome/llm_policy.py`. 

Generate the world model by LLM:
```
python mcts/virtualhome/llm_model.py
```

To run the code for LLM-MCTS, use
```
python mcts/virtualhome/mcts_agent.py \
    --exploration_constant 24 \
    --max_episode_len 50 \
    --max_depth 20 \
    --round 0 \
    --simulation_per_act 2 \
    --simulation_num 100 \
    --discount_factor 0.95  \
    --uct_type PUCT \
    --mode simple \
    --seen_item \
    --seen_apartment\
    --model gpt-3.5-turbo-0125 \
    --seen_comp
```

## Travel planning (tabular dataset + MCTS + local LLM)

This repo also includes a lightweight travel planner that treats the tabular dataset under `./database` (flights, accommodations, restaurants, attractions) as the environment and searches with MCTS. Key pieces:

- `mcts/travel/knowledge_base.py`: loads/normalizes the CSVs.
- `mcts/travel/travel_env.py`: defines a multi-day state (outbound/return flights, accommodation, per-day meal slots, per-day attraction slots), constraints (budget, daily meals, daily attractions), rewards/penalties, and terminal checks.
- `mcts/travel/llm_policy.py`: optional action prior scorer; can use a local LLM or SentenceTransformer embeddings; if unavailable, falls back to uniform priors (torch/transformers are optional).
- `mcts/mcts/mcts.py`: generic MCTS that consumes env observations/valid actions; uses the policy prior in PUCT.
- `scripts/run_travel_mcts.py`: entry point; supports direct CLI args or a natural language query parsed by a local LLM (Ollama-style `/api/generate`).

### State and action space
- State tracks: outbound flight, return flight, one accommodation, meals per day (breakfast/lunch/dinner), attractions per day (slots morning/afternoon/evening/night), total cost, preference hits, violations.
- Actions: `flight_out:*`, `flight_back:*`, `stay:*`, `eat:d{day}:{slot}:*`, `visit:d{day}:{slot}:*`, `finish`.
- Constraints: require outbound/return flights, accommodation, 3 meals per day, at least 2 (up to 3) attractions per day, stay within budget. Rewards penalize violations and reward preference matches/POIs; success only if all hard constraints met.

### Natural language parsing
If `--nl-query` is provided, the script calls a local LLM (default `deepseek-r1:14b` at `LOCAL_LLM_BASE`/`--local-base`) with a simple JSON-extraction prompt to fill origin/destination/start_date/duration/budget/preferences and attraction bounds. Missing fields can be supplied via CLI flags.

### Running the planner
Base example with explicit args:
```
python scripts/run_travel_mcts.py \
  --origin "St. Petersburg" --destination Rockford \
  --start-date 2022-03-16 --days 3 --budget 1700 \
  --local-model deepseek-r1:14b --local-base http://localhost:11434 \
  --device cpu --top-k 3 --debug
```

NL parsing example (origin/destination extracted from query):
```
python scripts/run_travel_mcts.py \
  --nl-query 'Please help me plan a trip from St. Petersburg to Rockford spanning 3 days from March 16th to March 18th, 2022. The travel should be planned for a single person with a budget of $1,700.' \
  --local-model deepseek-r1:14b --local-base http://localhost:11434 \
  --device cpu --top-k 3 --debug
```

Flags of note:
- `--local-model`: local LLM for action priors (optional; if missing, priors are uniform); also used by default for NL parsing.
- `--device`: cpu/mps/cuda:0 etc. (torch is optional; if absent, the planner still runs with uniform priors).
- `--nl-query`: free-form request; `--parser-model/--local-base/--parser-timeout` control the parser call.
- `--top-k`: limit candidates per category to shrink branching factor.
- `--debug`: print root node stats (Q/N) and per-step cost/violations.

### What MCTS does here
- Outer loop: from empty itinerary, repeatedly call `MCTSAgent.search` to pick the next action, apply it to the travel env, and continue until success or max steps.
- Inner loop: standard PUCT—uses env observation/valid actions, policy priors from `TravelLLMPolicy` (or uniform), expands/rolls out to estimate Q, and selects the highest-scoring action.
- Termination: only succeeds when outbound+return flights, accommodation, all meal slots, and minimum attractions per day are satisfied within budget; otherwise finish incurs penalties.

## Travel dataset + local LLM

The repository now includes a lightweight environment that consumes the tabular travel dataset under `./database` (flights, accommodations, restaurants, attractions) and runs MCTS with a local scoring model. Example:
```
python scripts/run_travel_mcts.py \
    --origin Detroit \
    --destination Norfolk \
    --budget 800 \
    --restaurants 2 \
    --attractions 2 \
    --preference seafood \
    --local-model /path/to/local/model
```
If `--local-model` is omitted, the scorer falls back to embedding similarity via SentenceTransformers; set `--embedding-model` to point at a local model if needed. Adjust `--top-k` to limit the number of candidate actions considered each step.

## Acknowledge

This repository is built upon a number of prior opensource works. 
* Our data generation and testing settings are adapted from https://github.com/xavierpuigf/watch_and_help. 
* The baseline (fine-tuned GPT2 policy) is adapted from https://github.com/ShuangLI59/Pre-Trained-Language-Models-for-Interactive-Decision-Making. Their training code is available in the supplementary materials at https://openreview.net/forum?id=FWMQYjFso-a. 
* GPT3.5 baseline is adapted from https://github.com/huangwl18/language-planner. 
* Our MCTS implementation is adapted from https://github.com/jys5609/MC-LAVE-RL. 
