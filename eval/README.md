# JarvisBench Eval

Survey-prediction evaluation pipeline. Given a user's daily activity log (and optionally profile/persona), an LLM is asked to predict the user's answers to a 100-question lifestyle survey. We compare against the user's real survey answers as ground truth.

## Layout

```
eval/
├── configs/
│   ├── conditions.yaml       # C0..C4 input conditions
│   └── models.yaml           # model_id, pricing, sampling params
├── prompts/
│   └── system.txt
├── src/                      # python package, run as `python -m eval.src.cli`
│   ├── loader.py
│   ├── serializer.py         # trajectory/persona -> text
│   ├── prompt_builder.py
│   ├── llm_client.py         # OpenAI client + disk cache (eval/runs/.cache/)
│   ├── parser.py             # robust JSON / regex letter parser
│   ├── aggregate.py          # writes _summary CSVs
│   └── cli.py
└── runs/                     # auto-created (git-ignored)
    ├── .cache/               # API response cache (hashed prompts)
    ├── {condition}__{model}/ # per (condition,model) outputs
    │   ├── config.snapshot.json
    │   ├── responses/{user}.json   # raw model output + tokens/cost
    │   └── parsed/{user}.json      # {q_no: letter}
    └── _summary/             # cross-(condition,model) CSVs
        ├── overall__{per_user,average,matrix}.csv
        ├── by_domain__{per_user,average}.csv
        ├── by_interaction__{per_user,average}.csv
        ├── by_clue_difficulty__{per_user,average}.csv
        ├── by_clue_type__{per_user,average}.csv
        └── by_planted__{per_user,average}.csv
```

## Conditions

| ID | Trajectory | Persona | survey_insight |
|---|---|---|---|
| C0_blank          | -          | -   | -   |
| C1_traj14d        | first 14 d | -   | -   |
| C2_full           | all days   | -   | -   |
| C3_full_persona   | all days   | yes | -   |
| C4_skyline        | all days   | yes | yes |

## Setup

```bash
pip install openai pyyaml pandas
export OPENAI_API_KEY=sk-...
pip install boto3   # only for provider "bedrock" (uses your AWS credentials)
```

`eval/configs/models.yaml` lists every evaluated model; keys are the names used
in the paper and `model_id` is the identifier sent to the provider.

## Commands

```bash
# from the JARVIS-Bench project root
cd /path/to/JARVIS-Bench

# inspect what's configured
python -m eval.src.cli list-users
python -m eval.src.cli list-config

# DRY RUN: builds prompts only, prints sizes (no API calls, no $)
python -m eval.src.cli run --dry-run

# small smoke test (1 user, 1 condition, 1 model)
python -m eval.src.cli run --users z8a2 --conditions C0_blank --models gpt-5-nano

# 2 default users x 5 conditions x the listed models
python -m eval.src.cli run --models gpt-5-nano gpt-5-mini gpt-5

# regenerate _summary CSVs from existing runs (no API)
python -m eval.src.cli aggregate
```

## Caching

Every API call is keyed by `sha256(model_id || system || user || params)` and
stored at `eval/runs/.cache/{ab}/{hash}.json`. Re-running the same configuration
hits the cache → free, instant, idempotent. Editing prompts/conditions changes
the hash → cache miss → re-call.

To force a re-call, delete the relevant cache file (or `rm -rf eval/runs/.cache/`).
