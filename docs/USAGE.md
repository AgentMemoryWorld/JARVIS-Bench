# Usage guide

[← Back to the project overview](../README.md)

All commands below run from the repository root. This guide covers dataset access and the evaluation workflow.

Code for the NeurIPS 2026 (Evaluations & Datasets track) paper
**JARVIS-Bench: Benchmarking Personal Intelligence Agents on Long-Horizon Real-User Daily Traces**
([OpenReview](https://openreview.net/forum?id=yGQt9wAGkS)). The dataset is available on
[Hugging Face](https://huggingface.co/datasets/WZDavid/JARVIS-Bench).

A benchmark for evaluating how well LLMs can model and predict an individual
user's lifestyle preferences, proactive-help needs, and emotional state from
their daily activity logs and images.

The benchmark covers four tasks, with 100 questions for preference reasoning:

| Task                | Variants                  | Inputs                                  | Metric                                            |
|---------------------|---------------------------|-----------------------------------------|---------------------------------------------------|
| **Survey**          | C0 to C4                     | trajectory ± persona ± survey-insight   | accuracy (4-way MCQ)                              |
| **Proactive help**  | text, image               | prior-day history + (optional) anchor image   | precision / recall / F1, judge-based reason match |
| **Emotion**         | text, image               | event context + (optional) image        | polarity acc, mood F1                             |
| **Domain Transfer** | S1 drop-one, S2 keep-only | trajectory with one domain removed or retained | Δ accuracy vs C2_full (necessity) / vs C0_blank (sufficiency) |

## Repository layout

```
JARVIS-Bench/
├── README.md                # visual project overview
├── docs/USAGE.md            # this guide
├── assets/                  # paper figures and project logo
├── LICENSE                  # MIT
├── .gitignore               # excludes user data + caches + secrets
├── JarvisBench_problem.csv  # 100-question survey (public benchmark items)
├── ground_truth/            # survey answer key (copied to Data/processed/survey.json)
├── tools/prepare_data.py    # imports prepared local User/Image files into the evaluation layout
├── eval/                    # evaluation pipeline (see eval/README.md)
│   ├── README.md
│   ├── configs/             # conditions.yaml, models.yaml, domain_mapping.yaml
│   ├── prompts/system.txt
│   └── src/                 # python package: `python -m eval.src.cli ...`
├── Data/                    # ground-truth + persona + images   (NOT in git)
└── synthetic/               # per-user synthetic trajectories   (NOT in git)
```

The `Data/` and `synthetic/` directories are **not** committed to this
repository. They contain user-derived data and must be obtained separately
(see "Obtaining the data" below).

## Obtaining the data

Load the dataset from [Hugging Face](https://huggingface.co/datasets/WZDavid/JARVIS-Bench),
under CC BY-NC 4.0:

```bash
pip install datasets
```

```python
from datasets import load_dataset

events = load_dataset("WZDavid/JARVIS-Bench", "events", split="full")
days = load_dataset("WZDavid/JARVIS-Bench", "days", split="full")
profiles = load_dataset("WZDavid/JARVIS-Bench", "profiles", split="full")
```

The tables contain event records, day metadata, and user profiles. JSON-suffixed
columns preserve the corresponding nested records; see the dataset card for
schema and reconstruction details. Pin a Hub revision when reporting results.

### Preparing evaluation inputs

The evaluation CLI reads per-user JSON files in the layout below. It does not
load the Hub tables directly. `tools/prepare_data.py` imports an already prepared
local `User/` directory or `User.zip`, plus `Image/` or `Image.zip` when needed;
it does not convert Hugging Face Parquet tables.

For an existing local directory with those inputs, run:

```bash
python tools/prepare_data.py /path/to/prepared-data --no-images
# Omit --no-images when the prepared directory also contains Image/ or Image.zip.
```

The script imports the files and copies `ground_truth/survey_ground_truth.json`
to `Data/processed/survey.json`. The evaluation layout is:

```
JARVIS-Bench/
├── Data/
│   └── processed/
│       ├── survey.json            # ground-truth survey answers, keyed by user_slug
│       ├── persona.json           # optional; no released condition reads it
│       └── image/
│           └── {user_slug}/
│               └── Day{N}/        # one folder per day, JPG/PNG photos inside
│                   ├── img_001.jpg
│                   └── ...
└── synthetic/
    └── {user_slug}/               # one folder per user
        ├── synthetic_trajectory.json   # REQUIRED: per-day event log
        ├── profile_persona.json        # required for C3 / C4 only
        └── skeleton.json               # required by aggregator for clue-slice metrics
```

`{user_slug}` is a stable 4-character ID (e.g., `0cxl`, `z8a2`). Image and
synthetic folders must use the same slug; the loader cross-references them.

### File-by-file requirements

| Path                                                | Used by                               | Required when                              |
|-----------------------------------------------------|---------------------------------------|--------------------------------------------|
| `JarvisBench_problem.csv`                           | survey loader                          | always                                     |
| `Data/processed/survey.json`                        | survey ground-truth                    | always                                     |
| `Data/processed/persona.json`                       | survey loader (persona injection)      | optional; no released condition uses it    |
| `Data/processed/image/{slug}/Day{N}/*.jpg`          | proactive and emotion eligibility      | text and image variants; see the shared-pool note below                   |
| `synthetic/{slug}/synthetic_trajectory.json`        | all tasks                              | per-user, always                           |
| `synthetic/{slug}/profile_persona.json`             | survey conditions C3, C4               | required for C3 and C4; otherwise optional                         |
| `synthetic/{slug}/skeleton.json`                    | survey aggregator (planted / clue slices) | required for clue-slice metrics; otherwise optional                      |

**Shared-pool note.** The current proactive and emotion implementations use a common eligible sample pool for their text and image variants. Images must be present on disk even for these text variants; only image variants attach them to the model call. Image files are not included in the Hugging Face dataset. Emotion targets are selected from the last five logged days; proactive targets span qualifying days across the trajectory. Model context includes prior-day history. Emotion context also includes earlier events on the target day. The `--no-images` preparation option is suitable for survey-only runs. `skeleton.json` is also absent from the Hub dataset; it is required for planted-clue slice metrics, but not overall survey accuracy.

Anything else inside `synthetic/{slug}/` (e.g. `chunks/`, `run_logs/`,
`qa_report.json`, `skeleton_windows/`) is intermediate synthesis output and
is **not** required to run the evaluation. Do not commit those even after
obtaining the data bundle.

## Quick start

This evaluation workflow assumes the prepared local inputs described above.
For direct dataset access, use the Hugging Face loading example instead.

```bash
# 1. Install dependencies
pip install openai pyyaml pandas
# Optional, only if you use these providers:
pip install anthropic google-genai google-cloud-aiplatform
pip install boto3                  # Amazon Bedrock models

# 2. Export API keys for the providers you want to call
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...      # optional
export GEMINI_API_KEY=...          # optional
# Amazon Bedrock models use your AWS credentials (aws configure, AWS_PROFILE, or SSO)

# 3. Import prepared local inputs for a survey run, then sanity-check
python tools/prepare_data.py /path/to/prepared-data --no-images
python -m eval.src.cli list-users
python -m eval.src.cli list-config

# 4. Smoke test (single user, single model)
python -m eval.src.cli run --task survey \
    --users z8a2 --models gpt-5-nano

# 5. With the required images also prepared, run all task variants for one user
python -m eval.src.cli run --task survey          --users z8a2 --models gpt-5-nano
python -m eval.src.cli run --task proactive_text  --users z8a2 --models gpt-5-nano
python -m eval.src.cli run --task proactive_image --users z8a2 --models gpt-5-nano
python -m eval.src.cli run --task emotion_text    --users z8a2 --models gpt-5-nano
python -m eval.src.cli run --task emotion_image   --users z8a2 --models gpt-5-nano

# 6. Domain Transfer (Task 4): re-run the survey while ablating one domain.
#    S1 "drop-one" measures necessity:    Δ accuracy vs C2_full baseline.
#    S2 "keep-only" measures sufficiency: Δ accuracy vs C0_blank baseline.
python -m eval.src.cli run --task survey --conditions C2_full \
    --users z8a2 --models gpt-5-nano \
    --ablations drop_food_dining drop_work_study drop_leisure
python -m eval.src.cli run --task survey --conditions C0_blank \
    --users z8a2 --models gpt-5-nano \
    --ablations keep_food_dining keep_work_study keep_leisure

# 7. (Re)aggregate result CSVs from cached runs. For proactive runs this also
#    calls the rationale-match judge (OpenAI gpt-5-mini by default, needs
#    OPENAI_API_KEY); add --no-judge to skip it or --judge-model-id to change it.
python -m eval.src.cli aggregate --task all
```

### Models

`eval/configs/models.yaml` has one entry per model and serving route, keyed
by the model name used in the paper. Gemini models also have a Google AI
Studio route with an `-api` suffix, which is the route the paper reports, and
a few screened models that are not in the paper (`step-3.5-flash`,
`hy3-preview`, `gemini-3.1-pro-preview`) are kept. Entries with `provider: "bedrock"` are called
through the Amazon Bedrock Converse API (region `us-west-2`) without a
temperature, with a 16k output cap and the reasoning setting listed under
`additional_request_fields` (effort `low` where the model exposes it).
For example:

```bash
python -m eval.src.cli run --task survey --users z8a2 --models gpt-6-astra claude-opus-4.5
```

Without `--models`, `run` calls every configured model, so always pass the
models you want. Bedrock needs model access for each provider enabled in
`us-west-2` (Bedrock console, Model access); an `AccessDeniedException` means
the account has no access to that model.

### Memory conditions

| Condition | User context |
| :--- | :--- |
| `C0_blank` | No user-specific context |
| `C1_traj14d` | First 14 days of trajectory |
| `C2_full` | Full trajectory |
| `C3_full_persona` | Full trajectory and enhanced profile persona, without survey insight |
| `C4_skyline` | Full trajectory and profile persona including survey insight |

C4 is an **oracle ceiling**, not an ordinary memory-only condition. See [`conditions.yaml`](../eval/configs/conditions.yaml) for the exact inputs.

### Domain Transfer details

Domain Transfer reuses the survey runner with an `--ablations` flag that
filters which event domains appear in the trajectory before it is serialized
into the prompt. The 7 buckets (defined in `eval/configs/domain_mapping.yaml`)
are: `daily_routine`, `food_dining`, `work_study`, `leisure`,
`commuting_travel`, `shopping`, `social_wellbeing`. Long-tail event domains
not in any bucket are always kept (treated as bucket-agnostic noise).

Two complementary ablation families:

- **S1: drop-one** (`drop_<bucket>`): start from `C2_full`, remove all
  events in one bucket, measure the accuracy drop. Large drop ⇒ this bucket
  is *necessary*.
- **S2: keep-only** (`keep_<bucket>`): start from `C0_blank`, add back
  only events in one bucket, measure the accuracy gain. Large gain ⇒ this
  bucket is *sufficient* on its own.

Full sweep is 7 buckets × 2 families × 42 users × N models. Runs land in
`eval/runs/{condition}__{model}__{ablation}/` and are picked up automatically
by `aggregate --task survey`.

Outputs land in `eval/runs/` (git-ignored) and are organized as
`{condition}__{model}/parsed/{user}.json` for survey runs and
`{task}__{user}__{model}/` for proactive / emotion runs. Cross-(user, model)
summary CSVs are written to `eval/runs/_summary/`.

See the [evaluation reference](../eval/README.md) for the full CLI reference, condition definitions,
caching behavior, and parallelism controls (`EVAL_PARALLEL` env var).

## Citation

If you use JARVIS-Bench in your research, please cite the paper:

```bibtex
@inproceedings{zhang2026jarvisbench,
  title     = {{JARVIS-Bench}: Benchmarking Personal Intelligence Agents on Long-Horizon Real-User Daily Traces},
  author    = {Zhang, Weizhi and
               Huang, Wei-Chieh and
               Liang, Yueqing and
               Jiang, Liwei and
               Wang, Zhengxiang and
               Yang, Liangwei and
               Li, Zechen and
               Wu, Yuchen and
               Zhang, Haozhen and
               Wang, Yu and
               Bei, Yuanchen and
               Zhou, Yue and
               Zhu, Siqi and
               Zou, Henry Peng and
               Liu, Jiahong and
               Zhang, Xinni and
               Martin, Paul and
               Imperial, Joseph Marvin and
               Luo, Yuyang and
               Xu, Xiongxiao and
               Huang, Baixiang and
               Wu, Shanglin and
               Resck, Lucas and
               Wang, Yibo and
               Liu, Yuqing and
               He, Langzhou and
               Li, Chengze and
               Tian, Yuxin and
               Zheng, Kening and
               Ma, Enze and
               Huynh, Boi and
               Hui, Zheng and
               Yang, Rui and
               Feng, Tao and
               Yang, Jie and
               Li, Shanghao and
               Wang, Haoran and
               Yang, Wooseong and
               Zhang, Hanrong and
               Ma, Huanhuan and
               Yoon, Daye and
               Wu, Yaozu and
               Lian, Shikan and
               Ruan, Xiaoqian and
               Wang, Fangxin and
               Park, Hyeonjeong and
               Hu, Hins and
               Fan, Wenzhe and
               Wang, Chen and
               Wang, Yifan and
               Huang, Jintao and
               Wang, Yijun and
               Wang, Haixin and
               Wu, Yating and
               Qian, Cheng and
               Gu, Yifan and
               Li, Dongyuan and
               Wang, Song and
               Caliskan, Aylin and
               Wang, Jindong and
               Luo, Xiao and
               Chen, Yankai and
               Shu, Kai and
               Liu, Xue and
               Hao, Yuexing and
               Li, Xiaomin and
               Yu, Philip S.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Track on Evaluations and Datasets},
  year      = {2026},
  url       = {https://openreview.net/forum?id=yGQt9wAGkS}
}
```

## License

Code is licensed under [MIT](../LICENSE). The separately distributed data is licensed under [CC BY-NC 4.0](https://huggingface.co/datasets/WZDavid/JARVIS-Bench).
