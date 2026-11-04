# JARVIS-Bench Survey Ground Truth

42 benchmark users (those with `synthetic/{slug}/synthetic_trajectory.json`) x 100 questions.
Each user's own 4-way MCQ answer (A/B/C/D) is the ground truth for that user.

- `survey_ground_truth.json` — `{user_slug: {language, answers: {"1": "A", ..., "100": "B"}}}` (same schema as `Data/processed/survey.json`)
- `survey_ground_truth_wide.csv` — one row per user, columns Q1..Q100
- `survey_ground_truth_long.csv` — one row per (user, question), with domain, question text, answer letter and answer text
- `JarvisBench_problem.csv` — the 100 questions and options

Source: `Data/processed/survey.json` (47 respondents; 5 without trajectories — 9csz, gwln, m6xj, wt7o, zqvd — are excluded here).
