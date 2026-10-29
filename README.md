# HR Policy ML Explorer — Beginner Project

## Purpose
Explore 10 machine-learning techniques with the supplied 10,000-row Quality of Life dataset. The FastAPI website has a model selector and an employee input form. It returns an interpretation and shows held-out test scores.

## Windows / VS Code setup
1. Extract the ZIP. Open the **inner `hr_policy_ml_project` folder** containing `main.py` in VS Code (File > Open Folder).
2. Open Terminal > New Terminal.
3. Install Python packages: `py -m pip install -r requirements.txt`
4. Start website: `py -m uvicorn main:app --reload`
5. Visit http://127.0.0.1:8000 and API docs http://127.0.0.1:8000/docs
6. To retrain models: stop the server, run `py train.py`, and restart the server.

If `py` is unavailable, use `python` instead.

## Business questions and algorithms
| Algorithm | What it demonstrates | Target definition |
|---|---|---|
| Linear Regression | Work hours vs historical age at death | age_at_death |
| Multiple Regression | Lifestyle and occupation vs historical age at death | age_at_death |
| Polynomial Regression | Nonlinear lifestyle relationships | age_at_death |
| Lasso | Regularized lifestyle relationships | age_at_death |
| Logistic Regression | Flag long workdays | work hours >= 9.5 |
| Random Forest | Flag workload with limited rest | work >= 9 AND rest < 6 |
| Decision Tree | Flag short sleep | sleep < 7 |
| Naive Bayes | Flag low activity | exercise < 1 |
| SVM | Classify workday as short/regular/extended | work < 8, 8–<10, >=10 |
| TF-IDF + Naive Bayes | Classify generated schedule descriptions | work >= 10 |

**CRITICAL HONESTY NOTE:** The six classification targets are rule-generated using the *same inputs* given to the models. Scores near 100% mean the models can recover straightforward schedule rules, **NOT** that they predict real burnout, health risk, retention, or HR policy success. TF-IDF text is generated from occupation, gender and a work-hour bucket; its near-perfect score is particularly expected. A real HR outcome dataset (employee satisfaction, overtime policy compliance, absenteeism, etc.) would be required to validate genuine business predictions. These flags are descriptive and should not be used for employee punishment, hiring, or other consequential decisions. Historical age-at-death models are correlations, not personal life-expectancy predictions.

## Model evaluation
- Fixed random 80% training / 20% held-out testing, seed 42.
- Numeric missing-value handling, scaling, and categorical one-hot encoding are fitted on the training split via sklearn Pipelines. No ID column used.
- Classification: accuracy, weighted precision, weighted recall, weighted F1, plus majority-class baseline accuracy/precision.
- Regression: MAE, R², plus training-mean baseline MAE/R².
- Results are saved in `models/results.json` and printed in `training_results.txt`.
- No test-set tuning was performed. The rule thresholds are illustrative business-policy choices, **not scientifically validated risk cutoffs**.
- The hour fields almost sum to 24; using them to predict each other would produce a misleadingly trivial regression. We deliberately did not do that.

## Files
- `train.py`: trains all 10 models and computes test scores.
- `main.py`: FastAPI app and JSON endpoints.
- `static/index.html`: frontend form.
- `static/style.css`: page appearance.
- `static/script.js`: sends form data to API and displays results.
- `quality_of_life.csv`: your supplied data.
- `models/*.joblib`: saved trained models.
- `models/results.json`: actual test metrics.

## Endpoints
- `GET /` website
- `GET /api/models` model descriptions and scores
- `POST /api/predict/{model_name}` predictions. JSON keys: `gender`, `occupation_type`, `avg_work_hours_per_day`, `avg_rest_hours_per_day`, `avg_sleep_hours_per_day`, `avg_exercise_hours_per_day`.

Example PowerShell test (while server is running):
```powershell
$body = '{"gender":"Female","occupation_type":"Teacher","avg_work_hours_per_day":10,"avg_rest_hours_per_day":5,"avg_sleep_hours_per_day":6,"avg_exercise_hours_per_day":0.5}'
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/predict/decision_tree -Method Post -ContentType 'application/json' -Body $body
```

## Limitations
No ground-truth HR risk labels exist. The demo cannot measure causal policy effects. Scores for different classification tasks are not directly comparable because the target rules differ. Some hours may not be realistic in isolation; users should enter plausible schedules. For a genuine HR risk model, collect consented, privacy-protected outcomes and evaluate fairness and calibration.
