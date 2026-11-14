# Postpartum Uterine Contraction Pain Prediction APP

## Files
- `app.py`: English Streamlit interface.
- `requirements.txt`: pinned runtime dependencies.
- `locked_primary_elasticnet_uncalibrated.joblib`: original locked, uncalibrated Elastic Net Pipeline for VAS >= 4.
- `model_metadata.json`: model settings and predictor-column metadata.

## Run on Windows
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Run on macOS or Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Model input design
The APP collects six clinical inputs. It automatically constructs the original 18-column
input DataFrame and calls the saved, fitted preprocessing Pipeline and Elastic Net model.
The 11 non-selected predictors are not requested from the user. They remain missing in
the constructed DataFrame and are passed through the original fitted preprocessing steps.
No preprocessing step or model parameter is refitted by the APP.

Previous vaginal deliveries are encoded as:
- 0 or 1: reference category
- 2: `total_vaginal_deliveries_2 = 1`
- 3 or more: `total_vaginal_deliveries_ge3 = 1`

Unknown binary responses and blank numeric inputs are passed as missing values and handled
by the original fitted imputation steps.

## Important notice
This tool is intended for research and supportive assessment only. It must not be used
for clinical decision-making. Do not enter personally identifiable information.
