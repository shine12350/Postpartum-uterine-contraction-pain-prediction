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

Total vaginal deliveries include the current delivery and are encoded as:
- 1: reference category (both dummy variables are 0)
- 2: `total_vaginal_deliveries_2 = 1`
- 3 or more: `total_vaginal_deliveries_ge3 = 1`

Unknown binary responses and blank numeric inputs are passed as missing values and handled
by the original fitted imputation steps.

## Intended use and privacy
The model estimates the probability of concurrent moderate-to-severe uterine contraction pain
(VAS >= 4) at baseline, using data collected within 24 hours after vaginal delivery and before
the first randomized intervention. Predictions are for reference only and must not be used for
clinical decision-making, diagnosis, or treatment recommendations.

The authors do not collect or retain information entered into the application, and user inputs
are not used for model retraining. The app is hosted on a third-party cloud platform, which may
process standard technical logs or metadata under its own privacy policy. Do not enter personally
identifiable information.
