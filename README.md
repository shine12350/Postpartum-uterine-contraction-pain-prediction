# Postpartum Uterine Contraction Pain Prediction APP

## Files
- `app.py`: English-language Streamlit interface.
- `requirements.txt`: pinned runtime dependencies.
- `locked_primary_elasticnet_uncalibrated.joblib`: original locked, uncalibrated Elastic Net model and fitted preprocessing pipeline for VAS >= 4.
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
The APP requests six clinical predictors retained by the final Elastic Net model:
- Total vaginal deliveries (including the current delivery)
- Dysmenorrhea
- Intrapartum oxytocin use
- Newborn weight
- Duration of delivery
- Episiotomy

Total vaginal deliveries is represented by two indicator variables (2 and >=3, with 1 as the reference category). Thus, the six clinical predictors correspond to seven model input columns.

The APP reconstructs the original 18-column input DataFrame. The remaining 11 columns are set to missing to preserve compatibility with the original fitted preprocessing pipeline. Their coefficients are zero in the final Elastic Net model and therefore they do not contribute to the predicted probabilities. The original fitted model and preprocessing pipeline are used without refitting.

For unavailable requested predictors, missing continuous values are imputed using the median and missing categorical values (including responses marked "Unknown") are imputed using the most frequent category learned from the development dataset. Blank numeric inputs and "Unknown" categorical responses are passed to the original fitted imputation steps.

## Intended use and privacy
The model estimates the probability of concurrent moderate-to-severe uterine contraction pain (VAS >= 4) at baseline, using data collected within 24 hours after vaginal delivery and before the first randomized intervention. Predictions are for reference only and must not be used for clinical decision-making, diagnosis, or treatment recommendations.

On first access, users are shown a privacy and intended-use statement and must acknowledge it before proceeding. The statement advises users not to enter personally identifiable information. Full study eligibility criteria are available in an expandable section.

The authors do not collect or retain information entered into the application, and user inputs are not used for model retraining. The app is hosted on a third-party cloud platform, which may process standard technical logs or metadata under its own privacy policy.
