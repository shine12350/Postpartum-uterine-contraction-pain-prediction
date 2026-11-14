# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "locked_primary_elasticnet_uncalibrated.joblib"
REQUIRED_MODEL_COLUMNS = [
    "age",
    "bmi",
    "temp",
    "hr",
    "sbp",
    "dbp",
    "gestational_age",
    "abortion1",
    "abortion2",
    "total_vaginal_deliveries_2",
    "total_vaginal_deliveries_ge3",
    "newborn_weight",
    "duration_of_delivery",
    "perineal_laceration",
    "episiotomy",
    "dysmenorrhea",
    "sym_back_pain",
    "intrapartum_oxytocin_use",
]
APP_SKLEARN_VERSION = "1.7.2"


st.set_page_config(
    page_title="Postpartum Uterine Contraction Pain Prediction",
    page_icon="🩺",
    layout="centered",
)


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The locked model file is missing. Keep the model file in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)


def binary_value(selection: str) -> float:
    if selection == "Yes":
        return 1.0
    if selection == "No":
        return 0.0
    return np.nan


def parse_optional_positive_number(raw_value: str, field_label: str):
    value = str(raw_value).strip()
    if value == "":
        return None, None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, f"{field_label} must be a valid number."
    if not math.isfinite(parsed) or parsed <= 0:
        return None, f"{field_label} must be greater than zero."
    return float(parsed), None


def build_model_input(bundle, deliveries, dysmenorrhea, oxytocin, newborn_weight, duration, episiotomy):
    predictor_columns = list(bundle["predictor_columns"])
    missing_columns = [
        column for column in REQUIRED_MODEL_COLUMNS
        if column not in predictor_columns
    ]
    if missing_columns or len(predictor_columns) != 18:
        raise ValueError(
            "The locked model does not contain the expected 18-predictor structure. "
            f"Missing columns: {missing_columns}"
        )

    # Initialize all original 18 columns. The 11 non-selected predictors remain
    # missing by design; their fitted preprocessing steps remain inside the original Pipeline.
    row = {column: np.nan for column in predictor_columns}

    row["newborn_weight"] = (
        np.nan if newborn_weight is None else float(newborn_weight)
    )
    row["duration_of_delivery"] = (
        np.nan if duration is None else float(duration)
    )
    row["episiotomy"] = binary_value(episiotomy)
    row["dysmenorrhea"] = binary_value(dysmenorrhea)
    row["intrapartum_oxytocin_use"] = binary_value(oxytocin)

    # One total vaginal delivery (including the current delivery) is the model reference category.
    row["total_vaginal_deliveries_2"] = float(deliveries == 2)
    row["total_vaginal_deliveries_ge3"] = float(deliveries >= 3)

    return pd.DataFrame([row], columns=predictor_columns)


def show_privacy_dialog():
    @st.dialog("Privacy Statement and Intended Use")
    def _privacy_dialog():
        st.write(
            "The model was developed to estimate the probability of moderate-to-severe "
            "uterine contraction pain (VAS ≥4) at the time of baseline assessment, using "
            "data collected within 24 hours after vaginal delivery and before the first "
            "randomized intervention."
        )
        st.write(
            "The development cohort consisted of adults aged 18 years or older with "
            "singleton live births at 37–42 weeks of gestation following vaginal delivery."
        )
        st.write(
            "Predictions are for reference only and should be interpreted with caution. "
            "They must not be used for clinical decision-making, diagnosis, or treatment recommendations."
        )
        st.write(
            "The authors do not collect or retain information entered into this application, "
            "and user inputs are not used for model retraining. This application is hosted on "
            "a third-party cloud platform, which may process standard technical logs or metadata "
            "in accordance with its own privacy policy."
        )
        st.write(
            "Do not enter real names, medical record numbers, or any other "
            "personally identifiable information."
        )
        with st.expander("View Full Study Eligibility Criteria"):
            st.markdown("**Inclusion criteria**")
            st.markdown(
                "1. Age ≥18 years.  \n"
                "2. Gestational age of 37–42 weeks.  \n"
                "3. Singleton live birth.  \n"
                "4. Vaginal delivery.  \n"
                "5. Intention to breastfeed.  \n"
                "6. Ability to understand the study procedures, comply with the intervention "
                "and follow-up assessments, and provide written informed consent."
            )
            st.markdown("**Exclusion criteria**")
            st.markdown(
                "1. Instrumental vaginal delivery, including vacuum extraction or forceps delivery.  \n"
                "2. Severe pregnancy-related or puerperal complications.  \n"
                "3. Pre-existing pelvic pathologies, such as adenomyosis or pelvic inflammatory disease.  \n"
                "4. Significant central nervous system disorders, such as epilepsy.  \n"
                "5. Rheumatic immune diseases, coagulation defects, or other serious medical and surgical conditions.  \n"
                "6. Psychiatric conditions, such as schizophrenia or mania, that could impair communication and comprehension.  \n"
                "7. A history of chronic analgesic dependency.  \n"
                "8. Concurrent enrollment in other clinical trials.  \n"
                "9. Intervention-related contraindications, including hypersensitivity to acupuncture or NSAIDs, "
                "local skin lesions at needling sites, and active gastrointestinal conditions such as peptic ulcers or bleeding."
            )
        if st.button(
            "I have read and agree",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["privacy_accepted"] = True
            st.rerun()

    _privacy_dialog()


def main():
    st.title("Postpartum Uterine Contraction Pain Prediction")
    st.caption(
        "A research tool for estimating the probability of moderate-to-severe uterine contraction pain (VAS ≥4) following vaginal delivery, based on baseline assessment conducted within 24 hours after delivery and before the first randomized intervention."
    )

    if sklearn.__version__ != APP_SKLEARN_VERSION:
        st.error(
            f"This model package requires scikit-learn {APP_SKLEARN_VERSION}. "
            f"The current environment has scikit-learn {sklearn.__version__}. "
            "Please install the pinned dependencies from requirements.txt."
        )
        st.stop()

    try:
        bundle = load_model_bundle()
    except Exception as exc:
        st.error(f"Unable to load the locked model: {exc}")
        st.stop()

    if bundle.get("model_name") != "ElasticNet" or float(bundle.get("vas_cutoff", -1)) != 4.0:
        st.error(
            "The APP requires the locked, uncalibrated Elastic Net model for VAS ≥ 4."
        )
        st.stop()

    if "privacy_accepted" not in st.session_state:
        st.session_state["privacy_accepted"] = False

    if not st.session_state["privacy_accepted"]:
        show_privacy_dialog()
        st.info("Please review and accept the privacy statement to continue.")
        st.stop()

    st.subheader("Clinical Information")
    st.caption(
        "Enter the six requested items. Select Unknown when a yes/no item is unavailable. "
        "Leave a numeric field blank if the value is unavailable."
    )

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            deliveries = st.selectbox(
                "Total vaginal deliveries (including current delivery)",
                options=["Please select"] + list(range(1, 11)),
            )
            st.caption("Enter an integer from 1 to 10.")
            dysmenorrhea = st.selectbox(
                "Dysmenorrhea",
                options=["Please select", "Yes", "No", "Unknown"],
            )
            oxytocin = st.selectbox(
                "Intrapartum oxytocin use",
                options=["Please select", "Yes", "No", "Unknown"],
            )
            st.caption(
                "Refers to oxytocin administered during labor, not routine postpartum oxytocin. "
                "Please check the delivery record or medication orders. Select 'Unknown' if the "
                "information is unavailable."
            )

        with right:
            newborn_weight_text = st.text_input(
                "Newborn weight (kg)",
                value="",
                placeholder="e.g., 3.25",
            )
            st.caption("Example: 3.25 kg. Please refer to the birth record.")
            duration_text = st.text_input(
                "Duration of delivery (min)",
                value="",
                placeholder="e.g., 360",
            )
            st.caption("Example: 360 min. Please refer to the delivery record.")
            episiotomy = st.selectbox(
                "Episiotomy",
                options=["Please select", "Yes", "No", "Unknown"],
            )
            st.caption(
                "Please check the delivery record to confirm. Do not confuse a spontaneous "
                "perineal tear with an episiotomy. Select 'Unknown' if the information is unavailable."
            )

        submitted = st.form_submit_button(
            "Generate prediction",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    required_choices = {
        "Total vaginal deliveries (including current delivery)": deliveries,
        "Dysmenorrhea": dysmenorrhea,
        "Intrapartum oxytocin use": oxytocin,
        "Episiotomy": episiotomy,
    }
    incomplete = [
        name for name, value in required_choices.items()
        if value == "Please select"
    ]
    if incomplete:
        st.error("Please complete the following fields: " + ", ".join(incomplete) + ".")
        return

    newborn_weight, newborn_error = parse_optional_positive_number(
        newborn_weight_text, "Newborn weight"
    )
    duration, duration_error = parse_optional_positive_number(
        duration_text, "Duration of delivery"
    )
    errors = [error for error in [newborn_error, duration_error] if error]
    if errors:
        for error in errors:
            st.error(error)
        return

    missing_inputs = []
    if newborn_weight is None:
        missing_inputs.append("Newborn weight")
    if duration is None:
        missing_inputs.append("Duration of delivery")
    if dysmenorrhea == "Unknown":
        missing_inputs.append("Dysmenorrhea")
    if oxytocin == "Unknown":
        missing_inputs.append("Intrapartum oxytocin use")
    if episiotomy == "Unknown":
        missing_inputs.append("Episiotomy")

    try:
        X_app = build_model_input(
            bundle=bundle,
            deliveries=int(deliveries),
            dysmenorrhea=dysmenorrhea,
            oxytocin=oxytocin,
            newborn_weight=newborn_weight,
            duration=duration,
            episiotomy=episiotomy,
        )
        probability = float(bundle["estimator"].predict_proba(X_app)[0, 1])
        threshold = float(bundle["locked_youden_threshold"])
    except Exception as exc:
        st.error(f"Prediction could not be generated: {exc}")
        return

    st.divider()
    st.subheader("Prediction Result")
    st.metric("Predicted probability of VAS ≥ 4", f"{probability * 100:.1f}%")
    st.caption(
        f"Classification threshold: {threshold:.3f} "
        "(development-set Youden threshold)"
    )

    if probability >= threshold:
        st.success("Model-predicted category: VAS ≥ 4")
    else:
        st.info("Model-predicted category: VAS < 4")

    st.warning(
        "The APP requests six clinical predictors retained by the final Elastic Net model. "
        "Missing or unknown values are imputed using statistics learned from the development dataset."
    )

    st.caption(
        "This output is a model-based research classification, not a diagnosis, "
        "clinical risk assessment, or treatment recommendation."
    )


if __name__ == "__main__":
    main()
