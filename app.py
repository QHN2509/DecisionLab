"""DecisionLab Streamlit research dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from decisionlab.app.prediction import (
    CORRELATION_LABELS,
    SHAPE_LABELS,
    format_gamble_rows,
    load_prediction_bundle,
    predict_scenario,
    what_if_predictions,
)
from decisionlab.features.behavioral import ScenarioFeatureInput

APP_CONFIG = Path(__file__).resolve().parent / "configs" / "app.json"


@st.cache_resource
def _load_bundle():
    return load_prediction_bundle()


@st.cache_data
def _load_app_config():
    return json.loads(APP_CONFIG.read_text(encoding="utf-8"))


def _scenario_inputs(config) -> ScenarioFeatureInput:
    defaults = config["default_scenario"]
    limits = config["input_limits"]
    st.sidebar.header("Construct a decision problem")
    st.sidebar.caption("Payoffs use the same abstract units as choices13k.")
    st.sidebar.subheader("Gamble A")
    high_a = st.sidebar.number_input(
        "High payoff A",
        min_value=limits["high_payoff_a"][0],
        max_value=limits["high_payoff_a"][1],
        value=defaults["high_payoff_a"],
        step=1.0,
    )
    probability_a = st.sidebar.slider(
        "Probability of high payoff A",
        limits["probability"][0],
        limits["probability"][1],
        defaults["high_probability_a"],
        0.01,
    )
    low_a = st.sidebar.number_input(
        "Low payoff A",
        min_value=limits["low_payoff_a"][0],
        max_value=limits["low_payoff_a"][1],
        value=defaults["low_payoff_a"],
        step=1.0,
    )

    st.sidebar.subheader("Gamble B")
    mean_b = st.sidebar.number_input(
        "B sublottery mean payoff",
        min_value=limits["sublottery_mean_b"][0],
        max_value=limits["sublottery_mean_b"][1],
        value=defaults["sublottery_mean_b"],
        step=1.0,
    )
    probability_b = st.sidebar.slider(
        "Probability of entering B sublottery",
        limits["probability"][0],
        limits["probability"][1],
        defaults["sublottery_probability_b"],
        0.01,
    )
    low_b = st.sidebar.number_input(
        "B low-branch payoff",
        min_value=limits["low_payoff_b"][0],
        max_value=limits["low_payoff_b"][1],
        value=defaults["low_payoff_b"],
        step=1.0,
    )
    shape_label = st.sidebar.selectbox(
        "B sublottery shape",
        list(SHAPE_LABELS.values()),
        index=defaults["lottery_shape_b"],
    )
    shape = next(value for value, label in SHAPE_LABELS.items() if label == shape_label)
    if shape == 0:
        outcome_count = 1
        st.sidebar.caption("A single-outcome sublottery has no skewness category.")
    else:
        outcome_count = st.sidebar.slider(
            "B sublottery outcomes",
            limits["lottery_outcomes_b"][0],
            limits["lottery_outcomes_b"][1],
            max(defaults["lottery_outcomes_b"], 2),
        )

    st.sidebar.subheader("Experimental conditions")
    feedback = st.sidebar.toggle("Outcome feedback", value=defaults["feedback"])
    ambiguity = st.sidebar.toggle(
        "B probabilities hidden from participants", value=defaults["ambiguity"]
    )
    correlation_label = st.sidebar.selectbox(
        "Payoff correlation category",
        list(CORRELATION_LABELS.values()),
        index=list(CORRELATION_LABELS).index(defaults["correlation"]),
    )
    correlation = next(
        value for value, label in CORRELATION_LABELS.items() if label == correlation_label
    )
    return ScenarioFeatureInput(
        high_payoff_a=high_a,
        high_probability_a=probability_a,
        low_payoff_a=low_a,
        sublottery_mean_b=mean_b,
        sublottery_probability_b=probability_b,
        low_payoff_b=low_b,
        lottery_shape_b=shape,
        lottery_outcomes_b=outcome_count,
        ambiguity=ambiguity,
        feedback=feedback,
        correlation=correlation,
    )


def _render_gambles(result) -> None:
    st.subheader("Decision problem")
    gamble_a_column, gamble_b_column = st.columns(2)
    with gamble_a_column:
        st.markdown("#### Gamble A")
        st.dataframe(
            format_gamble_rows(result.gamble_a),
            hide_index=True,
            width="stretch",
            column_config={
                "Probability": st.column_config.NumberColumn(format="%.3f"),
                "Payoff": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    with gamble_b_column:
        st.markdown("#### Gamble B")
        st.dataframe(
            format_gamble_rows(result.gamble_b),
            hide_index=True,
            width="stretch",
            column_config={
                "Probability": st.column_config.NumberColumn(format="%.3f"),
                "Payoff": st.column_config.NumberColumn(format="%.2f"),
            },
        )


def _render_prediction(result, bundle) -> None:
    st.subheader("Expected value and predicted aggregate behavior")
    ev_a, ev_b, prediction = st.columns(3)
    ev_a.metric("Expected value — A", f"{result.expected_value_a:.2f}")
    ev_b.metric("Expected value — B", f"{result.expected_value_b:.2f}")
    prediction.metric("Predicted aggregate B-choice rate", f"{result.predicted_b_rate:.1%}")
    st.markdown(
        f"**Simple expected-value benchmark:** {result.expected_value_benchmark} "
        f"(EV difference B − A: `{result.expected_value_difference:+.2f}`)."
    )
    st.write(
        f"The model predicts approximately **{result.predicted_b_rate:.1%} choosing B** and "
        f"**{result.predicted_a_rate:.1%} choosing A** in aggregate."
    )
    if result.outside_training_range:
        st.warning(
            "This scenario creates engineered values outside the observed training range: "
            + ", ".join(f"`{name}`" for name in result.outside_training_range)
            + ". Treat the prediction as extrapolation."
        )
    st.caption(
        f"Reference validation performance: MAE {bundle.validation_mae:.3f}, RMSE "
        f"{bundle.validation_rmse:.3f}. These are dataset-level errors, not a calibrated "
        "interval for this scenario."
    )


def _render_drivers(result) -> None:
    st.subheader("Important model drivers")
    st.caption(
        "Domains are ordered by held-out permutation importance. They describe predictive "
        "reliance, not causal effects or a local attribution decomposition."
    )
    rows = [
        {
            "Behavioral domain": row["behavioral_domain"],
            "Validation MAE increase": row["validation_mae_increase"],
            "Current scenario": row["scenario_value"],
        }
        for row in result.driver_rows
    ]
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={"Validation MAE increase": st.column_config.NumberColumn(format="%.4f")},
    )


def _render_what_if(scenario, bundle) -> None:
    st.subheader("What-if comparison")
    st.caption(
        "Each row changes only the displayed experimental condition and its engineered "
        "interaction. This is model sensitivity, not a causal feedback or ambiguity effect."
    )
    rows = what_if_predictions(scenario, bundle)
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "predicted_b_rate": st.column_config.NumberColumn(
                "Predicted B-choice rate", format="%.1%"
            ),
            "change_from_current": st.column_config.NumberColumn(
                "Change from current", format="%+.1%"
            ),
            "feedback": "Feedback",
            "ambiguity": "Ambiguity",
            "scenario": "Scenario",
        },
    )


def main() -> None:
    st.set_page_config(page_title="DecisionLab", page_icon="⚖️", layout="wide")
    st.title("DecisionLab")
    st.markdown(
        "Interactive prediction of **aggregate human risky-choice behavior** from the structure "
        "of a decision problem. Predictions are not personalized recommendations."
    )
    try:
        bundle = _load_bundle()
        scenario = _scenario_inputs(_load_app_config())
        result = predict_scenario(scenario, bundle)
    except (FileNotFoundError, KeyError, ValueError) as error:
        st.error(f"DecisionLab could not evaluate this scenario: {error}")
        st.stop()

    _render_gambles(result)
    _render_prediction(result, bundle)
    _render_drivers(result)
    _render_what_if(scenario, bundle)

    st.subheader("Scope and limitations")
    st.markdown(
        "\n".join(
            [
                "- The prediction target is `bRate`, an aggregate rate of choosing Gamble B—not "
                "an individual choice.",
                "- The model learned associations in choices13k; it does not establish why "
                "people choose an option.",
                "- Expected-value maximization is a simple comparison benchmark, not a "
                "definition of rationality.",
                "- Under ambiguity, the model uses analyst-known design probabilities that "
                "participants did not see. The ambiguity what-if therefore remains an "
                "oracle/design analysis.",
                "- Feedback is entangled with block and experience in the source study. "
                "What-if changes are model sensitivity checks, not treatment-effect estimates.",
                "- Inputs far from the training distribution can be unreliable even when "
                "individual feature ranges appear valid.",
            ]
        )
    )
    st.caption(
        f"Selected model: {bundle.selected_model.replace('_', ' ')} · "
        f"choices13k source commit {bundle.source_commit[:12]} · locked test set not evaluated"
    )


if __name__ == "__main__":
    main()
