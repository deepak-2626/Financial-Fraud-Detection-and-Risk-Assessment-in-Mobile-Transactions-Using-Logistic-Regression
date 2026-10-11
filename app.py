"""
app.py
------
Streamlit app: Financial Fraud Detection + Risk Assessment
Uses a trained Logistic Regression model (model.pkl) and metadata (metadata.json).

Run:
    streamlit run app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Fraud Detection & Risk Assessment",
    page_icon="🔍",
    layout="centered",
)

# ── Load model + metadata ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "model.pkl"))

@st.cache_data
def load_metadata():
    with open(os.path.join(BASE_DIR, "metadata.json")) as f:
        return json.load(f)

try:
    model    = load_model()
    metadata = load_metadata()
except FileNotFoundError:
    st.error("⚠️ Model not found. Please run `python train_model.py` first.")
    st.stop()

valid_types  = metadata["valid_types"]
type_risk    = metadata["type_risk"]
amount_stats = metadata["amount_stats"]
step_max     = metadata["step_max"]
amount_max   = metadata["amount_max"]
eval_metrics = metadata["eval"]


# ══════════════════════════════════════════════════════════════════════════════
# RISK ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_risk_score(
    step: int,
    amount: float,
    txn_type: str,
    old_balance_orig: float,
    new_balance_orig: float,
    old_balance_dest: float,
    new_balance_dest: float,
) -> dict:
    """
    Rule-based risk engine producing a 0–100 composite risk score.

    Four sub-scores:
      1. Transaction Type Risk  — TRANSFER/CASH_OUT have highest fraud rates
      2. Amount Anomaly         — deviation from type's typical amount
      3. Balance Wipe Signal    — sender balance drops to ~0 (classic fraud pattern)
      4. Time Risk              — late-night / early-morning transactions
    """
    scores = {}

    # 1. Transaction type risk (0–30 points)
    tier = type_risk.get(txn_type, "medium")
    type_score = {"low": 5, "medium": 15, "high": 30}.get(tier, 15)
    scores["Transaction Type Risk"] = type_score

    # 2. Amount anomaly (0–30 points)
    stats     = amount_stats.get(txn_type, {})
    mean_amt  = stats.get("mean", 0)
    std_amt   = stats.get("std", 1) or 1
    z         = abs(amount - mean_amt) / std_amt
    amt_score = min(30, round((z / 3) * 30, 1))
    scores["Amount Anomaly"] = amt_score

    # 3. Balance wipe signal (0–25 points)
    # In real fraud, the origin account balance is zeroed out
    balance_wipe = 0
    if old_balance_orig > 0:
        drain_ratio = (old_balance_orig - new_balance_orig) / old_balance_orig
        balance_wipe = min(25, round(drain_ratio * 25, 1))
    scores["Balance Drain Signal"] = balance_wipe

    # 4. Time risk (0–15 points) — step % 24 gives hour of day
    hour = step % 24
    if hour <= 2 or hour >= 22:
        time_score = 15
    elif hour <= 5 or hour >= 19:
        time_score = 9
    else:
        time_score = 3
    scores["Time of Transaction"] = time_score

    composite = min(100, round(sum(scores.values()), 1))
    return {"composite": composite, "breakdown": scores, "type_tier": tier}


def risk_label(score: float) -> tuple[str, str]:
    if score < 30:
        return "🟢 Low Risk", "#2ecc71"
    elif score < 60:
        return "🟡 Medium Risk", "#f39c12"
    else:
        return "🔴 High Risk", "#e74c3c"


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("🔍 Financial Fraud Detection & Risk Assessment")
st.markdown(
    "Enter a transaction's details below. The system will predict whether it "
    "is **fraudulent** and produce a **risk assessment score** based on "
    "transaction type, amount anomaly, balance patterns, and timing."
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("transaction_form"):
    st.subheader("Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        txn_type = st.selectbox(
            "Transaction Type",
            options=valid_types,
            help="TRANSFER and CASH_OUT have the highest fraud rates historically",
        )
        amount = st.number_input(
            "Amount",
            min_value=0.01,
            max_value=float(amount_max) * 5,
            value=500.00,
            step=0.01,
            format="%.2f",
        )
        step = st.slider(
            "Time Step (hour in 30-day window)",
            min_value=0,
            max_value=min(step_max, 743),
            value=12,
            help="1 step = 1 hour. Steps 0–743 cover a 30-day simulation.",
        )

    with col2:
        old_balance_orig = st.number_input(
            "Sender: Balance Before",
            min_value=0.0,
            value=10_000.0,
            step=100.0,
            format="%.2f",
            help="Sender's account balance before this transaction",
        )
        new_balance_orig = st.number_input(
            "Sender: Balance After",
            min_value=0.0,
            value=9_500.0,
            step=100.0,
            format="%.2f",
            help="Sender's account balance after this transaction",
        )
        old_balance_dest = st.number_input(
            "Recipient: Balance Before",
            min_value=0.0,
            value=0.0,
            step=100.0,
            format="%.2f",
        )
        new_balance_dest = st.number_input(
            "Recipient: Balance After",
            min_value=0.0,
            value=500.0,
            step=100.0,
            format="%.2f",
        )

    submitted = st.form_submit_button("🔎 Analyse Transaction", use_container_width=True)


# ── Results ───────────────────────────────────────────────────────────────────
if submitted:
    # Engineered features (must match train_model.py)
    balance_diff_orig = new_balance_orig - old_balance_orig + amount
    balance_diff_dest = new_balance_dest - old_balance_dest - amount

    input_df = pd.DataFrame([{
        "step":            step,
        "amount":          amount,
        "oldbalanceOrg":   old_balance_orig,
        "newbalanceOrig":  new_balance_orig,
        "oldbalanceDest":  old_balance_dest,
        "newbalanceDest":  new_balance_dest,
        "balanceDiffOrig": balance_diff_orig,
        "balanceDiffDest": balance_diff_dest,
        "type":            txn_type,
    }])

    # ── ML Prediction ─────────────────────────────────────────────────────────
    fraud_proba = model.predict_proba(input_df)[0][1]
    is_fraud    = fraud_proba >= 0.5

    # ── Risk Engine ───────────────────────────────────────────────────────────
    risk = compute_risk_score(
        step, amount, txn_type,
        old_balance_orig, new_balance_orig,
        old_balance_dest, new_balance_dest,
    )
    risk_score       = risk["composite"]
    risk_lbl, risk_color = risk_label(risk_score)

    st.divider()

    # ── Section 1: ML Fraud Prediction ────────────────────────────────────────
    st.subheader("🤖 Fraud Detection Result")

    pred_col1, pred_col2 = st.columns(2)
    with pred_col1:
        if is_fraud:
            st.error("**⚠️ FRAUDULENT TRANSACTION**")
        else:
            st.success("**✅ LEGITIMATE TRANSACTION**")
    with pred_col2:
        st.metric("Model Confidence", f"{fraud_proba * 100:.1f}%",
                  help="Logistic Regression probability that this transaction is fraud")

    # Gauge chart
    bar_color = "#e74c3c" if is_fraud else "#2ecc71"
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(fraud_proba * 100, 1),
        title={"text": "Fraud Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": bar_color},
            "steps": [
                {"range": [0, 50],   "color": "#d5f5e3"},
                {"range": [50, 75],  "color": "#fdebd0"},
                {"range": [75, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
        number={"suffix": "%"},
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # ── Section 2: Risk Assessment ────────────────────────────────────────────
    st.subheader("📊 Risk Assessment")

    risk_col1, risk_col2 = st.columns([1, 2])

    with risk_col1:
        st.markdown(
            f"""
            <div style="
                background-color: {risk_color}22;
                border: 2px solid {risk_color};
                border-radius: 12px;
                padding: 20px;
                text-align: center;
            ">
                <div style="font-size: 2.5rem; font-weight: bold; color: {risk_color};">
                    {risk_score}<span style="font-size:1rem;">/100</span>
                </div>
                <div style="font-size: 1.1rem; color: {risk_color}; margin-top: 6px;">
                    {risk_lbl}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with risk_col2:
        breakdown = risk["breakdown"]
        fig_bar = go.Figure(go.Bar(
            x=list(breakdown.values()),
            y=list(breakdown.keys()),
            orientation="h",
            marker_color=[
                "#e74c3c" if v >= 20 else "#f39c12" if v >= 10 else "#2ecc71"
                for v in breakdown.values()
            ],
            text=[f"{v:.0f} pts" for v in breakdown.values()],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Risk Score Breakdown",
            xaxis=dict(range=[0, 40], title="Points"),
            yaxis=dict(title=""),
            height=240,
            margin=dict(t=40, b=20, l=10, r=60),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Risk explanation
    with st.expander("📖 How is the risk score calculated?"):
        tier_label = risk["type_tier"]
        drain = ""
        if old_balance_orig > 0:
            drain_pct = ((old_balance_orig - new_balance_orig) / old_balance_orig) * 100
            drain = f"{drain_pct:.1f}% of sender balance was drained"
        else:
            drain = "sender had no prior balance"

        st.markdown(f"""
        The **Risk Score (0–100)** combines four independent signals:

        | Component | Max | This Transaction |
        |---|---|---|
        | **Transaction Type Risk** | 30 | {breakdown['Transaction Type Risk']:.0f} pts — `{txn_type}` is **{tier_label} risk** |
        | **Amount Anomaly** | 30 | {breakdown['Amount Anomaly']:.0f} pts — amount vs typical `{txn_type}` transactions |
        | **Balance Drain Signal** | 25 | {breakdown['Balance Drain Signal']:.0f} pts — {drain} |
        | **Time of Transaction** | 15 | {breakdown['Time of Transaction']:.0f} pts — hour **{step % 24:02d}:00** |

        **Total: {risk_score}/100 → {risk_lbl}**

        > The ML model and the risk score are **independent**. A transaction can be
        > predicted as legitimate but carry a high risk score — for example, a large
        > TRANSFER at 2 AM that drains the sender's account warrants scrutiny even if
        > the model is not fully confident.
        """)

    st.divider()

    # ── Section 3: Model Performance ──────────────────────────────────────────
    with st.expander("🧠 Model Performance"):
        report = eval_metrics["report"]
        cm     = eval_metrics["confusion_matrix"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC",           f"{eval_metrics['roc_auc']:.4f}")
        m2.metric("Precision (Fraud)",  f"{report['Fraud']['precision']:.3f}")
        m3.metric("Recall (Fraud)",     f"{report['Fraud']['recall']:.3f}")
        m4.metric("F1 (Fraud)",         f"{report['Fraud']['f1-score']:.3f}")

        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted Legit", "Predicted Fraud"],
            y=["Actual Legit", "Actual Fraud"],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        ))
        fig_cm.update_layout(
            title="Confusion Matrix (Test Set)",
            height=300,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        st.caption(
            "Model: Logistic Regression with `class_weight='balanced'`. "
            "Features: `step`, `amount`, balance fields, engineered balance diffs, `type`. "
            "No demographic features (age/gender) are used."
        )
