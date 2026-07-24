import sqlite3

import pandas as pd
import streamlit as st


DATABASE_PATH = "sentinelai.db"


def load_scan_data() -> pd.DataFrame:
    """Load all stored email scan results from SQLite."""

    connection = sqlite3.connect(DATABASE_PATH)

    try:
        query = """
        SELECT
            id,
            sender,
            subject,
            risk_score,
            risk_level,
            urls_found,
            urgency_risk,
            scanned_at
        FROM email_scans
        ORDER BY scanned_at DESC
        """

        return pd.read_sql_query(query, connection)

    finally:
        connection.close()


st.set_page_config(
    page_title="SentinelAI Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SentinelAI")
st.subheader("Email Threat Intelligence Dashboard")

try:
    scans = load_scan_data()

except Exception as error:
    st.error(f"Unable to load database: {error}")
    st.stop()


if scans.empty:
    st.warning("No email scan records are available yet.")
    st.stop()


# -----------------------------
# Dashboard summary calculations
# -----------------------------

total_emails = len(scans)

low_risk = len(
    scans[scans["risk_level"] == "LOW RISK"]
)

medium_risk = len(
    scans[scans["risk_level"] == "MEDIUM RISK"]
)

high_risk = len(
    scans[scans["risk_level"] == "HIGH RISK"]
)

average_risk = scans["risk_score"].mean()


# -----------------------------
# Summary cards
# -----------------------------

column1, column2, column3, column4, column5 = st.columns(5)

column1.metric("Emails Scanned", total_emails)
column2.metric("Low Risk", low_risk)
column3.metric("Medium Risk", medium_risk)
column4.metric("High Risk", high_risk)
column5.metric("Average Risk", f"{average_risk:.1f}/100")


# -----------------------------
# Risk-level chart
# -----------------------------

st.subheader("Risk-Level Distribution")

risk_counts = (
    scans["risk_level"]
    .value_counts()
    .rename_axis("Risk Level")
    .reset_index(name="Emails")
)

st.bar_chart(
    risk_counts,
    x="Risk Level",
    y="Emails"
)


# -----------------------------
# Recent threats
# -----------------------------

st.subheader("Recent Email Scans")

display_columns = [
    "sender",
    "subject",
    "risk_score",
    "risk_level",
    "urls_found",
    "urgency_risk",
    "scanned_at"
]

st.dataframe(
    scans[display_columns],
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# High-risk section
# -----------------------------

st.subheader("Suspicious Emails")

suspicious_emails = scans[
    scans["risk_level"].isin(
        ["MEDIUM RISK", "HIGH RISK"]
    )
]

if suspicious_emails.empty:
    st.success("No medium-risk or high-risk emails detected.")

else:
    st.dataframe(
        suspicious_emails[display_columns],
        use_container_width=True,
        hide_index=True
    )