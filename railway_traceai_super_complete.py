import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import joblib
import datetime

# -----------------------------
# SAMPLE DATA GENERATION
# -----------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    vendors = ["Vendor A", "Vendor B", "Vendor C"]
    parts = ["Rail Pad", "Fastener", "Clip", "Bolt"]

    data = pd.DataFrame({
        "PartID": range(1, 51),
        "Vendor": np.random.choice(vendors, 50),
        "PartName": np.random.choice(parts, 50),
        "Severity": np.random.randint(10, 100, 50),
        "WarrantyPeriod": np.random.randint(1, 5, 50),
        "DeliveryDate": pd.date_range("2023-01-01", periods=50, freq="15D"),
    })
    data["ExpiryDate"] = data["DeliveryDate"] + pd.to_timedelta(data["WarrantyPeriod"]*365, unit="D")
    return data

data = load_data()

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.set_page_config(page_title="Railway TraceAI", layout="wide")
st.title("🚄 Railway TraceAI – AI-powered QR + Anomaly Detection")

# Tabs
tabs = st.tabs(["Dashboard", "QR Scanner Simulation", "Analytics", "Reports"])

# -----------------------------
# DASHBOARD TAB
# -----------------------------
with tabs[0]:
    st.header("📊 System Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Parts", len(data))
    col2.metric("Vendors", data['Vendor'].nunique())
    col3.metric("High Severity Issues", (data['Severity'] > 70).sum())

    st.subheader("Recent Deliveries")
    st.dataframe(data.head(10))

# -----------------------------
# QR SCANNER TAB
# -----------------------------
with tabs[1]:
    st.header("📷 QR Scanner Simulation")
    part_id = st.number_input("Enter/Scan Part ID", min_value=1, max_value=len(data), value=1, step=1)

    part_info = data[data['PartID'] == part_id].iloc[0]
    st.success(f"Scanned Part: {part_info['PartName']} from {part_info['Vendor']}")
    st.json({
        "PartID": int(part_info["PartID"]),
        "Vendor": part_info["Vendor"],
        "PartName": part_info["PartName"],
        "Severity": int(part_info["Severity"]),
        "WarrantyPeriod": int(part_info["WarrantyPeriod"]),
        "DeliveryDate": str(part_info["DeliveryDate"].date()),
        "ExpiryDate": str(part_info["ExpiryDate"].date())
    })

# -----------------------------
# ANALYTICS TAB
# -----------------------------
with tabs[2]:
    st.header("📈 Analytics & Anomaly Detection")

    # Rule-based alerts
    st.subheader("⚠️ High Severity Parts")
    high_risk = data[data['Severity'] > 70]
    st.write(high_risk)

    st.subheader("⌛ Warranty Expiry Alerts")
    expiring = data[pd.to_datetime(data['ExpiryDate']) < pd.Timestamp.now()]
    st.write(expiring)

    # Isolation Forest ML
    st.subheader("🤖 ML-based Anomaly Detection")
    features = data[["Severity", "WarrantyPeriod"]]
    model = IsolationForest(contamination=0.1, random_state=42)
    preds = model.fit_predict(features)
    data["Anomaly"] = preds
    anomalies = data[data["Anomaly"] == -1]
    st.write(anomalies)

    fig, ax = plt.subplots()
    ax.scatter(data["Severity"], data["WarrantyPeriod"], c=(data["Anomaly"]==-1), cmap="coolwarm")
    ax.set_xlabel("Severity")
    ax.set_ylabel("WarrantyPeriod")
    st.pyplot(fig)

# -----------------------------
# REPORTS TAB
# -----------------------------
with tabs[3]:
    st.header("📑 Generate Reports")
    report_type = st.selectbox("Choose Report", ["Summary", "Vendor-wise", "Warranty Expiry"])
    if report_type == "Summary":
        st.write(data.describe())
    elif report_type == "Vendor-wise":
        st.bar_chart(data.groupby("Vendor")["Severity"].mean())
    else:
        st.write(expiring)

    if st.button("Export Report as CSV"):
        data.to_csv("railway_report.csv", index=False)
        st.success("✅ Report saved as railway_report.csv")

# -----------------------------
# END
# -----------------------------

