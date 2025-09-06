"""
Railway TraceAI — SUPER COMPLETE (single-file)
Save as: railway_traceai_super_complete.py
Run: streamlit run railway_traceai_super_complete.py

Requirements (recommended):
pip install streamlit pandas qrcode[pil] opencv-python numpy matplotlib pillow fpdf folium openpyxl scikit-learn
"""

import streamlit as st
import pandas as pd
import qrcode
import cv2
import numpy as np
import hashlib
import random
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
import json
import base64
import os
import zipfile
import pyperclip  # optional: copy to clipboard (pip install pyperclip)
from streamlit.components.v1 import html as st_html
from twilio.rest import Client

# --- Twilio Setup ---
TWILIO_SID = "your_account_sid"
TWILIO_AUTH = "your_auth_token"
TWILIO_PHONE = "+1234567890"   # Twilio number
ADMIN_PHONE = "+91xxxxxxxxxx"  # Your phone (destination)


def send_alert_sms(item_id, expiry_date):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH)
        msg = f"⚠ Alert: Item {item_id} warranty expiring on {expiry_date.date()}"
        client.messages.create(
            body=msg,
            from_=TWILIO_PHONE,
            to=ADMIN_PHONE
        )
        return f"SMS sent: {msg}"
    except Exception as e:
        return f"Error sending SMS: {e}"


# Optional ML
USE_SKLEARN = True
try:
    from sklearn.ensemble import IsolationForest
except Exception:
    USE_SKLEARN = False

# ---------------- Config & Helpers ----------------
st.set_page_config(
    page_title="Railway TraceAI — SUPER COMPLETE", layout="wide")
random.seed(42)
DATA_STORE = "rail_traceai_store.json"

VENDOR_DIR = [
    {"name": "ABC Steel Ltd", "contact": "‪+91-9876543210‬",
        "email": "abc@steel.example", "reliability": 88},
    {"name": "XYZ RailWorks", "contact": "‪+91-9123456789‬",
        "email": "xyz@rail.example", "reliability": 74},
    {"name": "Metro Fittings", "contact": "‪+91-9988776655‬",
        "email": "metro@fit.example", "reliability": 80},
    {"name": "Bharat Track Co.", "contact": "‪+91-9090909090‬",
        "email": "bharat@track.example", "reliability": 92}
]


def save_store(obj):
    try:
        with open(DATA_STORE, "w") as f:
            json.dump(obj, f, default=str)
    except Exception as e:
        st.warning("Could not save store: " + str(e))


def load_store():
    if os.path.exists(DATA_STORE):
        try:
            with open(DATA_STORE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def compute_block(prev_hash, payload_obj):
    payload = json.dumps(payload_obj, sort_keys=True)
    return hashlib.sha256((prev_hash + "|" + payload).encode()).hexdigest()


def create_qr_image(data_obj, img_size=320):
    qr = qrcode.QRCode(box_size=6, border=2)
    qr.add_data(json.dumps(data_obj))
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    img = img.resize((img_size, img_size), Image.NEAREST)
    return img


def df_to_excel_bytes(df_dict):
    with BytesIO() as b:
        with pd.ExcelWriter(b, engine='openpyxl') as writer:
            for name, df in df_dict.items():
                df.to_excel(writer, sheet_name=name, index=False)
        return b.getvalue()


def generate_certificate_image(batch_id, df_summary, exceptions, logo_text="Railway TraceAI"):
    # create a tall image certificate using PIL
    w, h = 1120, 1600
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font_b = ImageFont.truetype("arialbd.ttf", 36)
        font_m = ImageFont.truetype("arial.ttf", 18)
    except:
        font_b = ImageFont.load_default()
        font_m = ImageFont.load_default()
    draw.text((40, 30), "Indian Railways — Inspection Certificate",
              font=font_b, fill=(10, 10, 10))
    draw.text((40, 80), f"Batch ID: {batch_id}",
              font=font_m, fill=(30, 30, 30))
    draw.text(
        (40, 110), f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", font=font_m, fill=(30, 30, 30))
    draw.text(
        (40, 140), f"Total Items: {len(df_summary)}  |  Exceptions: {len(exceptions)}", font=font_m, fill=(30, 30, 30))
    # draw table header sample
    y = 200
    draw.text((40, y), "Top Exceptions (id, vendor, sev, status):",
              font=font_m, fill=(100, 0, 0))
    y += 30
    for i, r in enumerate(exceptions[:8]):
        draw.text(
            (40, y + i*28), f"{r['ItemID']} | {r['Vendor'][:20]} | Sev:{r['Severity']} | {r['Status']}", font=font_m, fill=(0, 0, 0))
    # QR thumbnails
    x = 40
    y = 450
    for i, r in enumerate(exceptions[:6]):
        payload = {"id": r["ItemID"], "lot": r["LotNo"],
                   "mfg": r["MfgDate"], "hash": r["BlockHash"]}
        qrimg = create_qr_image(payload, img_size=240)
        img.paste(qrimg, (x + (i % 3)*260, y + (i//3)*260))
    return img

# ---------------- Data generator ----------------


def simulated_severity(item_type, vendor_rel):
    base = 50 - (vendor_rel - 50) * 0.6
    type_mod = {"Elastic Rail Clip": 0, "Rail Pad": -
                2, "Liner": 5, "Sleeper": -3}.get(item_type, 0)
    noise = random.gauss(0, 12)
    sev = int(min(100, max(0, base + type_mod + noise)))
    return sev


def generate_items(n, seed=42):
    random.seed(seed)
    items = []
    prev_hash = "GENESIS"
    for i in range(n):
        itemid = f"ITM-{100000 + i}"
        itype = random.choice(
            ["Elastic Rail Clip", "Rail Pad", "Liner", "Sleeper"])
        vendor_info = random.choice(VENDOR_DIR)
        vendor = vendor_info["name"]
        vendor_contact = vendor_info["contact"]
        vendor_email = vendor_info["email"]
        vendor_rel = vendor_info["reliability"]
        lot = f"LOT-{1000 + random.randint(0, 999)}"
        mfg_date = (datetime.date.today() -
                    datetime.timedelta(days=random.randint(0, 1000))).isoformat()
        warranty_months = random.choice([12, 24, 36, 60])
        severity = simulated_severity(itype, vendor_rel)
        status = "PASS" if severity < 60 else "FAIL"
        block_payload = {"id": itemid, "lot": lot,
                         "mfg": mfg_date, "status": status, "severity": severity}
        block_hash = compute_block(prev_hash, block_payload)
        prev_hash = block_hash
        items.append({
            "ItemID": itemid,
            "Type": itype,
            "Vendor": vendor,
            "VendorContact": vendor_contact,
            "VendorEmail": vendor_email,
            "VendorReliability": vendor_rel,
            "LotNo": lot,
            "MfgDate": mfg_date,
            "WarrantyMonths": warranty_months,
            "Severity": severity,
            "Status": status,
            "BlockHash": block_hash
        })
    return items, prev_hash


# ---------------- UI ----------------
st.title("🚄 Railway TraceAI — SUPER COMPLETE")

# Sidebar controls
st.sidebar.header("Controls")
num_items = st.sidebar.number_input(
    "Items to generate", min_value=10, max_value=5000, value=350, step=10)
seed = st.sidebar.number_input("Random seed", value=42)
if st.sidebar.button("Generate Dataset (Manufacturer)"):
    items, prev = generate_items(int(num_items), int(seed))
    store = {"items": items, "prev_hash": prev,
             "created": datetime.datetime.now().isoformat()}
    save_store(store)
    st.success(f"Dataset generated: {len(items)} items.")
    st.balloons()

store = load_store()
items = store.get("items", [])

# Top row KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Items in store", len(items))
col2.metric("Last blockhash", store.get("prev_hash", "N/A"))
avg_sev = round(np.mean([it["Severity"] for it in items]) if items else 0, 2)
col3.metric("Average Severity", avg_sev)
col4.metric("Vendors", len(VENDOR_DIR))

# Tabs
tabs = st.tabs(["Overview", "Inventory", "Webcam Scan", "Analytics",
               "Digital Twin", "Reports", "Vendors", "Voice Bot", "Settings"])
tab_overview, tab_inventory, tab_webcam, tab_analytics, tab_digtwin, tab_reports, tab_vendor, tab_voice, tab_settings = tabs

# ---------- Overview ----------
with tab_overview:
    st.header("Overview")
    st.markdown("""
    Railway TraceAI — SUPER COMPLETE
    Features:
    - Generate items & QR PNGs (ZIP with Excel) for laser marking / hardware verification.
    - Upload or webcam-scan laser-marked QR → verify signature & block hash.
    - AI-simulated severity, anomaly detection (z-score + optional IsolationForest).
    - Printable inspection certificate (PDF) with QR thumbnails.
    - Mock WhatsApp/SMS/Email alert composer for vendors & inspectors.
    - Voice assistant (browser TTS) for quick queries.
    - All offline (no external API), single-file demo for judges.
    """)

# ---------- Inventory ----------
with tab_inventory:
    st.header("Inventory & Export")
    if not items:
        st.info("No dataset found. Generate using sidebar.")
    else:
        df = pd.DataFrame(items)
        st.dataframe(df[["ItemID", "Type", "Vendor", "LotNo", "MfgDate",
                     "WarrantyMonths", "Severity", "Status"]].sample(min(200, len(df))))
        if st.button("Export ZIP (Excel + QR PNGs)"):
            excel_bytes = df_to_excel_bytes({"Items": df})
            mem_zip = BytesIO()
            with zipfile.ZipFile(mem_zip, mode="w") as zf:
                zf.writestr("Items.xlsx", excel_bytes)
                for it in items:
                    payload = {"id": it["ItemID"], "lot": it["LotNo"],
                               "mfg": it["MfgDate"], "hash": it["BlockHash"]}
                    img = create_qr_image(payload, img_size=420)
                    bio = BytesIO()
                    img.save(bio, format="PNG")
                    bio.seek(0)
                    zf.writestr(f"qr_images/{it['ItemID']}.png", bio.read())
            mem_zip.seek(0)
            b64 = base64.b64encode(mem_zip.read()).decode()
            href = f'<a href="data:application/zip;base64,{b64}" download="Railway_Items_{datetime.date.today()}.zip">⬇ Download ZIP (Excel + QR PNGs)</a>'
            st.markdown(href, unsafe_allow_html=True)

# ---------- Webcam Scan ----------
with tab_webcam:
    st.header("Webcam Scan (Hardware QR verification)")
    st.write(
        "Use your laptop camera to capture the laser-marked QR (or upload an image).")
    snap = st.camera_input("Point camera at QR and capture")
    if snap:
        # decode
        bytes_data = snap.getvalue()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(img)
        if data:
            st.success("QR decoded")
            try:
                parsed = json.loads(data)
                st.json(parsed)
            except:
                st.write("Raw data:", data)
                parsed = None
            if parsed and "id" in parsed:
                found = [it for it in items if it["ItemID"] == parsed["id"]]
                if found:
                    it = found[0]
                    st.write("Matched item from inventory:")
                    st.write({"ItemID": it["ItemID"], "Vendor": it["Vendor"],
                             "Lot": it["LotNo"], "Mfg": it["MfgDate"], "Status": it["Status"]})
                    st.write("QR BlockHash:", parsed.get("hash"))
                    st.write("DB BlockHash:", it.get("BlockHash"))
                    if parsed.get("hash") == it.get("BlockHash"):
                        st.success("✅ BlockHash matches. Authentic item.")
                    else:
                        st.error("❌ BlockHash mismatch — possible tamper.")
                else:
                    st.warning("Item not present in current inventory.")
        else:
            st.error(
                "QR could not be decoded. Try a clearer image or use the original PNG from ZIP.")

    st.write("---")
    uploaded = st.file_uploader(
        "Or upload QR image (png/jpg)", type=["png", "jpg", "jpeg"])
    if uploaded:
        bytes_data = uploaded.read()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(img)
        if data:
            st.success("QR decoded")
            try:
                parsed = json.loads(data)
                st.json(parsed)
            except:
                st.write("Raw data:", data)

# ---------- Analytics ----------
with tab_analytics:
    st.header("Analytics, Anomaly Detection & Warranty Alerts")
    if not items:
        st.info("Generate dataset first.")
    else:
        df = pd.DataFrame(items)
        df["MfgDate_dt"] = pd.to_datetime(df["MfgDate"])
        df["ExpiryDate"] = df["MfgDate_dt"] + \
            pd.to_timedelta(df["WarrantyMonths"]*30, unit='d')
        df["WarrantyRemainingDays"] = (
            df["ExpiryDate"] - pd.Timestamp.now()).dt.days
        df["RiskLevel"] = df["Severity"].apply(
            lambda s: "High" if s > 70 else ("Medium" if s > 45 else "Low"))
        # anomaly: local z-score
        grouped = df.groupby("LotNo")["Severity"].agg(
            ["mean", "std"]).reset_index()

        def is_anom(row):
            g = grouped[grouped["LotNo"] == row["LotNo"]]
            if g.empty:
                return False
            mean = float(g["mean"])
            std = float(g["std"]) if float(g["std"]) > 0 else 1.0
            z = (row["Severity"] - mean)/std
            return abs(z) > 2.0
        df["Anomaly"] = df.apply(is_anom, axis=1)
        st.metric("Total Items", len(df))
        cols = st.columns(3)
        cols[0].metric("High Risk", int((df["Severity"] > 70).sum()))
        cols[1].metric("Average Severity", round(df["Severity"].mean(), 2))
        cols[2].metric("Anomalies", int(df["Anomaly"].sum()))
        # IsolationForest optional
        if USE_SKLEARN:
            try:
                clf = IsolationForest(contamination=0.02, random_state=42)
                vals = df[["Severity"]].values
                preds = clf.fit_predict(vals)
                df["IF_Anomaly"] = preds == -1
                st.write("IsolationForest anomalies:",
                         int(df["IF_Anomaly"].sum()))
            except Exception as e:
                st.warning("IsolationForest failed: " + str(e))
        # charts
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        df["RiskLevel"].value_counts().reindex(["Low", "Medium", "High"]).plot(
            kind="bar", ax=ax[0], color=["#2ecc71", "#f1c40f", "#e74c3c"])
        ax[0].set_title("Risk Level Distribution")
        df["Vendor"].value_counts().head(8).plot(
            kind="bar", ax=ax[1], color="#3498db")
        ax[1].set_title("Top Vendors (by items)")
        st.pyplot(fig)

        # --- Mini Dashboard: Pie + Failed Items ---
        st.subheader("📊 Mini Dashboard")

        # Pie chart of items by vendor
        vendor_counts = df["Vendor"].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(vendor_counts, labels=vendor_counts.index,
                autopct="%1.1f%%", startangle=90)
        ax2.axis("equal")
        st.pyplot(fig2)

        # Table of failed items
        failed_items = df[df["Severity"] > 60]  # or df[df["Status"]=="FAIL"]
        st.subheader("⚠ Failed Items (auto-flagged)")
        st.dataframe(
            failed_items[["ItemID", "Vendor", "LotNo", "Severity", "Status"]].head(20))
        st.subheader("📦 UDM / TMS Overview")

        # UDM: Items by type and vendor
        udm_summary = df.groupby(["Type", "Vendor"]).agg(
            TotalItems=('ItemID', 'count'),
            AvgSeverity=('Severity', 'mean'),
            FailCount=('Status', lambda x: (x == 'FAIL').sum())
        ).reset_index()
        st.write("UDM Table: Items by Type & Vendor")
        st.dataframe(udm_summary.head(20))

        # TMS: Track status and flow simulation
        st.write("TMS Table: Item Movement / Status")
        # simulate simple workflow steps
        steps = ["Manufactured", "Supplied", "Received",
                 "Installed", "Inspected", "InService"]
        tms_records = []
        for _, it in df.sample(min(50, len(df))).iterrows():
            step_dates = pd.date_range(start=pd.Timestamp(
                it["MfgDate"]), periods=len(steps), freq="30D")
            for s, d in zip(steps, step_dates):
                tms_records.append({"ItemID": it["ItemID"], "Step": s, "Date": d,
                                   "Severity": it["Severity"], "Status": it["Status"]})

        tms_df = pd.DataFrame(tms_records)
        st.dataframe(tms_df.head(30))

        # warranty table
        soon = df[df["WarrantyRemainingDays"] <
                  90].sort_values("WarrantyRemainingDays")
        st.subheader("Expiring Warranties (<90 days)")
        st.dataframe(soon[["ItemID", "Vendor", "ExpiryDate",
                           "WarrantyRemainingDays"]].head(30))
        # --- SMS/WhatsApp Alerts ---
st.subheader("🔔 Alerts for Expiring Warranties")

alert_mode = st.selectbox(
    "Choose alert channel:",
    ["SMS", "WhatsApp", "Email"]
)

expiring_soon = soon[soon["WarrantyRemainingDays"] < 30]

if not expiring_soon.empty:
    st.write(f"Found {len(expiring_soon)} items expiring in <30 days.")

    if st.button("🚀 Send Now"):
        alert_logs = []
        for _, row in expiring_soon.iterrows():
            msg = f"Alert: Item {row['ItemID']} (Vendor: {row['Vendor']}) warranty expires on {row['ExpiryDate'].date()} via {alert_mode}"
            st.success(msg)   # demo output
            alert_logs.append({
                "ItemID": row["ItemID"],
                "Vendor": row["Vendor"],
                "ExpiryDate": row["ExpiryDate"].date(),
                "Channel": alert_mode,
                "Status": "Sent"
            })

        log_df = pd.DataFrame(alert_logs)
        st.subheader("📜 Alert Log")
        st.dataframe(log_df)
else:
    st.success("No urgent warranty alerts 🚀")
   # --- Warranty Expiry Heatmap ---
    st.subheader("📅 Warranty Expiry Heatmap")

    if not df.empty:
        df["ExpiryMonth"] = df["ExpiryDate"].dt.to_period("M").astype(str)
        expiry_counts = df["ExpiryMonth"].value_counts().sort_index()

        expiry_df = pd.DataFrame({
            "ExpiryMonth": expiry_counts.index,
            "Count": expiry_counts.values
        })
        expiry_df["Year"] = expiry_df["ExpiryMonth"].str[:4]
        expiry_df["Month"] = expiry_df["ExpiryMonth"].str[5:7]

        pivot = expiry_df.pivot(
            index="Year", columns="Month", values="Count").fillna(0)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(pivot.values, cmap="YlOrRd")

        # ticks + labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        ax.set_title("Warranty Expiry Distribution")

        # annotate each cell
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, int(pivot.values[i, j]),
                        ha="center", va="center", color="black")

        fig.colorbar(im, ax=ax, shrink=0.7, label="Expiring Items")
        st.pyplot(fig)
    else:
        st.info("No data available for heatmap.")

# ---------- Digital Twin ----------
with tab_digtwin:
    st.header("Digital Twin (Simulation)")
    if not items:
        st.info("Generate data first.")
    else:
        steps = ["Manufactured", "Supplied", "Received", "Installed",
                 "Inspected", "InService", "Repaired/Retired"]
        sample_n = min(80, len(items))
        sample = random.sample(items, sample_n)
        frames = []
        for it in sample:
            t0 = datetime.datetime.now() - datetime.timedelta(days=random.randint(30, 600))
            for s in steps:
                t0 += datetime.timedelta(days=random.randint(7, 90))
                frames.append(
                    {"ItemID": it["ItemID"], "Step": s, "TS": t0, "Severity": it["Severity"]})
        dft = pd.DataFrame(frames)
        st.dataframe(dft.head(150))
        # show simple interactive chart
        st.line_chart(dft.groupby(pd.Grouper(key="TS", freq="30D"))[
                      "Severity"].mean())

   # --- Vendor Geo Map ---
    st.subheader("🌍 Vendor Locations Map")
    try:
        import folium
        from streamlit_folium import st_folium

        # Sample coordinates for vendors
        vendor_locations = {
            "ABC Steel Ltd": [28.6139, 77.2090],       # Delhi
            "XYZ RailWorks": [19.0760, 72.8777],       # Mumbai
            "Metro Fittings": [13.0827, 80.2707],      # Chennai
            "Bharat Track Co.": [22.5726, 88.3639]     # Kolkata
        }

        m = folium.Map(location=[22.5, 80],
                       zoom_start=5, tiles="CartoDB positron")

        for v in VENDOR_DIR:
            name = v["name"]
            coord = vendor_locations.get(
                name, [20.5937, 78.9629])  # fallback: India center
            popup_text = f"""
                <b>{name}</b><br>
                Contact: {v['contact']}<br>
                Email: {v['email']}<br>
                Reliability: {v['reliability']}%
                """
            folium.Marker(
                location=coord,
                popup=popup_text,
                tooltip=f"{name} 📦",
                icon=folium.Icon(color="blue", icon="industry", prefix="fa")
            ).add_to(m)

        st_data = st_folium(m, width=700, height=500)
    except Exception as e:
        st.warning(f"Folium map could not load: {e}")

# ---------- Reports ----------
with tab_reports:
    st.header("Reports: Excel, Certificate, Mock Alerts")
    if not items:
        st.info("Generate dataset first.")
    else:
        df = pd.DataFrame(items)
        summary = df.copy()
        exceptions = df[df["Status"] == "FAIL"].copy()
        col1, col2, col3 = st.columns(3)
        with col1:
            excel_bytes = df_to_excel_bytes(
                {"InspectionSummary": summary, "ExceptionLog": exceptions})
            st.download_button("⬇ Download Excel Summary", data=excel_bytes,
                               file_name=f"InspectionSummary_{datetime.date.today()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with col2:
            batch_id = f"BATCH-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            cert_img = generate_certificate_image(
                batch_id, summary.to_dict('records'), exceptions.to_dict('records'))
            bio = BytesIO()
            cert_img.save(bio, format="PNG")
            bio.seek(0)
            st.image(bio, caption="Generated Certificate Preview",
                     use_column_width=False, width=600)
            # provide PDF download
            pdf_file = f"InspectionCertificate_{batch_id}.pdf"
            # create PDF from PIL image
            cert_img_rgb = cert_img.convert("RGB")
            cert_img_rgb.save(pdf_file, "PDF", resolution=100.0)
            with open(pdf_file, "rb") as f:
                st.download_button("⬇ Download Certificate (PDF)",
                                   f, file_name=pdf_file, mime="application/pdf")
        with col3:
            st.subheader("Mock Alerts (WhatsApp/SMS/Email)")
            sel_type = st.selectbox(
                "Alert channel", ["WhatsApp", "SMS", "Email"])
            sel_vendor = st.selectbox("Select vendor to alert", [
                                      v["name"] for v in VENDOR_DIR])
            message = st.text_area("Compose message (auto-filled below)",
                                   value=f"Alert: Immediate attention required for vendor {sel_vendor}. Some items failed inspection. Please respond.")
            if st.button("Prepare Alert"):
                # mock sending: show message + copy button
                st.success(
                    f"{sel_type} message prepared for {sel_vendor}. (NOT SENT — Mock)")
                st.write("---")
                st.code(message)
                # copy to clipboard via pyperclip or JS
                try:
                    pyperclip.copy(message)
                    st.info(
                        "Message copied to clipboard (pyperclip). Paste to WhatsApp or SMS app.")
                except Exception:
                    # fallback: show JS copy button
                    st.markdown(f"""
                    <button onclick="navigator.clipboard.writeText({json.dumps(message)})">Copy message to clipboard</button>
                    """, unsafe_allow_html=True)

# ---------- Vendors ----------
with tab_vendor:
    st.header("Vendors & Smart Card")
    vdf = pd.DataFrame(VENDOR_DIR)
    st.dataframe(vdf)
    sel = st.selectbox("Select vendor", [v["name"] for v in VENDOR_DIR])
    vinfo = next(filter(lambda x: x["name"] == sel, VENDOR_DIR))
    st.subheader(f"{vinfo['name']} — Smart Card")
    st.write(f"Contact: {vinfo['contact']} • Email: {vinfo['email']}")
    # compute vendor metrics from dataset
    if items:
        df = pd.DataFrame(items)
        vi = df[df["Vendor"] == vinfo["name"]]
        st.metric("Items in dataset", len(vi))
        st.metric("Avg Severity", round(
            vi["Severity"].mean() if len(vi) > 0 else 0, 2))
        st.metric("Reliability (preset)", vinfo["reliability"])
        st.dataframe(vi[["ItemID", "LotNo", "Severity", "Status"]].head(20))

# ---------- Voice Bot ----------
with tab_voice:
    st.header("Voice Assistant (Browser TTS)")
    st.write(
        "Type a command like: 'show high risk items' or 'summary report' and press Speak.")
    cmd = st.text_input("Command", value="show high risk items")
    if st.button("Speak"):
        # produce a spoken response via browser TTS using embedded JS
        # We'll create a simple response text
        resp = ""
        if "high" in cmd.lower():
            resp = "There are {} high risk items detected.".format(
                int((np.array([it["Severity"] for it in items]) > 70).sum()) if items else 0)
        elif "summary" in cmd.lower():
            resp = f"Dataset contains {len(items)} items. Average severity is {avg_sev}."
        else:
            resp = "Command not recognised in demo. Try 'show high risk items' or 'summary report'."
        # embed JS to call speechSynthesis
        safe_text = resp.replace("'", "\\'")
        st_html(f"""
        <script>
          const msg = '{safe_text}';
          const u = new SpeechSynthesisUtterance(msg);
          window.speechSynthesis.speak(u);
        </script>
        """, height=50)
        st.success("Spoken: " + resp)
        st.write(resp)

# ---------- Settings ----------
with tab_settings:
    st.header("Settings & Advanced Options")
    st.write(
        "Optional: enable IsolationForest anomaly detector (scikit-learn required).")
    st.write("If you'd like to load a custom AI model (TFLite/Keras), upload it here as a placeholder (demo only).")
    model_file = st.file_uploader("Upload model file (placeholder)", type=[
                                  "tflite", "h5", "pth", "onnx"])
    if model_file:
        st.success("Model file uploaded (placeholder) — demo app does not execute models here. Replace simulated_severity with real model inference in production.")

st.markdown("---")
st.markdown("Railway TraceAI — SUPER COMPLETE — demo ready. For production, replace simulated AI with trained models, store blockchain hashes on immutable ledger, and integrate secure UDM/TMS APIs.")
