import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="ShieldDoc", page_icon="🔒", layout="wide")
st.title("🔒 ShieldDoc — PII Scanner")
st.markdown("Upload a document to scan for personally identifiable information.")

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload a file",
    type=["pdf", "png", "jpg", "jpeg", "txt"]
)

if uploaded_file is not None:
    # Figure out which endpoint to hit based on file type
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext in ["png", "jpg", "jpeg"]:
        endpoint = "/scan/image"
    elif ext == "pdf":
        endpoint = "/scan/pdf"
    else:
        endpoint = "/scan/text"

    if st.button("Scan for PII"):
        with st.spinner("Scanning..."):
            response = requests.post(
                f"{API_URL}{endpoint}",
                files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            )

        if response.status_code == 200:
            data = response.json()

            # Layout: two columns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📄 Extracted Text")
                st.text_area("Extracted", data["extracted_text"], height=300, key="extracted")

            with col2:
                st.subheader("🔴 Redacted Text")
                st.text_area("Redacted", data["redacted_text"], height=300, key="redacted")
            # PII Findings Table
            st.subheader("🔍 PII Found")
            if data["pii_found"]:
                st.error(f"Found {len(data['pii_found'])} PII instance(s)")
                
                # Build a clean table
                table_data = [
                    {
                        "Type": m["entity_type"],
                        "Value": m["value"],
                        "Position": f"{m['start']}–{m['end']}",
                        "Confidence": m["confidence"]
                    }
                    for m in data["pii_found"]
                ]
                st.table(table_data)
            else:
                st.success("No PII detected!")

        else:
            st.error(f"API error: {response.status_code} — {response.text}")