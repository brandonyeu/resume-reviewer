import streamlit as st
import requests

API_URL = "https://lard-coconut-willow.ngrok-free.dev"

st.title("AI Resume Improver")

resume = st.text_area("Paste your resume")
job_desc = st.text_area("Paste job description (optional)")

if st.button("Improve Resume"):
    response = requests.post(
        f"{API_URL}/improve",
        json={"resume": resume, "job_desc": job_desc}
    )
    st.write(response.json()["result"])

if st.button("Generate Cover Letter"):
    response = requests.post(
        f"{API_URL}/cover",
        json={"resume": resume, "job_desc": job_desc}
    )
    st.write(response.json()["result"])