# import streamlit as st
# import requests

# API_URL = "https://lard-coconut-willow.ngrok-free.dev"

# st.title("AI Resume Improver")

# resume = st.text_area("Paste your resume", height=250)
# job_desc = st.text_area("Paste job description (optional)", height=150)

# if not resume.strip():
#     st.warning("Please enter a resume before continuing.")
#     st.stop()

# if st.button("Improve Resume"):
#     with st.spinner("Improving your resume..."):
#         response = requests.post(
#             f"{API_URL}/improve",
#             json={"resume": resume, "job_desc": job_desc},
#             timeout=60
#         )

#         data = response.json()

#         st.markdown("## Revised Resume")
#         st.markdown(data["revised_resume"])

#         st.markdown("## What Changed & Why")

#         for change in data.get("changes", []):
#             st.markdown(f"- {change}")


# if st.button("Generate Cover Letter"):
#     with st.spinner("Generating cover letter..."):
#         try:
#             response = requests.post(
#                 f"{API_URL}/cover",
#                 json={"resume": resume, "job_desc": job_desc},
#                 timeout=60
#             )

#             data = response.json()

#             if response.status_code == 200:
#                 st.markdown("## Cover Letter")
#                 st.markdown(data.get("result", "No result returned"))
#             else:
#                 st.error(f"API Error: {response.text}")

#         except Exception as e:
#             st.error(f"Request failed: {str(e)}")
import streamlit as st
import requests

# -----------------------
# CONFIG
# -----------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
} if HF_TOKEN else {}

# -----------------------
# QUERY FUNCTION
# -----------------------
def query_hf(prompt):
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.3,
                    "return_full_text": False
                }
            },
            timeout=60
        )

        if response.status_code != 200:
            return f"Error from Hugging Face:\n{response.text}"

        data = response.json()

        if isinstance(data, list):
            return data[0].get("generated_text", "")
        elif isinstance(data, dict):
            if "generated_text" in data:
                return data["generated_text"]
            if "error" in data:
                return f"HF Error: {data['error']}"

        return str(data)

    except Exception as e:
        return f"Request failed: {str(e)}"

# -----------------------
# PROMPT FUNCTION
# -----------------------
def improve_resume(text, job_desc=""):
    prompt = f"""
You are a professional resume editor.

You are given ONLY a section of a resume.

OUTPUT FORMAT:

REVISED SECTION:
<improved version>

CHANGES:
- change + why
- change + why
- change + why

RULES:
- KEEP the same structure as input
- DO NOT add new sections
- ONLY improve wording and clarity
- DO NOT invent information

INPUT:
{text}

JOB DESCRIPTION:
{job_desc}
"""
    return query_hf(prompt)

# -----------------------
# UI
# -----------------------
st.title("AI Resume Improver")

if not HF_TOKEN:
    st.warning("⚠️ Hugging Face token not found. Add HF_TOKEN in Streamlit secrets.")

resume = st.text_area("Paste your resume section", height=250)
job_desc = st.text_area("Paste job description (optional)", height=150)

if st.button("Improve Resume"):
    if not resume.strip():
        st.warning("Please enter a resume section.")
        st.stop()

    with st.spinner("Improving your resume..."):
        result = improve_resume(resume, job_desc)

        if result.startswith("Error") or result.startswith("HF Error") or result.startswith("Request failed"):
            st.error(result)
            st.stop()

        if "CHANGES:" in result:
            section_part, changes_part = result.split("CHANGES:", 1)
        else:
            section_part = result
            changes_part = "No changes provided"

        section_part = section_part.replace("REVISED SECTION:", "").strip()

        st.markdown("## Revised Section")
        st.markdown(section_part)

        st.markdown("## What Changed & Why")
        for line in changes_part.strip().split("\n"):
            if line.strip():
                st.markdown(line)