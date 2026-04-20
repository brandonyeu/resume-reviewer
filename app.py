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
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

API_URL = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
} if GROQ_API_KEY else {}

# -----------------------
# QUERY FUNCTION
# -----------------------
def query_groq(messages):
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "model": "llama3-8b-8192",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 500
            },
            timeout=30
        )

        if response.status_code != 200:
            return f"Error from Groq:\n{response.text}"

        data = response.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Request failed: {str(e)}"

# -----------------------
# RESUME FUNCTION
# -----------------------
def improve_resume(text, job_desc=""):
    messages = [
        {
            "role": "system",
            "content": "You are a professional resume editor."
        },
        {
            "role": "user",
            "content": f"""
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
- DO NOT add new sections (no Skills, no Summary)
- ONLY improve wording, clarity, and impact
- DO NOT invent information
- Keep bullet points if present

INPUT:
{text}

JOB DESCRIPTION:
{job_desc}
"""
        }
    ]

    return query_groq(messages)

# -----------------------
# UI
# -----------------------
st.title("AI Resume Improver")

if not GROQ_API_KEY:
    st.warning("⚠️ Add GROQ_API_KEY to Streamlit secrets to use the app.")

resume = st.text_area("Paste your resume section", height=250)
job_desc = st.text_area("Paste job description (optional)", height=150)

if st.button("Improve Resume"):
    if not resume.strip():
        st.warning("Please enter a resume section.")
        st.stop()

    with st.spinner("Improving your resume..."):
        result = improve_resume(resume, job_desc)

        if result.startswith("Error") or result.startswith("Request failed"):
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