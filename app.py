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
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cpu"

# -----------------------
# LAZY MODEL LOADING
# -----------------------
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        low_cpu_mem_usage=True
    )

    model.to(device)
    return tokenizer, model


# -----------------------
# GENERATION FUNCTION
# -----------------------
def generate(prompt):
    tokenizer, model = load_model()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.3,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )

    generated = outputs[0][input_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# -----------------------
# JSON PARSER
# -----------------------
def extract_json(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None


def safe_parse(output):
    parsed = extract_json(output)

    if not parsed:
        return {
            "revised_resume": output,
            "changes": [
                "Model output was not valid JSON",
                "Returned raw output instead"
            ]
        }

    if not parsed.get("changes"):
        parsed["changes"] = ["No change explanation provided"]

    return parsed


# -----------------------
# PROMPT
# -----------------------
def improve_resume(text, job_desc=""):
    prompt = f"""
You are a professional resume editor.

You are given ONLY a section of a resume (NOT the full resume).

Your job is to improve THIS SECTION ONLY.

OUTPUT FORMAT (follow EXACTLY):

REVISED SECTION:
<same structure as input, but improved>

CHANGES:
- explain what was changed and why
- explain what was changed and why
- explain what was changed and why

STRICT RULES:
- DO NOT add new sections (NO Skills, NO Summary, etc.)
- DO NOT invent information
- KEEP the same structure as the input
- If input is a job experience:
    → keep job title, company, dates
    → keep bullet format
- ONLY rewrite and improve wording
- Make bullets more impactful, concise, and technical
- Each change must explain WHY

INPUT SECTION:
{text}

JOB DESCRIPTION (optional):
{job_desc}
"""
    return generate(prompt)

# -----------------------
# UI
# -----------------------
st.title("AI Resume Improver")

resume = st.text_area("Paste your resume", height=250)
job_desc = st.text_area("Paste job description (optional)", height=150)

if st.button("Improve Resume"):
    if not resume.strip():
        st.warning("Please enter a resume first.")
        st.stop()

    with st.spinner("Loading model and improving resume... (first run takes ~20–40s)"):
        result = improve_resume(resume, job_desc)

        if "CHANGES:" in result:
            section_part, changes_part = result.split("CHANGES:", 1)
        else:
            section_part = result
            changes_part = "No changes provided"

        section_part = section_part.replace("REVISED SECTION:", "").strip()

        st.markdown("## ✨ Revised Section")
        st.markdown(section_part)

        st.markdown("## 🧠 What Changed & Why")
        for line in changes_part.strip().split("\n"):
            if line.strip():
                st.markdown(line)