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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -----------------------
# LOAD MODEL (cached)
# -----------------------
@st.cache_resource
def load_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
    )

    return tokenizer, model


tokenizer, model = load_model()

# -----------------------
# GENERATION FUNCTION
# -----------------------
def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    input_length = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=600,
        temperature=0.3,
        do_sample=True,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )

    generated = outputs[0][input_length:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# -----------------------
# SAFE PARSER
# -----------------------
def safe_parse(output):
    try:
        return json.loads(output)
    except:
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass

    return {
        "revised_resume": output,
        "changes": [
            "Could not parse structured output",
            "Returned raw model response"
        ]
    }


# -----------------------
# PROMPT
# -----------------------
def improve_resume(text, job_desc):
    prompt = f"""[INST]
You are a professional resume editor.

Return ONLY valid JSON:

{{
  "revised_resume": "formatted resume in markdown",
  "changes": [
    "Change 1 + why",
    "Change 2 + why",
    "Change 3 + why"
  ]
}}

Rules:
- Must format resume properly (Title | Company | Dates + bullets)
- Must include at least 3 changes with explanations
- Must not include extra text outside JSON

Resume:
{text}

Job Description:
{job_desc}
[/INST]
"""
    return safe_parse(generate(prompt))


# -----------------------
# STREAMLIT UI
# -----------------------
st.title("AI Resume Improver")

resume = st.text_area("Paste your resume", height=250)
job_desc = st.text_area("Paste job description (optional)", height=150)

if st.button("Improve Resume"):
    with st.spinner("Improving resume..."):
        result = improve_resume(resume, job_desc)

        st.markdown("## ✨ Revised Resume")
        st.markdown(result["revised_resume"])

        st.markdown("## 🧠 What Changed & Why")
        for c in result["changes"]:
            st.markdown(f"- {c}")