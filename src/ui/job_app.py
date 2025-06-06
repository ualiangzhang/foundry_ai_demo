# app.py

import os
import streamlit as st
import openai

# —— 1. Configure OpenAI API Key —— #
# It’s recommended to set OPENAI_API_KEY in your environment.
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_GOES_HERE")


# —— 2. Define a function that asks OpenAI whether to filter the JD —— #
def needs_filtering(jd_text: str) -> (bool, list):
    """
    Given the full JD text, call OpenAI to check if the JD meets any of:
      1) Requires on-site in California with < $150k salary
      2) Requires Security clearance
      3) Travel > 20%
      4) Company size < 50 employees
      5) Not full-time (contract, intern, part-time, etc.)
    Returns (filter_flag, reasons_list). If filter_flag=True, reasons_list contains the matching criteria numbers.
    """
    system_prompt = (
        "You are a recruiting assistant. The user will provide the entire Job Description (JD). "
        "Please check if the JD satisfies any of the following conditions:\n"
        "1. Requires on-site work AND location is in California AND salary < $150k per year\n"
        "2. Requires Security clearance\n"
        "3. Travel requirement > 20%\n"
        "4. Company size < 50 employees\n"
        "5. Position is not full-time (e.g., contract, intern, part-time, etc.)\n\n"
        "Output exactly a JSON object of the form:\n"
        "{\n"
        "  \"filter\": \"yes\" or \"no\",\n"
        "  \"reasons\": [ list of condition numbers as strings, e.g. [\"1\",\"4\"] ]\n"
        "}\n\n"
        "For example, if the JD requires on-site in California with salary 140k, output:\n"
        "{\"filter\":\"yes\",\"reasons\":[\"1\"]}\n"
        "If none of the conditions is met, output:\n"
        "{\"filter\":\"no\",\"reasons\":[]}\n"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": jd_text}
        ],
        temperature=0
    )
    reply = response.choices[0].message["content"].strip()

    # Try to parse JSON
    try:
        import json
        result = json.loads(reply)
        return (result.get("filter", "").lower() == "yes", result.get("reasons", []))
    except Exception:
        # If parsing fails, default to not filtering
        return False, []


# —— 3. Build the Streamlit interface —— #
st.set_page_config(page_title="JD Quick Filter", page_icon="🔍", layout="centered")
st.title("🔍 JD Quick Filter")

st.markdown(
    """
    Paste the full Job Description (JD) text below, then click “Check Filter” to have the app:

    1. Exclude any JD that requires **on-site** work in **California** with salary **< $150k/year**  
    2. Exclude any JD that requires **Security clearance**  
    3. Exclude any JD that requires **Travel > 20%**  
    4. Exclude any JD whose company size is **< 50 employees**  
    5. Exclude any JD that is **not full-time** (e.g., contract, intern, or part-time)
    """
)

jd_input = st.text_area("Paste the Job Description text here:", height=300)

if st.button("Check Filter"):
    if not jd_input.strip():
        st.warning("⚠️ Please paste the JD text first.")
    else:
        with st.spinner("Analyzing with OpenAI…"):
            filter_flag, reasons = needs_filtering(jd_input)
        if filter_flag:
            st.error("🚫 This job has been filtered out. Matching condition(s): " + ", ".join(reasons))
            st.write("Details of each condition:")
            reason_map = {
                "1": "On-site in California with salary < $150k/year",
                "2": "Requires Security clearance",
                "3": "Travel requirement > 20%",
                "4": "Company size < 50 employees",
                "5": "Position is not full-time"
            }
            for r in reasons:
                desc = reason_map.get(r, "Unknown condition")
                st.write(f"- {r}. {desc}")
        else:
            st.success("✅ This job is NOT filtered out. You can continue reviewing or applying.")

# —— 4. Instructions to run locally —— #
# 1. Install dependencies:
#    pip install streamlit openai
# 2. Set OPENAI_API_KEY in your environment:
#    export OPENAI_API_KEY="sk-xxxxxx"
# 3. Run:
#    streamlit run app.py
