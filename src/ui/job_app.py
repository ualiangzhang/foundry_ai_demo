import os
import re
import json
import openai

openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_GOES_HERE")


def ask_gpt_for_missing(jd_text: str, missing_fields: list) -> dict:
    """
    Ask GPT to fill in any missing fields and extract a list of technical skills.
    Returns a dict with keys = missing_fields + ["skills"].
    """
    prompt = (
        "You are a recruiting assistant. Below is a full Job Description (JD). "
        "I have already extracted some fields via regex but failed to get: "
        f"{', '.join(missing_fields)}.\n\n"
        "Please read the JD and return a JSON object containing exactly these keys:\n"
        + "\n".join([f"- \"{field}\": <value>" for field in missing_fields]) + "\n"
        "- \"skills\": a JSON array of all professional/technical skills mentioned in the JD.\n\n"
        "If you cannot find a particular field, set its value to null. Do NOT output any extra keys.\n\n"
        "Return only valid JSON.\n\n"
        "=== JOB DESCRIPTION START ===\n\n"
        f"{jd_text}\n\n"
        "=== JOB DESCRIPTION END ===\n"
    )

    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who extracts structured fields from a job description."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0
    )
    answer = resp.choices[0].message.content.strip()

    try:
        return json.loads(answer)
    except json.JSONDecodeError:
        return {}


def extract_from_jd(jd_text: str) -> dict:
    """
    1) Use regex to extract:
       - salary_low, salary_high  (USD/year)
       - employee_count_max       (integer)
       - state_abbrev             (2-letter US state code)
       - travel_pct               (integer percent)
       - clearance_required       (True/False)
       - full_time                (True/False)
       - skills                   (initial pass via regex)
    2) If any of the above fields is missing, call GPT to fill them.
    3) Return a dict with all extracted fields + “conclusion” string.
    """

    info = {
        "salary_low": None,
        "salary_high": None,
        "employee_count_max": None,
        "state_abbrev": None,
        "travel_pct": None,
        "clearance_required": None,
        "full_time": None,
        "skills": []
    }

    text = jd_text

    # 1) SALARY RANGE
    salary_patterns = [
        re.compile(r"\$\s*([\d,]+)[Kk]?/?yr?\s*[-–]\s*\$\s*([\d,]+)[Kk]?/?yr?", re.IGNORECASE),
        re.compile(r"USD\s*([\d,]+)[Kk]?\s*[-–]\s*([\d,]+)[Kk]?", re.IGNORECASE),
        re.compile(r"\$\s*([\d,]+)\s*[-–]\s*\$\s*([\d,]+)", re.IGNORECASE)
    ]
    for pat in salary_patterns:
        m = pat.search(text)
        if m:
            low_raw, high_raw = m.groups()
            def to_int(x):
                x = x.replace(",", "")
                if x.lower().endswith("k"):
                    return int(float(x[:-1]) * 1000)
                return int(x)
            try:
                info["salary_low"] = to_int(low_raw)
                info["salary_high"] = to_int(high_raw)
                break
            except ValueError:
                pass

    # 2) EMPLOYEE COUNT
    emp_pat_range = re.compile(r"(\d+)\s*[-–]\s*(\d+)\s+employees", re.IGNORECASE)
    emp_pat_single = re.compile(r"(\d+)\s+employees", re.IGNORECASE)
    m = emp_pat_range.search(text)
    if m:
        info["employee_count_max"] = int(m.group(2))
    else:
        m2 = emp_pat_single.search(text)
        if m2:
            info["employee_count_max"] = int(m2.group(1))

    # 3) STATE ABBREVIATION (US)
    state_pat = re.compile(r"\b(?:in\s+)?[A-Za-z .]+,\s*([A-Z]{2})\b")
    m = state_pat.search(text)
    if m:
        info["state_abbrev"] = m.group(1)

    # 4) TRAVEL PERCENTAGE
    travel_pat = re.compile(r"(\d{1,3})\s*%\s*(?:travel|Travel)", re.IGNORECASE)
    m = travel_pat.search(text)
    if m:
        info["travel_pct"] = int(m.group(1))

    # 5) SECURITY CLEARANCE?
    if re.search(r"security clearance|requires clearance|Top Secret|Secret clearance|public trust", text, re.IGNORECASE):
        info["clearance_required"] = True
    else:
        info["clearance_required"] = False

    # 6) FULL-TIME?
    if re.search(r"\bfull[- ]?time\b", text, re.IGNORECASE):
        info["full_time"] = True
    elif re.search(r"\bcontract\b|\bintern\b|\bpart[- ]?time\b", text, re.IGNORECASE):
        info["full_time"] = False
    else:
        info["full_time"] = None

    # 7) SKILLS (expanded list of common data-science & related technologies)
    possible_tech = [
        # Programming Languages
        "Python", "R", "Java", "Scala", "C\\+\\+", "C#", "Go", "Julia", "Ruby",
        "JavaScript", "TypeScript", "SQL",

        # Data Manipulation / Analysis
        "Pandas", "NumPy", "SciPy", "dplyr", "tidyr", "data\\.table", "data\\.table",

        # Machine Learning Frameworks
        "scikit[- ]?learn", "TensorFlow", "PyTorch", "Keras", "XGBoost", "LightGBM",
        "CatBoost", "H2O", "MLflow", "PySpark", "Spark MLlib", "RapidMiner", "WEKA",

        # Deep Learning & NLP
        "Transformers", "HuggingFace", "BERT", "GPT-\\d", "OpenNMT", "spaCy", "NLTK",

        # Big Data Ecosystem
        "Hadoop", "Hive", "HBase", "HDFS", "Spark", "Flink", "Kafka", "Presto", "Trino",
        "Impala", "Cassandra", "MongoDB", "Elasticsearch", "Redis", "Couchbase", "Neo4j",

        # Cloud Platforms / Services
        "AWS", "Amazon Web Services", "Azure", "Google Cloud", "GCP", "BigQuery",
        "Redshift", "Snowflake", "Databricks", "S3", "EC2", "Lambda", "Cloud Functions",
        "Azure Data Factory", "Azure ML", "AWS SageMaker", "GCP AI Platform",

        # Data Engineering / Orchestration
        "Airflow", "Apache Airflow", "Luigi", "Prefect", "Dagster", "Talend", "NiFi",
        "Kafka Streams", "StreamSets", "NiFi", "Informatica",

        # Containerization / Deployment / CI-CD
        "Docker", "Kubernetes", "Helm", "Kubeflow", "Argo", "Jenkins", "CircleCI", "GitHub Actions",
        "Travis CI", "Azure DevOps", "Terraform", "Ansible", "Chef", "Puppet",

        # Visualization / BI
        "Tableau", "Power BI", "Looker", "Qlik", "D3\\.js", "Matplotlib", "Seaborn", "Plotly",
        "ggplot2", "Shiny", "Dash", "Streamlit",

        # Database Technologies
        "PostgreSQL", "MySQL", "Oracle", "SQL Server", "MariaDB", "SQLite", "Teradata",
        "Vertica", "Greenplum", "SAP HANA", "Exasol",

        # Workflow / Version Control / Misc
        "Git", "SVN", "Mercurial", "JIRA", "Confluence", "Slack", "Confluence", "Jupyter Notebook",
        "JupyterLab", "Zeppelin", "VS Code", "PyCharm", "RStudio"
    ]

    skills_found = set()
    for tech in possible_tech:
        # Use word boundaries, or exact matches for multi-word names
        pattern = rf"\b{tech}\b"
        if re.search(pattern, text, re.IGNORECASE):
            # Normalize capitalization to the pattern itself (remove regex escapes)
            clean_tech = tech.replace("\\", "")
            skills_found.add(clean_tech)

    info["skills"] = sorted(skills_found)

    # 8) Identify which fields are still missing
    missing = []
    for field in ["salary_low", "salary_high", "employee_count_max", "state_abbrev", "travel_pct", "full_time"]:
        if info[field] is None:
            missing.append(field)

    if not info["skills"]:
        missing.append("skills")

    # 9) If anything is missing, ask GPT to fill in those fields + skills
    if missing:
        gpt_result = ask_gpt_for_missing(jd_text, missing)
        for key in missing:
            if key in gpt_result and gpt_result[key] is not None:
                info[key] = gpt_result[key]

    # 10) Build a “conclusion” string summarizing what we found
    conclusions = []
    if info["salary_low"] is not None and info["salary_high"] is not None:
        conclusions.append(f"Salary range: ${info['salary_low']:,} – ${info['salary_high']:,}/yr")
    if info["employee_count_max"] is not None:
        conclusions.append(f"Max employees: {info['employee_count_max']}")
    if info["state_abbrev"]:
        conclusions.append(f"State: {info['state_abbrev']}")
    if info["travel_pct"] is not None:
        conclusions.append(f"Travel requirement: {info['travel_pct']}%")
    if info["clearance_required"] is True:
        conclusions.append("Requires security clearance")
    elif info["clearance_required"] is False:
        conclusions.append("Does not require security clearance")
    if info["full_time"] is True:
        conclusions.append("Full-time position")
    elif info["full_time"] is False:
        conclusions.append("Not full-time (contract/intern/part-time)")
    if info["skills"]:
        conclusions.append(f"Skills mentioned: {', '.join(info['skills'])}")

    info["conclusion"] = "; ".join(conclusions) if conclusions else "No structured info could be extracted."

    return info


# —————————————————————————————————————————————————————————————————————————————
# Example usage
# —————————————————————————————————————————————————————————————————————————————
if __name__ == "__main__":
    jd_text = """
    Software Engineer, Data Platform
    Bellevue, WA · 2 hours ago · 11 people clicked apply
    Promoted by hirer · Responses managed off LinkedIn

    $147K/yr - $221K/yr
    Matches your job preferences, minimum pay preference is 120000.

    On-site
    Matches your job preferences, workplace type is On-site.

    Full-time
    Matches your job preferences, job type is Full-time.

    Qualifications

    Bachelor’s or Master’s degree with 3+ years of relevant experience.
    Proven expertise in deploying open-source solutions such as Kubernetes, Kafka, Hadoop, Flink, Spark, Presto, and Trino.
    Familiarity with cloud provider solutions, including AWS and GCP.
    Previous experience in data platform construction is essential.
    Open-source contributions are a plus.

    Required Skills

    Expertise in data engineering and platform development.
    Strong problem-solving skills and ability to work collaboratively.

    Preferred Skills

    Experience with large-scale data processing.
    Knowledge of data governance and security best practices.

    The US base salary range for this full-time position is listed below.
    Annual Base Pay Range: $147,000—$221,000 USD.

    Jobright.ai company logo
    Jobright.ai
    2–10 employees
    """

    result = extract_from_jd(jd_text)

    print("=== Extracted Information ===")
    for key, val in result.items():
        if key != "skills":
            print(f"{key}: {val}")
    print("skills:", result["skills"])
    print("\nConclusion:\n", result["conclusion"])
