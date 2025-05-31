#!/usr/bin/env python3
"""
scripts/build_stf_train_with_info.py

This script creates a small ShareGPT-style supervised fine-tuning (SFT) dataset
for LLaMA-3 based on positive and negative examples of startup feedback.

It outputs:
    - data_processed/sft_train.jsonl : Each line is a JSON with "messages" containing
      a system prompt, user prompt (including context), and assistant output.
    - data_processed/dataset_info.json : JSON metadata describing the SFT dataset.

Usage:
    python scripts/build_stf_train_with_info.py

Ensure:
    - The 'data_processed' directory exists or can be created by this script.
    - The POSITIVE and NEGATIVE example lists are defined within the script.

Logging:
    The script logs progress and errors to stdout. Check logs for any issues.

Example:
    mkdir -p data_processed
    python scripts/build_stf_train_with_info.py
"""

import json
import logging
import pathlib
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─────────────────────────── paths ─────────────────────────────────────────────
OUT_DIR: pathlib.Path = pathlib.Path("data_processed")
OUT_DIR.mkdir(exist_ok=True, parents=True)

SFT_FILE: pathlib.Path = OUT_DIR / "sft_train.jsonl"
DATASET_INFO: pathlib.Path = OUT_DIR / "dataset_info.json"

# ─────────────────────────── Prompt templates ───────────────────────────────────
SYSTEM_PROMPT: str = (
    "You are a veteran VC partner. Using ONLY the reference snippets, "
    "produce exactly FOUR numbered recommendations covering market, "
    "product, business model and team. Each bullet ≤ 50 words, no questions. "
    "If the snippets are empty, reply exactly `INSUFFICIENT_CONTEXT`."
)
USER_PREFIX: str = "### Startup summary\n"
CTX_PREFIX: str = "\n\n### Reference snippets\n"

# ─────────────────────────── Positive examples (25) ─────────────────────────────
POSITIVE: List[Dict[str, str]] = [
    {
        "input": (
            "Our SaaS platform aggregates social-media mentions of small retailers "
            "in real time, offering sentiment analysis and competitor benchmarks."
        ),
        "context": "Social sentiment tools adoption among SMB retailers grew 37 % YoY (Gartner 2024).",
        "output": (
            "1. **Market** – Focus on brick-and-mortar boutiques first; the 37 % YoY uptake (Gartner) shows unmet demand.\n"
            "2. **Product** – Ship hyper-local heat-map widget; differentiates vs. Hootsuite (<CTX1>).\n"
            "3. **Biz model** – Bundle a $99/mo ROI dashboard; highlight how 5-pt sentiment lift → +8 % sales.\n"
            "4. **Team** – Hire a retail data-partnership lead to secure POS integrations."
        )
    },
    {
        "input": (
            "We built a mobile app that scans grocery receipts, tracks household food "
            "waste, and recommends recipes based on leftovers."
        ),
        "context": "USDA estimates $1,500/yr of edible food wasted per household (2023).",
        "output": (
            "1. **Market** – $48 B addressable (32 M HH × $1.5 k waste).\n"
            "2. **Product** – Publish OCR accuracy ≥ 98 % on mixed fonts to trump Google Vision.\n"
            "3. **Biz model** – Affiliate links to Instacart: $2 ARPU/mo realistic.\n"
            "4. **Team** – Bring on a Registered Dietitian for regulatory trust."
        )
    },
    {
        "input": (
            "We manufacture a LiDAR-based robotic lawn mower that learns garden layouts."
        ),
        "context": "Robotic mower TAM projected $4.3 B by 2028 (Allied MR 2023).",
        "output": (
            "1. **Market** – Lead with premium prosumer segment (≥ $1k ASP) inside $4.3 B TAM.\n"
            "2. **Product** – Publish 2 cm positioning error vs. Husqvarna’s 5 cm to win specs wars.\n"
            "3. **Biz model** – Add $20/mo SaaS for zone analytics; smooth revenue curve.\n"
            "4. **Team** – Recruit UL certification veteran before Q4 retail push."
        )
    },
    {
        "input": (
            "We’re launching a telehealth platform combining wearables with doctor video calls."
        ),
        "context": "Chronic-care telehealth cuts readmissions by 18 % (JAMA 2023).",
        "output": (
            "1. **Market** – Target CHF + diabetes first; hospitals pay to cut 18 % readmission penalty.\n"
            "2. **Product** – Integrate Dexcom CGM to broaden chronic cohort.\n"
            "3. **Biz model** – Bill CPT 99457; show CFOs $3 ROI per $1 spent.\n"
            "4. **Team** – Add HIPAA compliance officer; speeds payer contracts."
        )
    },
    {
        "input": (
            "Our marketplace matches freelance graphic designers with SMEs using an AI ranking algorithm."
        ),
        "context": "SME design spend hits $27 B globally (IBIS 2024).",
        "output": (
            "1. **Market** – Own the $5 B North American slice first; less price compression.\n"
            "2. **Product** – Train embeddings on style + past NPS to beat generic AI rankers.\n"
            "3. **Biz model** – Tier fee: 5 % top-talent, 15 % new entrants; increases take while retaining supply.\n"
            "4. **Team** – Hire community manager to vet portfolios and run timed skill tests."
        )
    },
    {
        "input": "We use computer vision to scan retail shelves and trigger restock alerts.",
        "context": "Shelf-scanning robots reduce out-of-stock by 25 % in pilot stores (Stanford Retail 2023).",
        "output": (
            "1. **Market** – Target grocery chains first; 25 % OOS reduction is a clear ROI.\n"
            "2. **Product** – Publish recall ≥ 0.95 on 1k SKUs vs. alternatives.\n"
            "3. **Biz model** – Offer $99/mo per store with SLA-backed uptime.\n"
            "4. **Team** – Partner with a systems integrator to speed DS integration."
        )
    },
    {
        "input": "Our blockchain platform tracks perishable-goods provenance and reduces spoilage.",
        "context": "4 % spoilage drop in 3-month banana pilot saved $120k/yr (Kroger study 2023).",
        "output": (
            "1. **Market** – Focus on bananas → pilot proved $120k savings; expand to berries.\n"
            "2. **Product** – Migrate to Hyperledger Fabric; cut tx cost from $0.30 to <$0.01.\n"
            "3. **Biz model** – Charge 1 % of spoilage savings; high-margin, usage-based pricing.\n"
            "4. **Team** – Add supply-chain technologist with retail P&L experience."
        )
    },
    {
        "input": "Our adaptive math-tutoring app gamifies exercises and gives real-time hints.",
        "context": "A/B test: +17 % math scores vs. Khan Academy over 6 weeks (EdTech Research 2024).",
        "output": (
            "1. **Market** – Target schools first; 17 % lift resonates with district buyers.\n"
            "2. **Product** – Map to Common Core; include teacher dashboard for class progress.\n"
            "3. **Biz model** – Freemium: first 3 units free, $6/mo/full access; proven conversion.\n"
            "4. **Team** – Hire former teacher-administrator to accelerate district sales."
        )
    },
    {
        "input": "Our ML underwriting engine approves SMB loans in minutes by analyzing real-time cash-flow data.",
        "context": "SMB loan defaults dropped 20 % vs. FICO-only models (Fed 2023 pilot).",
        "output": (
            "1. **Market** – Embed with POS providers (Square has 2M merchants); 20 % default drop is a strong plug.\n"
            "2. **Product** – Publish ROC AUC ≥ 0.92 vs. FICO 0.74 on 50k loans.\n"
            "3. **Biz model** – Offer revenue-share with banks: 2 % origination fee, shared risk.\n"
            "4. **Team** – Hire compliance director to manage KYC/AML oversight."
        )
    },
    {
        "input": "Our wearable detects atrial fibrillation in real time and alerts cardiologists.",
        "context": "AFib screening devices have 93 % sensitivity in outpatient trials (Circulation 2023).",
        "output": (
            "1. **Market** – Sell direct to cardiology groups; 93 % sensitivity beats AliveCor’s 89 %.\n"
            "2. **Product** – Embed edge AI chip to reduce battery drain to <3 %/day.\n"
            "3. **Biz model** – Monitor subscription $99/yr + hardware $149; consensus yields 40 % margin.\n"
            "4. **Team** – Partner with Mayo Clinic for clinical validation."
        )
    },
    {
        "input": (
            "We deploy low-cost air-quality sensors in cities and sell real-time AQI data to residents."
        ),
        "context": "EPA compliance data shows 15 % under-reporting in city sensors vs. lab monitors.",
        "output": (
            "1. **Market** – Target cities with known under-reporting; value proposition clear.\n"
            "2. **Product** – Calibrate vs. EPA ref monitors: show MAE ≤ 3 µg/m³.\n"
            "3. **Biz model** – Tiered API: 500 calls free, $0.002/call after; attracts dev ecosystem.\n"
            "4. **Team** – Hire PhD environmental engineer to oversee calibration."
        )
    },
    {
        "input": "Our NLP tool flags risky clauses in contracts and suggests edits.",
        "context": "85 % of medium-sized firms lack in-house legal teams (LEXISNEXIS 2023 survey).",
        "output": (
            "1. **Market** – Target midsize firms (85 % underserved); legal spend $24B/yr.\n"
            "2. **Product** – Publish Precision 0.93, Recall 0.88 on 100k clauses.\n"
            "3. **Biz model** – $0.35/page or $99/mo unlimited; aligns with SMB budgets.\n"
            "4. **Team** – Integrate with DocuSign & Word; hire former GC to navigate adoption."
        )
    },
    {
        "input": "Our drones capture aerial imagery to detect crop diseases early.",
        "context": "Early detection can reduce pesticide spend by 18 % (USDA 2022).",
        "output": (
            "1. **Market** – Target large soybean farms; 18 % cost saving resonates.\n"
            "2. **Product** – Publish disease-classification F1 ≥ 0.9 across 8 staple crops.\n"
            "3. **Biz model** – Subscription per acre $2/acre/season; farmers prefer OPEX.\n"
            "4. **Team** – Partner with pesticide vendors to bundle targeted treatment kits."
        )
    },
    {
        "input": "Our AR app stages virtual furniture in empty homes to boost real-estate sales.",
        "context": "Staged homes sell 21 days faster on average (NAR 2023 report).",
        "output": (
            "1. **Market** – Proof: homes sell 21 days faster; pitch to listing agents.\n"
            "2. **Product** – Add e-commerce links so buyers purchase featured furniture; new revenue stream.\n"
            "3. **Biz model** – Charge $199 per listing; $99 installation fee.\n"
            "4. **Team** – Integrate with Matterport scans; hire UX designer for seamless AR UX."
        )
    },
    {
        "input": (
            "Our cloud platform offers amateur gamers live coaching from pro players plus AI performance analytics."
        ),
        "context": "Median skill lift +200 Elo after 10 sessions (Watson Esports 2024 study).",
        "output": (
            "1. **Market** – Target mid-tier gamers first; +200 Elo stat is a strong hook.\n"
            "2. **Product** – Segment by game genre (MOBA vs. FPS) to tailor analytics.\n"
            "3. **Biz model** – Revenue-share 70/30 with coaches; recruit 50 verified pros to seed supply.\n"
            "4. **Team** – Hire growth hacker experienced in Twitch integrations for viral loops."
        )
    },
    {
        "input": "Our app generates multi-country itineraries in seconds using NLP.",
        "context": "64 % of travelers use itinerary planners; global TAM $1.6B (Phocuswright 2023).",
        "output": (
            "1. **Market** – Partner with travel blogs to capture 64 % who plan itineraries.\n"
            "2. **Product** – Add drag-and-drop reorder UI; improves retention by 12 % (UX study 2023).\n"
            "3. **Biz model** – Affiliate lodging: 10 % commission; $5 ARPU/mo feasible.\n"
            "4. **Team** – Hire ex-Trivago developer to optimize scraping reliability."
        )
    },
    {
        "input": (
            "Our AI chatbot delivers CBT techniques for mild anxiety with option to escalate to human therapists."
        ),
        "context": "Digital CBT reduces anxiety symptoms by 42 % in 8 weeks (JAMA Psychiatry 2023).",
        "output": (
            "1. **Market** – Seek employers: mental health benefit saves average $3 per $1 invested.\n"
            "2. **Product** – Publish 42 % symptom reduction stat vs. baseline.\n"
            "3. **Biz model** – Sliding-scale subscription $5–$25/mo; improves accessibility.\n"
            "4. **Team** – Implement crisis-trigger phrases → human handoff within 30s."
        )
    },
    {
        "input": (
            "Our marketplace lets neighbors trade surplus homegrown produce using a credit system."
        ),
        "context": "Community food sharing reduces landfill by 15 % in pilot towns (EPA 2022).",
        "output": (
            "1. **Market** – Highlight 15 % landfill reduction; appeals to eco-conscious users.\n"
            "2. **Product** – Gamify contributions: leaderboard of CO₂ avoided.\n"
            "3. **Biz model** – Offer premium ‘organic verified’ badge at $2/mo for power growers.\n"
            "4. **Team** – Partner with city compost programs to close the loop."
        )
    },
    {
        "input": (
            "Our AI ranks job applicants for recruiters based on CV skill match and cultural fit."
        ),
        "context": "70 % of firms report bias in manual CV screening (SHRM 2023).",
        "output": (
            "1. **Market** – Target RPO firms that lose $8 B/yr to bad hires; AI bias reduction is a strong pitch.\n"
            "2. **Product** – Publish bias-audit results; show 25 % lower gender bias vs. baseline.\n"
            "3. **Biz model** – Charge per job slot $49; unlimited scoring.\n"
            "4. **Team** – Add diversity & inclusion lead to oversee fairness algorithms."
        )
    },
    {
        "input": (
            "We use CRISPR-based liquid biopsy to detect early-stage pancreatic cancer from a 5 ml blood sample."
        ),
        "context": "Pancreatic cancer 5-year survival < 10 %; early detection boosts to 35 % (NIH 2023).",
        "output": (
            "1. **Market** – Sell to oncology clinics: early detection boosts survival to 35 %.\n"
            "2. **Product** – Publish 92 % sensitivity/89 % specificity on 300-patient study.\n"
            "3. **Biz model** – $499/test reimbursed by insurers; margin 60 %.\n"
            "4. **Team** – Begin Phase II trial with Mayo Clinic collaboration."
        )
    },
    {
        "input": (
            "Our cloud platform offers access to a 1K-qubit neutral-atom quantum computer via Python SDK."
        ),
        "context": "Quantum volume 512 achieved; 2-qubit gate fidelity 99.5 % (Q Research 2024).",
        "output": (
            "1. **Market** – Target academic research first; 512 quantum volume is a compelling hook.\n"
            "2. **Product** – Offer hybrid QAOA + GPU fallback; expands developer base.\n"
            "3. **Biz model** – Research subscription $15k/mo for 100h quantum time.\n"
            "4. **Team** – Add classical HPC veteran to optimize hybrid workloads."
        )
    },
    {
        "input": (
            "We provide a SaaS digital-twin of municipal traffic networks to help city planners optimize signal timing."
        ),
        "context": "Pilot city saw 12 % travel-time reduction over 6 months (MIT Smart City 2023).",
        "output": (
            "1. **Market** – Package to mid-size cities (pop 500k–1M); 12 % reduction is high-ROI.\n"
            "2. **Product** – Export open-data to meet city transparency mandates.\n"
            "3. **Biz model** – Annual license $250k/1M population + $50k consulting.\n"
            "4. **Team** – Hire NIST compliance expert to streamline federal grants."
        )
    },
    {
        "input": (
            "Our non-profit startup distributes low-cost solar lanterns to off-grid communities and funds through carbon-credit sales."
        ),
        "context": "Each lantern avoids 200 kg CO₂/yr compared to kerosene (UNEP 2023).",
        "output": (
            "1. **Market** – Emphasize 200 kg CO₂/yr impact; taps into CSR budgets.\n"
            "2. **Product** – Structure verified carbon credit sales at $8/ton → $1.6M ARR at 40k lanterns.\n"
            "3. **Biz model** – Set up local repair hubs; extends device life by 50 %.\n"
            "4. **Team** – Collaborate with regional NGOs to scale community outreach."
        )
    },
    {
        "input": (
            "Our GameFi project rewards players with tradable tokens for completing quests in a fantasy MMORPG."
        ),
        "context": "Mid-core MMORPGs see 26 % Day 30 retention vs. F2P average 12 % (AppAnnie 2024).",
        "output": (
            "1. **Market** – Target mid-core segment; 26 % Day-30 retention is an entry KPI.\n"
            "2. **Product** – Publish token sink mechanics to maintain price stability.\n"
            "3. **Biz model** – Onboard custodial wallets to cut friction by 40 % (Polygon study).\n"
            "4. **Team** – Audit smart contracts with CertiK before mainnet launch."
        )
    },
    {
        "input": (
            "Our PropTech startup uses an AI model to automatically value residential properties based on MLS data."
        ),
        "context": "AVMs underwrite 70 % of US mortgages pre-qual (Fannie Mae 2024).",
        "output": (
            "1. **Market** – Capture broker API demand; AVMs handle 7M transactions/yr.\n"
            "2. **Product** – Demonstrate MAE ≤ $5k vs. Redfin’s $6.2k in 5 metros.\n"
            "3. **Biz model** – Charge $1/api call or $499/mo broker license; 85 % gross margin.\n"
            "4. **Team** – Hire USPAP-certified appraiser to meet regulatory audits."
        )
    },
]

# ────────────────────────── Negative “no-context” examples ────────────────────────
NEGATIVE: List[Dict[str, str]] = [
    {
        "input": "We design algae-based protein bars for endurance athletes.",
        "context": "",
        "output": "INSUFFICIENT_CONTEXT",
    },
    {
        "input": "We rent autonomous delivery robots to college campuses.",
        "context": "",
        "output": "INSUFFICIENT_CONTEXT",
    },
    {
        "input": "We build AR glasses for industrial maintenance workers.",
        "context": "",
        "output": "INSUFFICIENT_CONTEXT",
    },
]


# ───────────────────────── helper to convert to ShareGPT format ─────────────────
def to_conversation(ex: Dict[str, str]) -> Dict[str, List[Dict[str, str]]]:
    """Convert a single example into ShareGPT-style conversation format.

    Args:
        ex: A dictionary containing "input", "context", and "output" keys.

    Returns:
        A dictionary with a "messages" list containing three message objects:
        - system message with SYSTEM_PROMPT
        - user message combining USER_PREFIX, input, CTX_PREFIX, and context
        - assistant message with the expected output
    """
    user_msg: str = USER_PREFIX + ex["input"] + CTX_PREFIX + ex["context"]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": ex["output"]}
        ]
    }


# ───────────────────────── Write SFT JSONL ─────────────────────────────────────
def write_sft_file(file_path: pathlib.Path, examples: List[Dict[str, str]]) -> int:
    """Write the provided positive and negative examples to a JSONL file.

    Args:
        file_path: Path to the output JSONL file.
        examples: List of example dictionaries with "input", "context", "output".

    Returns:
        The total number of examples written.

    Raises:
        IOError: If writing to the file fails.
    """
    count: int = 0
    try:
        with file_path.open("w", encoding="utf-8") as fp:
            for ex in examples:
                conv_obj = to_conversation(ex)
                fp.write(json.dumps(conv_obj, ensure_ascii=False) + "\n")
                count += 1
        logger.info(f"✅  Wrote {count} examples → {file_path}")
        return count
    except (OSError, IOError) as e:
        logger.error(f"Failed to write SFT file '{file_path}': {e}")
        raise


# ───────────────────────── Write dataset_info.json ────────────────────────────
def write_dataset_info(file_path: pathlib.Path, info: Dict) -> None:
    """Write the dataset metadata to a JSON file.

    Args:
        file_path: Path to the output dataset_info.json file.
        info: Dictionary containing dataset metadata.

    Raises:
        IOError: If writing to the file fails.
    """
    try:
        with file_path.open("w", encoding="utf-8") as fp:
            json.dump(info, fp, indent=2, ensure_ascii=False)
        logger.info(f"✅  Created {file_path}")
    except (OSError, IOError) as e:
        logger.error(f"Failed to write dataset info file '{file_path}': {e}")
        raise


def main() -> None:
    """Main entry point for building the SFT dataset and metadata."""
    all_rows: List[Dict[str, str]] = POSITIVE + NEGATIVE

    # Write SFT training file
    examples_written: int = write_sft_file(SFT_FILE, all_rows)

    # Prepare dataset info metadata
    info: Dict = {
        "sft_train": {
            "file_name": SFT_FILE.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system"
            }
        }
    }

    # Write dataset_info.json
    write_dataset_info(DATASET_INFO, info)

    logger.info(f"Process completed: {examples_written} examples and metadata written.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
