#!/usr/bin/env python3
"""
scripts/build_sft_train_llama3.py

Generates two files under data_processed/:

  1) sft_train.jsonl    – 25 SFT examples in Llama-3 chat format
  2) dataset_info.json  – metadata for LLaMA-Factory to load the above file
                          using template "llama3".

Usage:
  python scripts/build_sft_train_llama3.py
"""

import json
import pathlib

# 1) Ensure output directory exists
OUT_DIR = pathlib.Path("data_processed")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# 2) Paths for our two outputs
SFT_FILE     = OUT_DIR / "sft_train.jsonl"
DATASET_INFO = OUT_DIR / "dataset_info.json"

# 3) System‐level “You are …” prompt
SYSTEM_PROMPT = "You are an expert startup advisor."

# 4) We still define our 25 “instruction / input / output” triples,
#    but we will wrap them into chat‐style messages below.

INSTRUCTION_TEMPLATE = (
    "Evaluate the following summary and suggest concrete improvements."
)

# 4a) Here are 25 examples (unchanged content from before), each with:
#     { "instruction": <string>, "input": <string>, "output": <string> }
examples_alpaca = [
    # 1. SaaS / Marketing
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our SaaS platform aggregates social-media mentions of small retailers "
            "in real time, offering sentiment analysis and competitor benchmarks."
        ),
        "output": (
            "1. Narrow ICP: are you serving brick-and-mortar boutiques or e-commerce shops? "
            "A focused niche lowers CAC.\n"
            "2. Differentiate vs. free tools: highlight unique ML features (e.g., hyper-local "
            "sentiment heatmaps).\n"
            "3. Publish model benchmarks (F1 ≥ 0.88 on retail sentiment) to build trust.\n"
            "4. Add ROI calculator—show how a 5-point sentiment boost drives revenue."
        )
    },
    # 2. Consumer App / Food Waste
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "We built a mobile app that scans grocery receipts, tracks household food "
            "waste, and recommends recipes based on leftovers."
        ),
        "output": (
            "1. Quantify TAM: 36 M U.S. households waste $1,500/yr in food—cite USDA stats.\n"
            "2. Release OCR accuracy numbers (≥ 98 % on mixed fonts) vs. Google Vision.\n"
            "3. Integrate retailer loyalty APIs to auto-import purchase data—reduces friction.\n"
            "4. Experiment with affiliate links to grocery delivery for monetization."
        )
    },
    # 3. Hardware / Robotics
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "We manufacture a LiDAR-based robotic lawn mower that learns garden layouts "
            "over time, so homeowners can set it and forget it."
        ),
        "output": (
            "1. Publish field-test data: 2 cm positioning error on uneven terrain in rain "
            "gives you an edge over Husqvarna (5 cm).\n"
            "2. Add theft-deterrent and OTA software-update plan—reduces warranty cost.\n"
            "3. Consider a service+subscription model (hardware 40 % margin up-front, "
            "$20/mo SaaS for zone analytics).\n"
            "4. Secure UL safety certification before Q4 retail channel push."
        )
    },
    # 4. Telehealth
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "We’re launching a telehealth platform that combines wearable vitals with "
            "doctor video calls for chronic-care monitoring."
        ),
        "output": (
            "1. File HIPAA compliance statement—include AWS BAA references.\n"
            "2. Pilot with 2 outpatient clinics; track 90-day readmission reduction KPI.\n"
            "3. Expand device list beyond Apple Watch: partner with Dexcom for glucose.\n"
            "4. Clarify payer strategy: CPT 99457 reimbursement or direct employer plans?"
        )
    },
    # 5. Marketplace / Freelance
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our marketplace matches freelance graphic designers with SMEs using an "
            "AI ranking algorithm."
        ),
        "output": (
            "|Dimension|Current|Improvement|\n"
            "|---|---|---|\n"
            "|Differentiation|Generic AI score|Include design-style embeddings & past "
            "NPS data|\n"
            "|Supply QA|Portfolio only|Add timed skill test and reference checks|\n"
            "|Pricing|10 % fee|Tiered—5 % for top talent, 15 % for new|\n"
            "|Retention|None|Introduce project-timeline boards + milestone escrow|"
        )
    },
    # 6. Retail CV
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "We use computer vision to scan retail shelves and trigger restock alerts."
        ),
        "output": (
            "1. Edge vs. cloud inference cost comparison—store managers need predictable "
            "OPEX.\n"
            "2. Publish confusion-matrix across 1 K SKUs; aim for recall ≥ 0.95.\n"
            "3. Pre-integrate with Shopify POS and Oracle Netsuite for fast onboarding.\n"
            "4. Offer pay-per-store pricing ($99/mo) to speed small-chain adoption."
        )
    },
    # 7. Blockchain / Supply Chain
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our blockchain platform tracks perishable-goods provenance and reduces spoilage."
        ),
        "output": (
            "1. Replace proof-of-work chain with permissioned Hyperledger to cut tx cost "
            "from $0.30 to <$0.01.\n"
            "2. Show pilot: 4 % spoilage drop in 3-month banana shipment study.\n"
            "3. Bundle low-cost BLE temp sensors; embed cost into SaaS contract.\n"
            "4. Offer ESG scoring dashboard—retailers can report CO₂ reduction."
        )
    },
    # 8. EdTech
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our adaptive math-tutoring app gamifies exercises and gives real-time hints."
        ),
        "output": (
            "1. Map each level to Common Core IDs; teachers need alignment proof.\n"
            "2. Publish A/B trial: +17 % test-score improvement over Khan baseline.\n"
            "3. Add teacher admin console (class progress heat-map) for district sales.\n"
            "4. Freemium plan: 3 units free, $6/mo for full access—lower paywall friction."
        )
    },
    # 9. FinTech
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our ML underwriting engine approves SMB loans in minutes by analyzing real-time "
            "cash-flow data."
        ),
        "output": (
            "1. Show ROC AUC ≥ 0.92 vs. FICO 0.74 on 50 K historical loans.\n"
            "2. Outline compliance: CRA, ECOA, KYC/AML.\n"
            "3. Negotiate bank-partnership warehouse line—$50 M tranche covers first "
            "6 mo originations.\n"
            "4. Offer revenue-share model to POS providers (Square, Lightspeed) for "
            "embedded lending."
        )
    },
    # 10. Health Wearable
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our wearable detects atrial fibrillation in real time and alerts cardiologists."
        ),
        "output": (
            "1. File FDA 510(k) predicate comparison—AliveCor KardiaMobile as reference.\n"
            "2. Embed edge AI chip to reduce battery drain to <3 %/day.\n"
            "3. Co-author clinical paper with Mayo Clinic—boosts credibility with insurers.\n"
            "4. Monetize via annual monitoring subscription ($99) + device sale ($149)."
        )
    },
    # 11. CleanTech IoT
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "We deploy low-cost air-quality sensors in cities and sell real-time AQI data "
            "to residents."
        ),
        "output": (
            "1. Calibrate sensors quarterly vs. EPA reference monitors—publish MAE ≤ 3 µg/m³.\n"
            "2. Offer tiered data API for app developers (500 calls free, $0.002/call thereafter).\n"
            "3. Partner with insurance firms—use AQI to price respiratory-health premiums.\n"
            "4. Apply for municipal grants to subsidize sensor rollout."
        )
    },
    # 12. LegalTech
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our NLP tool flags risky clauses in contracts and suggests edits."
        ),
        "output": (
            "1. Train on >100 K annotated clauses; publish precision/recall 0.93/0.88.\n"
            "2. Add jurisdiction toggles (US/UK/EU) to handle local clause variants.\n"
            "3. Integrate with DocuSign and Microsoft Word for frictionless workflow.\n"
            "4. Pricing: $0.35 per contract page or $99/mo unlimited."
        )
    },
    # 13. AgriTech Drones
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our drones capture aerial imagery to detect crop diseases early."
        ),
        "output": (
            "1. Offer subscription per acre ($2/acre/season) instead of hardware sales—"
            "farmers prefer OPEX.\n"
            "2. Show disease-classification F1 ≥ 0.9 across 8 staple crops.\n"
            "3. Partner with pesticide vendors—upsell targeted treatment kits.\n"
            "4. Provide offline edge-processing for low-connectivity regions."
        )
    },
    # 14. AR / Real Estate
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our AR app stages virtual furniture in empty homes to boost real-estate sales."
        ),
        "output": (
            "1. Publish case study: staged homes sell 21 days faster (n=52 listings).\n"
            "2. Add e-commerce links so buyers can purchase displayed furniture—extra revenue stream.\n"
            "3. Provide CAD export for architects—broadens TAM.\n"
            "4. Bundle with Matterport 3-D scans to reduce onboarding friction."
        )
    },
    # 15. eSports Coaching
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our cloud platform offers amateur gamers live coaching from pro players plus AI performance analytics."
        ),
        "output": (
            "1. Segment by game genre—MOBA vs. FPS require different KPIs; tailor dashboards.\n"
            "2. Publish skill-lift stat: median Elo +200 after 10 sessions.\n"
            "3. Recruit 50 verified pro coaches via revenue-share (70/30) to seed supply.\n"
            "4. Streamline mobile spectator mode—lowers friction for Android users."
        )
    },
    # 16. TravelTech NLP
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our app generates multi-country itineraries in seconds using NLP."
        ),
        "output": (
            "1. List data sources: Amadeus, Skyscanner APIs—real-time pricing is table-stakes.\n"
            "2. Add ‘drag-and-drop’ reorder UI for customization; increases retention.\n"
            "3. Launch affiliate booking flow—10 % commission on lodging.\n"
            "4. Consider freemium: 3 itineraries free, $5 per extra."
        )
    },
    # 17. MentalHealthTech
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our AI chatbot delivers CBT techniques for mild anxiety with option to escalate to human therapists."
        ),
        "output": (
            "1. Publish safety audit—95 % correct CBT technique selection on annotated dataset of 1 K queries.\n"
            "2. Implement crisis-trigger phrases → immediate human handoff within 30 s.\n"
            "3. Obtain SOC 2 Type II + HIPAA audits—enter enterprise employer market.\n"
            "4. Sliding-scale pricing ($5–$25/mo) improves accessibility, aligns with mission."
        )
    },
    # 18. Sustainability Marketplace
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our marketplace lets neighbors trade surplus homegrown produce using a credit system."
        ),
        "output": (
            "1. Clarify logistics: integrate USPS flat-rate boxes or local pickup points.\n"
            "2. Gamify contributions—leaderboard of CO₂ avoided to incentivize supply.\n"
            "3. Offer premium ‘organic verified’ badge ($2/mo) for power growers.\n"
            "4. Partner with city compost programs—close the loop."
        )
    },
    # 19. HRTech
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our AI ranks job applicants for recruiters based on CV skill match and cultural fit."
        ),
        "output": (
            "1. Publish bias-audit results (gender, ethnicity) and mitigation methods (reweighing, counterfactual fairness).\n"
            "2. Add Chrome extension for recruiters—reduces ATS friction.\n"
            "3. Charge per job slot (starts at $49) with unlimited applicant scoring.\n"
            "4. Provide API for high-volume clients like RPO firms."
        )
    },
    # 20. Biotech
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "We use CRISPR-based liquid biopsy to detect early-stage pancreatic cancer "
            "from a 5 ml blood sample."
        ),
        "output": (
            "SWOT:\n"
            "S • Non-invasive test with 92 % sensitivity (n=300) — clear USP.\n"
            "W • Clinical trial Ph1 only; need >1 000 cohort for FDA PMA.\n"
            "O • Partner with Roche to leverage existing distribution to oncologists.\n"
            "T • Guardant & Grail have deeper pockets—focus on a single cancer to stay agile."
        )
    },
    # 21. Quantum Computing
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our cloud platform offers access to a 1 K-qubit neutral-atom quantum computer "
            "via Python SDK."
        ),
        "output": (
            "1. Publish quantum volume and 2-qubit gate fidelity—VCs need hard numbers.\n"
            "2. Offer hybrid classical + quantum solvers (QAOA + GPU fallback) to improve "
            "developer adoption.\n"
            "3. Start with research institutes (ARR $2 M) before enterprise expansion.\n"
            "4. Plan roadmap to 10 K qubits by 2027—communicate path to quantum advantage."
        )
    },
    # 22. GovTech / Smart City SaaS
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "We provide a SaaS digital-twin of municipal traffic networks to help city "
            "planners optimize signal timing and reduce congestion."
        ),
        "output": (
            "1. Reference compliance with NIST 800-53 & FedRAMP Moderate—procurement teams care.\n"
            "2. Offer ROI case study: 12 % travel-time reduction in Pilot City over 6 months.\n"
            "3. Provide open-data export so cities meet transparency regulations.\n"
            "4. Pricing: annual license $250 K per 1 M population + professional services."
        )
    },
    # 23. Social Enterprise / Off-Grid Solar
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our non-profit startup distributes low-cost solar lanterns to off-grid "
            "communities and funds the program through carbon-offset credits."
        ),
        "output": (
            "1. Quantify impact: each lantern avoids 200 kg CO₂/yr by replacing kerosene.\n"
            "2. Structure revenue: sell verified carbon credits ($8/ton) to corporates—"
            "projected ARR $1.6 M at 40 K lanterns.\n"
            "3. Build local repair hubs—boosts job creation and device longevity.\n"
            "4. Track SDG 7, 13 progress—critical when applying for impact grants."
        )
    },
    # 24. Web3 / GameFi
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our GameFi project rewards players with tradable tokens for completing quests "
            "in a fantasy MMORPG."
        ),
        "output": (
            "1. Publish tokenomics sheet: total supply, inflation schedule, and sink mechanics "
            "to curb hyper-inflation.\n"
            "2. Undergo smart-contract audit (e.g., CertiK) before mainnet launch.\n"
            "3. Add free-to-play onboarding with custodial wallets—reduces Web3 friction.\n"
            "4. Plan secondary-market liquidity via market-maker agreement; target 10 % "
            "daily turnover."
        )
    },
    # 25. PropTech / AI-Based Property Valuation
    {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": (
            "Our PropTech startup uses an AI model to automatically value residential properties "
            "based on MLS data, local comparables, and real-time market trends."
        ),
        "output": (
            "1. Validate model accuracy: compare AI valuations to closing prices in 5 major metros; "
            "publish Mean Absolute Error (MAE) ≤ $5,000.\n"
            "2. Explain data sources: which MLS feeds and third-party APIs (Zillow, Redfin) are used, "
            "and how often are they updated?\n"
            "3. Address regulatory/underwriting compliance: outline how your model meets Appraisal "
            "Foundation (USPAP) standards for automated valuations.\n"
            "4. Secure partnerships with real estate brokerages: integrate with their CRM to provide "
            "instant comps on listing pages.\n"
            "5. Discuss monetization: charge per valuation API call ($1/call) or offer monthly subscriptions "
            "for brokerages at $499/mo."
        )
    },
]

# 5) Now transform each Alpaca‐style example into Llama3 chat‐style JSON:
with open(SFT_FILE, "w", encoding="utf-8") as fp:
    for ex in examples_alpaca:
        user_content = f"{ex['instruction']}\n\n{ex['input']}"
        chat_example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content}
            ],
            "answer": ex["output"]
        }
        fp.write(json.dumps(chat_example, ensure_ascii=False) + "\n")

print(f"✅ Wrote {len(examples_alpaca)} chat‐format examples to {SFT_FILE}")

# 6) Create dataset_info.json for LLaMA-Factory with template "llama3"
dataset_info = {
    "sft_train": {
        "file_name": SFT_FILE.name,
        "format": "llama3",
        "columns": {
            "messages": "messages",
            "answer": "answer"
        }
    }
}

with open(DATASET_INFO, "w", encoding="utf-8") as fp:
    json.dump(dataset_info, fp, ensure_ascii=False, indent=2)

print(f"✅ Created dataset metadata at {DATASET_INFO}")
