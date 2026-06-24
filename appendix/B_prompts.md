# Appendix B — Prompt Templates

This appendix lists the verbatim prompts used in the project: the dataset
generation prompt (B.1) and the pipeline verdict prompts (B.2).

## B.1 Dataset generation prompt

The following prompt was submitted to **Gemini 3.1 Pro** on **30 March 2026** to
produce the initial 150-claim dataset. It specifies the seven-column schema, the
balanced label distribution (50 SUPPORTED / 50 REFUTED / 50 MISLEADING), the
topic coverage, the mix of claim styles and difficulty levels, and the restricted
list of authoritative source domains from which `Source_URL` and
`Evidence_Sentence` must be drawn.

```text
Create a dataset with exactly 150 English claims about measles for a Master's
thesis on automated public health fact-checking.

Output the dataset as a table that can be copied into Excel.

Use these columns in this exact order:
ID | Claim | Topic | Label | Difficulty | Source_URL | Evidence_Sentence

Requirements:
- Create exactly 150 claims.
- Use only English.
- Use IDs in the format MS001–MS150.
- Balance the labels evenly (uppercase):
  - 50 SUPPORTED
  - 50 REFUTED
  - 50 MISLEADING
- Claims should be short, clear, and realistic.
- Cover these topics: Transmission, Symptoms, Vaccination, Vaccine Safety,
  Immunity, Treatment, Epidemiology, Policy, History, Conspiracy, Pseudoscience.
- Use different claim styles:
  - normal factual claims
  - misleading or oversimplified claims
  - vaccine myths
  - social-media-style misinformation
- For REFUTED claims, include a mix of:
  - plausible but false claims
  - common misinformation
  - at least 15 absurd, conspiratorial, or pseudo-medical claims that sound like
    fringe social media posts (use Topic "Conspiracy" or "Pseudoscience" for these).
- MISLEADING claims should sound plausible but be technically inaccurate,
  oversimplified, cherry-picked, or missing important context — not outright absurd.
- Do not make all false claims absurd.
- Avoid duplicates.
- Keep each claim to one sentence if possible.
- Use Difficulty values: Easy, Medium, Hard.
- Every claim must have a Source_URL pointing to a real page from one of:
  CDC (cdc.gov), WHO (who.int), ECDC (ecdc.europa.eu), NFID (nfid.org),
  CHOP (chop.edu), NHS (nhs.uk), Johns Hopkins (publichealth.jhu.edu),
  or PubMed (pubmed.ncbi.nlm.nih.gov).
- Evidence_Sentence should be a short sentence taken from or closely paraphrasing
  the cited source that justifies the label.

After the table, provide the same dataset again as CSV in one code block.
Output only the dataset and the CSV, with no introduction or commentary.
```

*Listing B.1: Verbatim Gemini 3.1 Pro prompt used to generate the initial
150-claim dataset on 30 March 2026.*

## B.2 Pipeline verdict prompts

The verdict prompts used at inference time by the local LLMs (Qwen3 8B and
Mistral 7B served via Ollama) are defined in
[`../pipeline/verdict.py`](../pipeline/verdict.py). The prompt is composed of a
**system prompt** (which differs between the standard and the abstain-enabled
conditions) and a **user prompt** assembled per claim from the claim text, the
retrieved evidence block, and the required response format.

### B.2.1 System prompt — standard conditions (no abstain)

```text
You are a fact_checker assistant. Your task is to classify a health claim as
SUPPORTED, REFUTED, or MISLEADING and provide a short justification. Base your
verdict strictly on the provided evidence. If no evidence is provided, use your
general knowledge but flag uncertainty.
```

*Listing B.2: System prompt for the Baseline, RAG-only and RAG+NLI conditions.*

### B.2.2 System prompt — abstain conditions

```text
You are a fact_checker assistant. Your task is to classify a health claim into
exactly ONE of four categories:
  - SUPPORTED: evidence (or your reliable knowledge in baseline mode) explicitly
    confirms the claim.
  - REFUTED: evidence (or your reliable knowledge in baseline mode) explicitly
    contradicts the claim.
  - MISLEADING: the claim is partially true but cherry-picked, oversimplified, or
    missing critical context.
  - NOT_ENOUGH_EVIDENCE: the available information is insufficient to confirm,
    refute, or identify the claim as misleading.

IMPORTANT RULES:
  1. Do NOT default to REFUTED when the information does not directly address the
     claim. REFUTED requires an explicit contradiction.
  2. Use NOT_ENOUGH_EVIDENCE when the information is absent, generic, or
     tangential to the claim.
  3. Base your verdict strictly on the provided evidence (in RAG modes) or your
     reliable knowledge (in baseline mode).
```

*Listing B.3: System prompt for the +Abstain conditions.*

### B.2.3 User prompt construction

The user prompt is built per claim. It always begins with the claim, followed by
an evidence block that depends on the mode, and ends with the required response
format.

**Baseline** (no retrieval) — the evidence block is replaced by:

```text
CLAIM: <claim>

No external evidence is provided. Classify based on your parametric knowledge.
Flag any uncertainty.
```

**RAG-only** — retrieved passages are listed without NLI labels:

```text
CLAIM: <claim>

RETRIEVED EVIDENCE:
  [1] (<source>) <passage text>
  [2] (<source>) <passage text>
  ...
```

**RAG+NLI** — each retrieved passage is prefixed with its NLI label:

```text
CLAIM: <claim>

RETRIEVED EVIDENCE WITH NLI LABELS:
  [1] (<source>) [CONTRADICTION|ENTAILMENT|NEUTRAL] <passage text>
  [2] (<source>) [CONTRADICTION|ENTAILMENT|NEUTRAL] <passage text>
  ...
```

**Response format** appended to every user prompt — standard conditions:

```text
Respond with exactly:
VERDICT: <SUPPORTED | REFUTED | MISLEADING>
JUSTIFICATION: <2-4 sentences referencing the evidence>
```

**Response format** — abstain conditions:

```text
Respond with exactly:
VERDICT: <SUPPORTED | REFUTED | MISLEADING | NOT_ENOUGH_EVIDENCE>
JUSTIFICATION: <2-4 sentences referencing the evidence>
```

*Listing B.4: Per-claim user prompt construction across the three retrieval modes
and the two abstain settings (`pipeline/verdict.py`).*
