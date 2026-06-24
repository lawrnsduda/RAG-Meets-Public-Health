# Appendix A — Measles Claim Dataset

This appendix documents the 150-claim measles evaluation dataset described in
Section 5.2. The full dataset, with the complete seven-column schema, is committed
in the repository:

> **[`../data/measles_claims_dataset.csv`](../data/measles_claims_dataset.csv)**

I do not reproduce the 150 claims here — the CSV is the authoritative copy. This
appendix records the schema, the label distribution and ID mapping, and how I
built and verified the dataset.

## A.1 Schema

| Column | Description |
|---|---|
| `ID` | Row identifier, re-keyed by label block (see A.3) |
| `Claim` | The measles-related claim (one sentence) |
| `Topic` | One of: Transmission, Symptoms, Vaccination, Vaccine Safety, Immunity, Treatment, Epidemiology, Policy, History, Conspiracy, Pseudoscience |
| `Label` | Ground-truth label: `SUPPORTED`, `REFUTED`, or `MISLEADING` |
| `Difficulty` | `Easy`, `Medium`, or `Hard` |
| `Source_URL` | Authoritative source page used for verification |
| `Evidence_Sentence` | Sentence from / paraphrasing the source that justifies the label |

## A.2 Label distribution and ID mapping

The dataset is organised in three equally sized blocks of 50 claims.

| Label | Number of claims | ID range |
|---|---|---|
| SUPPORTED | 50 | MS001 – MS050 |
| REFUTED | 50 | MR001 – MR050 |
| MISLEADING | 50 | MM001 – MM050 |
| **Total** | **150** | |

## A.3 Identifier re-keying

The generator produced sequential `MS001–MS150` identifiers as a generation
order. The externally visible row IDs were re-keyed by label block
(`MS001–MS050` for SUPPORTED, `MR001–MR050` for REFUTED, `MM001–MM050` for
MISLEADING) so the label is immediately recognisable in result tables, confusion
matrices and the qualitative error analysis. This re-keying is **cosmetic** and
does not affect the content of any claim, its label, or the post-generation
verification.

## A.4 Construction and verification

Each claim was generated with **Gemini 3.1 Pro** (Google, 2026) using the
structured prompt in [Appendix B](B_prompts.md). I then reviewed every claim
through **independent literature research** against peer-reviewed and
authoritative sources (WHO, CDC, ECDC, OWID, PubMed). I did not adopt the
generated labels unchanged — I cross-checked each claim against the cited source
and corrected the label wherever my verification disagreed with the generator.

### Block-specific notes

- **SUPPORTED (MS001–MS050).** Each claim states a fact directly verifiable
  against peer-reviewed or authoritative sources; evidence sentences confirm the
  claim verbatim or in near-identical wording.
- **REFUTED (MR001–MR050).** Each claim contradicts the scientific consensus.
  The block is a deliberate mix of plausible falsehoods, widespread vaccine
  myths, and conspiratorial claims, to reflect the heterogeneity of real-world
  measles misinformation (Schlicht et al., 2024). Evidence sentences from WHO,
  CDC, ECDC or PubMed contradict each claim.
- **MISLEADING (MM001–MM050).** Each claim contains a kernel of truth but is
  incomplete, cherry-picked, temporally distorted, or framed so as to lead to an
  incorrect conclusion. For each MISLEADING claim the annotated explanation
  follows a two-part *"What is true / What is misleading"* format, used as ground
  truth for the qualitative justification-quality evaluation.
