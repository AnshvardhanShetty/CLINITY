# Clinical State Compiler - Production Architecture

## Design Principles

### 1. Safety First
- **Never hallucinate** - only extract what is explicitly stated
- **Flag uncertainty** - clearly mark low-confidence extractions
- **Mandatory safety items** - allergies, DNAR status, critical results always surfaced
- **Fail safe** - if unsure, show the source text rather than a summary

### 2. Provenance is Non-Negotiable
- Every extracted item links to exact source document and text excerpt
- Clinician can click through to verify any claim
- Audit trail for medicolegal purposes

### 3. Designed for Time Pressure
- Output must be readable in <30 seconds
- Critical items at the top
- Visual hierarchy (urgency markers, color coding)
- Progressive disclosure (summary → details on demand)

### 4. Trust Through Transparency
- Show confidence levels
- Highlight conflicts between sources
- Explicitly state what's missing
- Never hide uncertainty

---

## Improved Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Images    │  │    Text     │  │   Audio     │  │    PDF      │ │
│  │ (photos of  │  │  (typed     │  │ (dictated   │  │  (letters,  │ │
│  │  handover)  │  │   notes)    │  │  handover)  │  │  reports)   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                │                │                │        │
│         ▼                ▼                ▼                ▼        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              DOCUMENT PREPROCESSOR                           │   │
│  │  - Image enhancement (contrast, deskew, denoise)            │   │
│  │  - Patient boundary detection (multiple patients per doc)   │   │
│  │  - Document type classification                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EXTRACTION LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              MULTI-PASS EXTRACTION                           │   │
│  │                                                              │   │
│  │  Pass 1: Raw Extraction                                      │   │
│  │  - Extract all clinical entities                             │   │
│  │  - Tag with source location (doc, line, char position)       │   │
│  │  - Assign initial confidence                                 │   │
│  │                                                              │   │
│  │  Pass 2: Verification                                        │   │
│  │  - Re-check each extraction against source                   │   │
│  │  - Verify critical items (allergies, meds, results)          │   │
│  │  - Flag uncertain or conflicting items                       │   │
│  │                                                              │   │
│  │  Pass 3: Safety Check                                        │   │
│  │  - Ensure mandatory fields present (allergies, resus status) │   │
│  │  - Flag critical abnormal results                            │   │
│  │  - Detect drug interactions or contraindications             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Output: Structured extraction with confidence + provenance          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      COMPILATION LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              CONFLICT RESOLUTION                             │   │
│  │  - Detect contradictions between sources                     │   │
│  │  - Apply recency rules (newer > older)                       │   │
│  │  - Flag unresolved conflicts for clinician review            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                   │                                  │
│                                   ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              PRIORITIZATION                                  │   │
│  │  - Rank by clinical urgency                                  │   │
│  │  - Surface safety-critical items first                       │   │
│  │  - Apply mode-specific filtering (handover vs discharge)     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                   │                                  │
│                                   ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              SYNTHESIS                                       │   │
│  │  - Generate concise current status summary                   │   │
│  │  - Compile structured sections                               │   │
│  │  - Attach provenance to every item                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    CLINICAL SNAPSHOT                           │ │
│  │                                                                │ │
│  │  ╔═══════════════════════════════════════════════════════════╗│ │
│  │  ║ ⚠️  SAFETY ALERTS                                         ║│ │
│  │  ║ • ALLERGY: Penicillin (rash) [DOC001]                    ║│ │
│  │  ║ • DNAR confirmed [DOC002]                                 ║│ │
│  │  ║ • CRITICAL: K+ 6.2 (↑) [DOC003]                          ║│ │
│  │  ╚═══════════════════════════════════════════════════════════╝│ │
│  │                                                                │ │
│  │  ACTIVE PROBLEMS                          Confidence           │ │
│  │  1. NSTEMI - troponin rising              ████████░░ 85%      │ │
│  │  2. AKI on CKD3 - Cr 180                  ██████████ 95%      │ │
│  │  3. ?New AF - rate controlled             ██████░░░░ 60%      │ │
│  │                                                                │ │
│  │  CURRENT STATUS                                                │ │
│  │  "72M Day 3, NSTEMI with improving AKI. Awaiting cardiology   │ │
│  │   review for ?angiogram. Clinically stable overnight."        │ │
│  │                                                                │ │
│  │  PENDING [3 items]                                             │ │
│  │  !! Cardiology review - not yet seen [DOC001]                 │ │
│  │  !  ECHO - requested, outstanding [DOC001]                    │ │
│  │     Chase blood cultures [DOC002]                              │ │
│  │                                                                │ │
│  │  UNCLEAR / CONFLICTS                                           │ │
│  │  ⚡ Penicillin allergy: "?allergic" vs "tolerated amox"       │ │
│  │     → Sources conflict [DOC001 vs DOC002] - VERIFY            │ │
│  │                                                                │ │
│  │  ────────────────────────────────────────────────────────────│ │
│  │  Sources: [DOC001] Ward round 15/01 | [DOC002] Nursing 16/01 │ │
│  │           [DOC003] Labs 16/01                                 │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Improvements

### 1. Multi-Pass Extraction with Verification

Instead of single-shot extraction, use multiple passes:

```
Pass 1: "Extract all clinical problems from this document"
Pass 2: "For each extracted problem, quote the exact source text that supports it"
Pass 3: "Are there any critical safety items (allergies, DNAR, critical results) that were missed?"
```

This catches errors and ensures nothing critical is missed.

### 2. Structured Output Schema

Force MedGemma to output structured JSON, then render it:

```json
{
  "patient_id": "HDU1",
  "extractions": [
    {
      "type": "problem",
      "text": "NSTEMI - troponin rising",
      "status": "active",
      "confidence": 0.85,
      "source": {
        "doc_id": "DOC001",
        "excerpt": "NSTEMI - troponin rising, cardiology review requested",
        "line": 12
      }
    }
  ],
  "safety_items": {
    "allergies": [...],
    "dnar_status": "for_resuscitation",
    "critical_results": [...]
  },
  "missing_mandatory": ["weight", "fluid_balance"]
}
```

### 3. Confidence Scoring

Each extraction gets a confidence score based on:
- Clarity of source text
- Verification pass agreement
- Presence of uncertainty markers ("?", "possibly", "unclear")

Display confidence visually so clinicians know what to double-check.

### 4. Safety-Critical Item Detection

Hardcoded rules for items that MUST be surfaced:
- Allergies (always top of output)
- Resuscitation status
- Critical lab values (K+ >6, Na+ <125, etc.)
- High-risk medications (anticoagulants, insulin, opioids)
- Infection control status (MRSA, C.diff)

### 5. Conflict Detection

When sources disagree:
- Flag the conflict explicitly
- Show both versions with sources
- Don't pick one - let the clinician decide

### 6. Multi-Patient Document Handling

For ward handover sheets with multiple patients:
1. First pass: detect patient boundaries
2. Extract per-patient
3. Output separate snapshots or a ward list view

---

## What This Means for the Hackathon

For the prototype, implement:
1. ✅ Vision-first extraction (done)
2. 🔲 Multi-pass verification
3. 🔲 Structured JSON output schema
4. 🔲 Safety item highlighting
5. 🔲 Confidence display
6. 🔲 Simple UI with provenance links

This demonstrates:
- Technical depth (multi-pass, structured output)
- Clinical understanding (safety-first design)
- Real-world applicability (handles messy real documents)
- Responsible AI (transparency, uncertainty handling)
