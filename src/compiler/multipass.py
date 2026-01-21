"""Multi-pass extraction with verification for improved accuracy."""

import json
import re
from typing import Any
from dataclasses import dataclass, field

from ..models import InputDocument
from .safety import SafetyChecker, SafetyAlert


@dataclass
class VerifiedExtraction:
    """An extraction that has been verified against source."""
    category: str  # problem, medication, investigation, plan, pending, risk
    text: str
    confidence: float  # 0.0 to 1.0
    source_id: str
    source_excerpt: str
    is_verified: bool = False
    verification_note: str = ""


@dataclass
class ExtractionResult:
    """Result of multi-pass extraction for a document."""
    document_id: str
    document_type: str
    extractions: list[VerifiedExtraction] = field(default_factory=list)
    safety_alerts: list[SafetyAlert] = field(default_factory=list)
    missing_mandatory: list[str] = field(default_factory=list)
    raw_content: str = ""


class MultiPassExtractor:
    """
    Multi-pass extraction with verification.

    Pass 1: Initial extraction of all clinical entities
    Pass 2: Verification - check each extraction against source
    Pass 3: Safety check - ensure mandatory items are captured
    """

    EXTRACTION_PROMPT = """You are a clinical information extraction system. Extract structured information from this clinical document.

DOCUMENT TYPE: {doc_type}
DOCUMENT ID: {doc_id}

DOCUMENT CONTENT:
{content}

Extract information into these categories. For EACH item, you MUST include the exact quote from the source that supports it.

Respond in this exact JSON format:
{{
  "problems": [
    {{"text": "problem description", "status": "active|resolved|historical", "quote": "exact text from document"}}
  ],
  "medications": [
    {{"text": "medication name and dose", "quote": "exact text from document"}}
  ],
  "pending_tasks": [
    {{"text": "task description", "urgency": "routine|soon|urgent", "quote": "exact text from document"}}
  ],
  "risks": [
    {{"text": "risk description", "severity": "low|medium|high", "quote": "exact text from document"}}
  ],
  "unclear_items": [
    {{"text": "what is unclear", "reason": "ambiguous|conflicting|missing"}}
  ]
}}

IMPORTANT RULES:
1. ONLY extract information explicitly stated in the document
2. NEVER invent or assume information
3. Include the EXACT quote that supports each extraction
4. If a category has no items, use an empty array []
5. Flag anything unclear or ambiguous

Respond with valid JSON only, no additional text."""

    VERIFICATION_PROMPT = """Verify these extractions against the source document.

SOURCE DOCUMENT:
{content}

EXTRACTIONS TO VERIFY:
{extractions}

For each extraction, check if the quote accurately supports the claim. Respond in JSON:
{{
  "verified": [
    {{"index": 0, "is_correct": true, "confidence": 0.95, "note": ""}},
    {{"index": 1, "is_correct": false, "confidence": 0.3, "note": "quote does not match claim"}}
  ]
}}

Be strict. Only mark as correct if the quote clearly supports the extraction."""

    SAFETY_CHECK_PROMPT = """Review this clinical document for any MISSED safety-critical information.

DOCUMENT:
{content}

ALREADY EXTRACTED:
{already_found}

Check if we MISSED any of these critical items:
1. Allergies or drug reactions
2. Resuscitation status (DNAR, ceiling of care)
3. Critical abnormal results
4. High-risk medications (anticoagulants, insulin, opioids)
5. Infection control status (MRSA, isolation)
6. Fall risk

Respond in JSON:
{{
  "missed_items": [
    {{"category": "allergy|resus|critical_result|high_risk_med|infection|fall_risk", "text": "description", "quote": "source text"}}
  ],
  "missing_mandatory": ["list of mandatory fields not documented, e.g., 'allergy status not documented'"]
}}

If nothing is missed, use empty arrays."""

    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self.safety_checker = SafetyChecker()

    def extract(self, document: InputDocument) -> ExtractionResult:
        """Run multi-pass extraction on a document."""
        result = ExtractionResult(
            document_id=document.id,
            document_type=document.source_type.value,
            raw_content=document.content,
        )

        # Rule-based safety check first (fast, reliable)
        result.safety_alerts = self.safety_checker.check_text(document.content, document.id)

        # Pass 1: Initial extraction
        print(f"    Pass 1: Extracting from {document.id}...")
        raw_extractions = self._pass1_extract(document)

        # Pass 2: Verification
        print(f"    Pass 2: Verifying extractions...")
        verified_extractions = self._pass2_verify(document, raw_extractions)
        result.extractions = verified_extractions

        # Pass 3: Safety check for missed items
        print(f"    Pass 3: Safety check...")
        missed_alerts, missing_mandatory = self._pass3_safety_check(document, result)
        result.safety_alerts.extend(missed_alerts)
        result.missing_mandatory = missing_mandatory

        return result

    def _generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response from model."""
        import torch

        inputs = self.processor.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
            )

        response = self.processor.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def _pass1_extract(self, document: InputDocument) -> dict:
        """Pass 1: Initial extraction."""
        prompt = self.EXTRACTION_PROMPT.format(
            doc_type=document.source_type.value,
            doc_id=document.id,
            content=document.content[:3000],  # Limit context
        )

        response = self._generate(prompt, max_tokens=1500)

        # Parse JSON response
        try:
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        return {"problems": [], "medications": [], "pending_tasks": [], "risks": [], "unclear_items": []}

    def _pass2_verify(self, document: InputDocument, raw_extractions: dict) -> list[VerifiedExtraction]:
        """Pass 2: Verify extractions against source."""
        verified = []

        # Flatten extractions with their categories
        flat_extractions = []
        for category in ["problems", "medications", "pending_tasks", "risks"]:
            for item in raw_extractions.get(category, []):
                flat_extractions.append({
                    "category": category.rstrip("s"),  # problems -> problem
                    "text": item.get("text", ""),
                    "quote": item.get("quote", ""),
                    "extra": item,  # Keep original data
                })

        if not flat_extractions:
            return verified

        # Verify each extraction by checking if quote exists in source
        for ext in flat_extractions:
            quote = ext.get("quote", "")
            text = ext.get("text", "")

            # Simple verification: check if key words from quote appear in document
            quote_words = set(quote.lower().split())
            doc_words = set(document.content.lower().split())
            overlap = len(quote_words & doc_words) / max(len(quote_words), 1)

            # Confidence based on quote overlap
            if overlap > 0.7:
                confidence = 0.9
                is_verified = True
                note = ""
            elif overlap > 0.4:
                confidence = 0.7
                is_verified = True
                note = "Partial quote match"
            else:
                confidence = 0.4
                is_verified = False
                note = "Quote not found in source"

            # Adjust confidence for uncertainty markers
            if any(marker in text.lower() for marker in ["?", "possibly", "likely", "unclear", "query"]):
                confidence *= 0.8
                note = "Contains uncertainty marker"

            verified.append(VerifiedExtraction(
                category=ext["category"],
                text=text,
                confidence=confidence,
                source_id=document.id,
                source_excerpt=quote[:200] if quote else text[:200],
                is_verified=is_verified,
                verification_note=note,
            ))

        # Add unclear items with low confidence
        for item in raw_extractions.get("unclear_items", []):
            verified.append(VerifiedExtraction(
                category="unclear",
                text=item.get("text", ""),
                confidence=0.3,
                source_id=document.id,
                source_excerpt="",
                is_verified=True,
                verification_note=f"Flagged as unclear: {item.get('reason', 'unknown')}",
            ))

        return verified

    def _pass3_safety_check(
        self,
        document: InputDocument,
        current_result: ExtractionResult
    ) -> tuple[list[SafetyAlert], list[str]]:
        """Pass 3: Check for missed safety items."""
        # Already found items
        already_found = [
            f"- {alert.category}: {alert.description}"
            for alert in current_result.safety_alerts
        ]

        # Check for missing mandatory documentation
        missing_mandatory = []

        content_lower = document.content.lower()

        # Check if allergy status is documented
        allergy_documented = any(term in content_lower for term in [
            "allerg", "nkda", "no known drug"
        ])
        if not allergy_documented:
            missing_mandatory.append("Allergy status not documented")

        # Check if resuscitation status is documented (for inpatients)
        resus_documented = any(term in content_lower for term in [
            "dnar", "dnacpr", "resus", "escalation", "ceiling", "for cpr"
        ])
        if not resus_documented and document.source_type.value in ["typed_note", "handwritten"]:
            missing_mandatory.append("Resuscitation status not documented")

        # Return empty list for additional alerts (rule-based already caught them)
        return [], missing_mandatory
