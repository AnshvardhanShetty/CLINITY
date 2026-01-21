"""Safety-critical item detection and validation."""

import re
from typing import Literal
from dataclasses import dataclass


@dataclass
class SafetyAlert:
    """A safety-critical item that must be surfaced."""
    category: Literal["allergy", "resus_status", "critical_result", "high_risk_med", "infection_control", "fall_risk"]
    description: str
    severity: Literal["critical", "high", "medium"]
    source_id: str
    source_excerpt: str


class SafetyChecker:
    """Detect and validate safety-critical items."""

    # Critical lab value thresholds
    CRITICAL_LABS = {
        "potassium": {"low": 2.5, "high": 6.0, "unit": "mmol/L"},
        "sodium": {"low": 120, "high": 160, "unit": "mmol/L"},
        "glucose": {"low": 2.0, "high": 25.0, "unit": "mmol/L"},
        "haemoglobin": {"low": 70, "high": None, "unit": "g/L"},
        "platelets": {"low": 50, "high": None, "unit": "x10^9/L"},
        "inr": {"low": None, "high": 5.0, "unit": ""},
        "creatinine": {"low": None, "high": 400, "unit": "umol/L"},
    }

    # High-risk medications
    HIGH_RISK_MEDS = [
        "warfarin", "heparin", "enoxaparin", "rivaroxaban", "apixaban", "dabigatran",  # Anticoagulants
        "insulin", "metformin", "gliclazide", "glipizide",  # Diabetes meds
        "morphine", "oxycodone", "fentanyl", "codeine", "tramadol",  # Opioids
        "digoxin", "amiodarone",  # Cardiac
        "methotrexate", "chemotherapy",  # Cytotoxics
        "lithium",  # Psych
    ]

    # Allergy keywords
    ALLERGY_PATTERNS = [
        r"allerg(?:y|ies|ic)\s*(?:to)?:?\s*([^,.\n]+)",
        r"nkda",  # No known drug allergies
        r"allergies?:\s*([^,.\n]+)",
    ]

    # Resuscitation status
    RESUS_PATTERNS = [
        r"(dnar|dnacpr|not for resus|for resuscitation|full escalation|ceiling of care|for cpr)",
        r"resus(?:citation)?\s*status:?\s*([^,.\n]+)",
    ]

    def check_text(self, text: str, source_id: str) -> list[SafetyAlert]:
        """Scan text for safety-critical items."""
        alerts = []
        text_lower = text.lower()

        # Check for allergies
        alerts.extend(self._check_allergies(text, text_lower, source_id))

        # Check for resuscitation status
        alerts.extend(self._check_resus_status(text, text_lower, source_id))

        # Check for critical lab values
        alerts.extend(self._check_critical_labs(text, text_lower, source_id))

        # Check for high-risk medications
        alerts.extend(self._check_high_risk_meds(text, text_lower, source_id))

        # Check for infection control
        alerts.extend(self._check_infection_control(text, text_lower, source_id))

        # Check for fall risk
        alerts.extend(self._check_fall_risk(text, text_lower, source_id))

        return alerts

    def _check_allergies(self, text: str, text_lower: str, source_id: str) -> list[SafetyAlert]:
        alerts = []
        for pattern in self.ALLERGY_PATTERNS:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                excerpt = text[max(0, match.start()-20):min(len(text), match.end()+20)]
                allergy_text = match.group(1) if match.lastindex else match.group(0)

                if "nkda" in allergy_text or "no known" in allergy_text:
                    continue  # Not an actual allergy

                alerts.append(SafetyAlert(
                    category="allergy",
                    description=f"ALLERGY: {allergy_text.strip().title()}",
                    severity="critical",
                    source_id=source_id,
                    source_excerpt=excerpt.strip(),
                ))
        return alerts

    def _check_resus_status(self, text: str, text_lower: str, source_id: str) -> list[SafetyAlert]:
        alerts = []
        for pattern in self.RESUS_PATTERNS:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                excerpt = text[max(0, match.start()-10):min(len(text), match.end()+10)]
                status = match.group(1) if match.lastindex else match.group(0)

                severity = "critical" if "dnar" in status or "not for" in status else "high"

                alerts.append(SafetyAlert(
                    category="resus_status",
                    description=f"RESUS STATUS: {status.upper()}",
                    severity=severity,
                    source_id=source_id,
                    source_excerpt=excerpt.strip(),
                ))
        return alerts

    def _check_critical_labs(self, text: str, text_lower: str, source_id: str) -> list[SafetyAlert]:
        alerts = []

        # Pattern: lab_name: value or lab_name value (unit)
        lab_patterns = [
            (r"potassium[:\s]+(\d+\.?\d*)", "potassium"),
            (r"k\+?[:\s]+(\d+\.?\d*)", "potassium"),
            (r"sodium[:\s]+(\d+\.?\d*)", "sodium"),
            (r"na\+?[:\s]+(\d+\.?\d*)", "sodium"),
            (r"glucose[:\s]+(\d+\.?\d*)", "glucose"),
            (r"h(?:ae)?moglobin[:\s]+(\d+\.?\d*)", "haemoglobin"),
            (r"\bhb[:\s]+(\d+\.?\d*)", "haemoglobin"),
            (r"platelets?[:\s]+(\d+\.?\d*)", "platelets"),
            (r"inr[:\s]+(\d+\.?\d*)", "inr"),
            (r"creatinine[:\s]+(\d+\.?\d*)", "creatinine"),
        ]

        for pattern, lab_name in lab_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                try:
                    value = float(match.group(1))
                    thresholds = self.CRITICAL_LABS.get(lab_name, {})

                    is_critical = False
                    direction = ""

                    if thresholds.get("low") and value < thresholds["low"]:
                        is_critical = True
                        direction = "↓↓"
                    elif thresholds.get("high") and value > thresholds["high"]:
                        is_critical = True
                        direction = "↑↑"

                    if is_critical:
                        excerpt = text[max(0, match.start()-10):min(len(text), match.end()+10)]
                        alerts.append(SafetyAlert(
                            category="critical_result",
                            description=f"CRITICAL: {lab_name.title()} {value} {direction}",
                            severity="critical",
                            source_id=source_id,
                            source_excerpt=excerpt.strip(),
                        ))
                except ValueError:
                    continue

        return alerts

    def _check_high_risk_meds(self, text: str, text_lower: str, source_id: str) -> list[SafetyAlert]:
        alerts = []
        for med in self.HIGH_RISK_MEDS:
            if med in text_lower:
                # Find the context
                idx = text_lower.find(med)
                excerpt = text[max(0, idx-20):min(len(text), idx+len(med)+20)]

                alerts.append(SafetyAlert(
                    category="high_risk_med",
                    description=f"HIGH-RISK MED: {med.title()}",
                    severity="high",
                    source_id=source_id,
                    source_excerpt=excerpt.strip(),
                ))
        return alerts

    def _check_infection_control(self, text: str, text_lower: str, source_id: str) -> list[SafetyAlert]:
        alerts = []
        infection_terms = ["mrsa", "c.diff", "c difficile", "vre", "esbl", "cpe", "covid", "tb ", "tuberculosis", "isolation"]

        for term in infection_terms:
            if term in text_lower:
                idx = text_lower.find(term)
                excerpt = text[max(0, idx-15):min(len(text), idx+len(term)+15)]

                alerts.append(SafetyAlert(
                    category="infection_control",
                    description=f"INFECTION CONTROL: {term.upper()}",
                    severity="high",
                    source_id=source_id,
                    source_excerpt=excerpt.strip(),
                ))
        return alerts

    def _check_fall_risk(self, text: str, text_lower: str, source_id: str) -> list[SafetyAlert]:
        alerts = []
        fall_terms = ["fall risk", "falls risk", "high falls", "bed rails", "1:1 supervision", "mobility aid"]

        for term in fall_terms:
            if term in text_lower:
                idx = text_lower.find(term)
                excerpt = text[max(0, idx-15):min(len(text), idx+len(term)+15)]

                alerts.append(SafetyAlert(
                    category="fall_risk",
                    description=f"FALL RISK: {excerpt.strip()}",
                    severity="medium",
                    source_id=source_id,
                    source_excerpt=excerpt.strip(),
                ))
                break  # Only one fall risk alert needed

        return alerts


# Singleton instance
safety_checker = SafetyChecker()
