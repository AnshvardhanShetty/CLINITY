"""Core Clinical State Compiler - assembles inputs into a unified clinical snapshot."""

from datetime import datetime
from typing import Literal

from ..config import config
from ..models import (
    InputDocument,
    ClinicalSnapshot,
    ExtractedItem,
    PendingItem,
    RiskFlag,
    UnclearItem,
)
from .medgemma import MedGemmaProcessor


class ClinicalCompiler:
    """
    The core compiler that transforms multiple clinical documents
    into a single, authoritative clinical snapshot.
    """

    def __init__(self):
        self.medgemma = MedGemmaProcessor()
        self._extracted_data: list[dict] = []

    def compile(
        self,
        documents: list[InputDocument],
        mode: Literal["handover", "discharge", "gp_summary", "general"] = "general",
    ) -> ClinicalSnapshot:
        """
        Compile multiple documents into a single clinical snapshot.

        Args:
            documents: List of InputDocuments to compile
            mode: Output mode affecting prioritization and structure

        Returns:
            ClinicalSnapshot with all extracted and reconciled information
        """
        if not documents:
            raise ValueError("No documents provided to compile")

        # Step 1: Extract information from each document
        print(f"Extracting information from {len(documents)} documents...")
        self._extracted_data = []
        for doc in documents:
            print(f"  Processing: {doc.id} ({doc.source_type.value})")
            extracted = self.medgemma.extract_clinical_info(doc)
            extracted["_document"] = doc
            self._extracted_data.append(extracted)

        # Step 2: Merge and deduplicate
        print("Merging extracted information...")
        merged = self._merge_extractions(self._extracted_data)

        # Step 3: Resolve conflicts
        print("Resolving conflicts...")
        resolved, conflicts = self._resolve_all_conflicts(merged)

        # Step 4: Prioritize based on mode
        print(f"Prioritizing for mode: {mode}")
        prioritized = self._prioritize_for_mode(resolved, mode)

        # Step 5: Generate current status synthesis
        print("Synthesizing current status...")
        current_status = self.medgemma.synthesize_status(documents, self._extracted_data)

        # Step 6: Build the snapshot
        print("Building clinical snapshot...")
        snapshot = self._build_snapshot(
            documents=documents,
            prioritized=prioritized,
            conflicts=conflicts,
            current_status=current_status,
            mode=mode,
        )

        return snapshot

    def _merge_extractions(self, extractions: list[dict]) -> dict:
        """Merge extractions from multiple documents."""
        merged = {
            "active_problems": [],
            "medications": [],
            "investigations": [],
            "plans": [],
            "pending": [],
            "risks": [],
            "unclear": [],
            "events": [],
        }

        for extraction in extractions:
            doc = extraction.get("_document")
            source_id = doc.id if doc else "unknown"

            for problem in extraction.get("active_problems", []):
                merged["active_problems"].append({
                    **problem,
                    "source_id": source_id,
                    "timestamp": doc.timestamp if doc else None,
                })

            for item in extraction.get("pending", []):
                merged["pending"].append({
                    **item,
                    "source_id": source_id,
                })

            for risk in extraction.get("risks", []):
                merged["risks"].append({
                    **risk,
                    "source_id": source_id,
                })

            for unclear in extraction.get("unclear", []):
                merged["unclear"].append({
                    **unclear,
                    "source_id": source_id,
                })

            # Plans become events if they have timestamps
            for plan in extraction.get("plans", []):
                merged["events"].append({
                    **plan,
                    "source_id": source_id,
                    "timestamp": doc.timestamp if doc else None,
                })

        return merged

    def _resolve_all_conflicts(self, merged: dict) -> tuple[dict, list[dict]]:
        """Resolve conflicts across all categories."""
        all_conflicts = []

        # Resolve conflicts in problems
        if len(merged["active_problems"]) > 1:
            resolved_problems, conflicts = self.medgemma.resolve_conflicts(
                merged["active_problems"], "active problems"
            )
            merged["active_problems"] = resolved_problems
            all_conflicts.extend(conflicts)

        # Deduplicate pending items (simple text matching)
        seen_pending = set()
        unique_pending = []
        for item in merged["pending"]:
            text_lower = item["text"].lower().strip()
            if text_lower not in seen_pending:
                seen_pending.add(text_lower)
                unique_pending.append(item)
        merged["pending"] = unique_pending

        return merged, all_conflicts

    def _prioritize_for_mode(self, data: dict, mode: str) -> dict:
        """Prioritize information based on output mode."""
        # Mode-specific prioritization
        if mode == "handover":
            # Handover: emphasize pending tasks, risks, immediate concerns
            # Sort pending by urgency indicators in text
            data["pending"] = self._sort_by_urgency(data["pending"])

        elif mode == "discharge":
            # Discharge: emphasize follow-up plans, medications, GP actions
            pass

        elif mode == "gp_summary":
            # GP: emphasize diagnoses, medications, outstanding investigations
            pass

        # Default: general prioritization by recency and importance
        return data

    def _sort_by_urgency(self, items: list[dict]) -> list[dict]:
        """Sort items by detected urgency."""
        urgency_keywords = {
            "high": ["urgent", "immediately", "asap", "stat", "emergency", "critical"],
            "medium": ["soon", "today", "tomorrow", "priority", "important"],
        }

        def get_urgency_score(item: dict) -> int:
            text = item.get("text", "").lower()
            for keyword in urgency_keywords["high"]:
                if keyword in text:
                    return 0
            for keyword in urgency_keywords["medium"]:
                if keyword in text:
                    return 1
            return 2

        return sorted(items, key=get_urgency_score)

    def _build_snapshot(
        self,
        documents: list[InputDocument],
        prioritized: dict,
        conflicts: list[dict],
        current_status: str,
        mode: str,
    ) -> ClinicalSnapshot:
        """Build the final ClinicalSnapshot object."""
        # Build source map
        sources = {
            doc.id: f"{doc.source_type.value}: {doc.filename or 'unnamed'}"
            for doc in documents
        }

        # Convert to model objects
        active_problems = [
            ExtractedItem(
                text=p["text"],
                category="problem",
                source_id=p["source_id"],
                source_excerpt=p["text"],  # Simplified
                confidence=self._assess_confidence(p),
                is_current=True,
                timestamp=p.get("timestamp"),
            )
            for p in prioritized.get("active_problems", [])
        ]

        key_events = [
            ExtractedItem(
                text=e["text"],
                category="event",
                source_id=e["source_id"],
                source_excerpt=e["text"],
                timestamp=e.get("timestamp"),
            )
            for e in prioritized.get("events", [])
        ]

        pending_tasks = [
            PendingItem(
                description=p["text"],
                source_id=p["source_id"],
                source_excerpt=p["text"],
                urgency=self._detect_urgency(p["text"]),
            )
            for p in prioritized.get("pending", [])
        ]

        risks = [
            RiskFlag(
                description=r["text"],
                source_id=r["source_id"],
                source_excerpt=r["text"],
                severity=self._detect_severity(r["text"]),
            )
            for r in prioritized.get("risks", [])
        ]

        unclear_items = [
            UnclearItem(
                description=u["text"],
                reason="ambiguous",
                source_ids=[u["source_id"]],
                source_excerpts=[u["text"]],
            )
            for u in prioritized.get("unclear", [])
        ]

        # Add conflicts as unclear items
        for conflict in conflicts:
            unclear_items.append(
                UnclearItem(
                    description=f"Conflicting information: {conflict['text']}",
                    reason="conflicting",
                    source_ids=[conflict["source_id"]],
                    source_excerpts=[conflict["text"]],
                )
            )

        return ClinicalSnapshot(
            generated_at=datetime.now(),
            input_document_count=len(documents),
            mode=mode,
            active_problems=active_problems,
            current_status=current_status,
            key_events=key_events,
            pending_tasks=pending_tasks,
            risks=risks,
            unclear_items=unclear_items,
            sources=sources,
        )

    def _assess_confidence(self, item: dict) -> Literal["high", "medium", "low"]:
        """Assess confidence level of an extracted item."""
        text = item.get("text", "").lower()

        low_confidence_markers = ["?", "possibly", "maybe", "unclear", "unsure", "query"]
        for marker in low_confidence_markers:
            if marker in text:
                return "low"

        return "high"

    def _detect_urgency(self, text: str) -> Literal["routine", "soon", "urgent"]:
        """Detect urgency from text."""
        text_lower = text.lower()

        urgent_keywords = ["urgent", "immediately", "asap", "stat", "emergency"]
        for kw in urgent_keywords:
            if kw in text_lower:
                return "urgent"

        soon_keywords = ["soon", "today", "tomorrow", "priority"]
        for kw in soon_keywords:
            if kw in text_lower:
                return "soon"

        return "routine"

    def _detect_severity(self, text: str) -> Literal["low", "medium", "high"]:
        """Detect severity from risk text."""
        text_lower = text.lower()

        high_keywords = ["critical", "severe", "life-threatening", "emergency", "urgent"]
        for kw in high_keywords:
            if kw in text_lower:
                return "high"

        medium_keywords = ["significant", "important", "warning", "caution", "monitor"]
        for kw in medium_keywords:
            if kw in text_lower:
                return "medium"

        return "low"
