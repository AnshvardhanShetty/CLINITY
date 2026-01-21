"""Data models for Clinical State Compiler."""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    """Types of input sources."""
    TYPED_NOTE = "typed_note"
    DICTATION = "dictation"
    HANDWRITTEN = "handwritten"
    CLINIC_LETTER = "clinic_letter"
    DISCHARGE_SUMMARY = "discharge_summary"
    LAB_RESULT = "lab_result"
    RADIOLOGY_REPORT = "radiology_report"
    MEDICAL_IMAGE = "medical_image"
    OTHER = "other"


class InputDocument(BaseModel):
    """A single input document to the compiler."""

    id: str
    source_type: SourceType
    content: str  # Extracted text content
    raw_content: Optional[bytes] = None  # Original file bytes (for images)
    timestamp: Optional[datetime] = None
    author: Optional[str] = None
    filename: Optional[str] = None
    confidence: float = 1.0  # OCR/ASR confidence score

    class Config:
        arbitrary_types_allowed = True


class ExtractedItem(BaseModel):
    """A single piece of extracted clinical information."""

    text: str
    category: str  # e.g., "problem", "medication", "investigation", "plan"
    source_id: str  # Links back to InputDocument.id
    source_excerpt: str  # The specific text this was extracted from
    confidence: Literal["high", "medium", "low"] = "medium"
    is_current: bool = True  # False if historical/resolved
    timestamp: Optional[datetime] = None


class PendingItem(BaseModel):
    """A pending task or investigation."""

    description: str
    source_id: str
    source_excerpt: str
    urgency: Literal["routine", "soon", "urgent"] = "routine"
    owner: Optional[str] = None  # Who is responsible


class RiskFlag(BaseModel):
    """A flagged risk or red flag."""

    description: str
    source_id: str
    source_excerpt: str
    severity: Literal["low", "medium", "high"] = "medium"


class UnclearItem(BaseModel):
    """Information that is ambiguous, conflicting, or missing."""

    description: str
    reason: Literal["ambiguous", "conflicting", "missing", "outdated"]
    source_ids: list[str] = Field(default_factory=list)
    source_excerpts: list[str] = Field(default_factory=list)


class ClinicalSnapshot(BaseModel):
    """The compiled clinical state output."""

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    input_document_count: int
    mode: str = "general"  # "handover", "discharge", "gp_summary", "general"

    # Core sections
    active_problems: list[ExtractedItem] = Field(default_factory=list)
    current_status: str = ""  # Free text summary of current state
    key_events: list[ExtractedItem] = Field(default_factory=list)  # Timeline
    pending_tasks: list[PendingItem] = Field(default_factory=list)
    risks: list[RiskFlag] = Field(default_factory=list)
    unclear_items: list[UnclearItem] = Field(default_factory=list)

    # Provenance
    sources: dict[str, str] = Field(default_factory=dict)  # id -> description

    def to_markdown(self) -> str:
        """Render snapshot as markdown document."""
        lines = [
            f"# Clinical Snapshot",
            f"*Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M')} | Mode: {self.mode} | Sources: {self.input_document_count}*",
            "",
        ]

        if self.active_problems:
            lines.append("## Active Problems")
            for i, p in enumerate(self.active_problems, 1):
                conf = "?" if p.confidence == "low" else ""
                lines.append(f"{i}. {p.text}{conf} `[{p.source_id}]`")
            lines.append("")

        if self.current_status:
            lines.append("## Current Status")
            lines.append(self.current_status)
            lines.append("")

        if self.key_events:
            lines.append("## Key Events")
            for e in self.key_events:
                ts = e.timestamp.strftime("%d/%m") if e.timestamp else "?"
                lines.append(f"- [{ts}] {e.text} `[{e.source_id}]`")
            lines.append("")

        if self.pending_tasks:
            lines.append("## Pending / Outstanding")
            for t in self.pending_tasks:
                urgency_marker = {"urgent": "!!", "soon": "!", "routine": ""}[t.urgency]
                lines.append(f"- {urgency_marker}{t.description} `[{t.source_id}]`")
            lines.append("")

        if self.risks:
            lines.append("## Risks / Red Flags")
            for r in self.risks:
                sev = {"high": "!!!", "medium": "!!", "low": "!"}[r.severity]
                lines.append(f"- {sev} {r.description} `[{r.source_id}]`")
            lines.append("")

        if self.unclear_items:
            lines.append("## Unclear / Missing Information")
            for u in self.unclear_items:
                lines.append(f"- **[{u.reason.upper()}]** {u.description}")
            lines.append("")

        if self.sources:
            lines.append("---")
            lines.append("### Source Key")
            for sid, desc in self.sources.items():
                lines.append(f"- `[{sid}]`: {desc}")

        return "\n".join(lines)
