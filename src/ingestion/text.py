"""Text document processing for typed notes, letters, and reports."""

from pathlib import Path
from typing import Union
from datetime import datetime
import re

from ..models import InputDocument, SourceType


class TextProcessor:
    """Process text-based clinical documents."""

    # Patterns to help identify document types
    TYPE_PATTERNS = {
        SourceType.DISCHARGE_SUMMARY: [
            r"discharge\s+summar",
            r"discharged?\s+(from|on|to)",
            r"admission\s+date.*discharge\s+date",
        ],
        SourceType.CLINIC_LETTER: [
            r"dear\s+(dr|doctor|colleague)",
            r"clinic\s+(letter|appointment|visit)",
            r"outpatient",
            r"follow[\s-]?up",
        ],
        SourceType.LAB_RESULT: [
            r"(lab|laboratory)\s+result",
            r"(haemoglobin|hb|wbc|platelets|creatinine|sodium|potassium)",
            r"reference\s+range",
            r"specimen",
        ],
        SourceType.RADIOLOGY_REPORT: [
            r"(x-?ray|ct\s+scan|mri|ultrasound|imaging)",
            r"(impression|findings|indication):",
            r"radiolog",
        ],
    }

    def detect_source_type(self, text: str) -> SourceType:
        """Attempt to detect document type from content."""
        text_lower = text.lower()

        for source_type, patterns in self.TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return source_type

        return SourceType.TYPED_NOTE

    def extract_metadata(self, text: str) -> dict:
        """Extract metadata like dates and authors from text."""
        metadata = {}

        # Try to find dates
        date_patterns = [
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})",
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["date_found"] = match.group(1)
                break

        # Try to find author/clinician
        author_patterns = [
            r"(?:signed|written|dictated|authored)\s+by[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(?:dr|doctor)[.\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        ]
        for pattern in author_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["author"] = match.group(1)
                break

        return metadata

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove excessive punctuation
        text = re.sub(r"[.]{3,}", "...", text)

        # Normalize common clinical abbreviations spacing
        # (keep them intact, just ensure spacing is consistent)

        return text.strip()

    def process_document(
        self,
        text_source: Union[str, Path, bytes, str],
        document_id: str,
        source_type: SourceType | None = None,
        filename: str | None = None,
        author: str | None = None,
        timestamp: datetime | None = None,
    ) -> InputDocument:
        """
        Process text and return an InputDocument.

        Args:
            text_source: File path, bytes, or raw text string
            document_id: Unique identifier for this document
            source_type: Document type (auto-detected if None)
            filename: Original filename
            author: Document author
            timestamp: Document timestamp
        """
        # Load text content
        if isinstance(text_source, Path) or (
            isinstance(text_source, str) and Path(text_source).exists()
        ):
            path = Path(text_source)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            filename = filename or path.name
        elif isinstance(text_source, bytes):
            content = text_source.decode("utf-8", errors="ignore")
        else:
            content = text_source

        # Clean text
        content = self.clean_text(content)

        # Auto-detect type if not provided
        if source_type is None:
            source_type = self.detect_source_type(content)

        # Extract metadata
        metadata = self.extract_metadata(content)
        if author is None:
            author = metadata.get("author")

        return InputDocument(
            id=document_id,
            source_type=source_type,
            content=content,
            filename=filename,
            author=author,
            timestamp=timestamp,
            confidence=1.0,  # Text documents have full confidence
        )
