"""
Clinical State Compiler - Main Application

Usage:
    python -m src.main compile --input <files> --mode <mode> --output <file>
    python -m src.main demo
"""

import argparse
from pathlib import Path
from datetime import datetime
import json

from .config import config
from .models import InputDocument, SourceType, ClinicalSnapshot
from .ingestion import OCRProcessor, AudioProcessor, TextProcessor
from .compiler import ClinicalCompiler


class ClinicalStateCompilerApp:
    """Main application for the Clinical State Compiler."""

    def __init__(self):
        self.ocr = OCRProcessor()
        self.audio = AudioProcessor()
        self.text = TextProcessor()
        self.compiler = ClinicalCompiler()
        self._doc_counter = 0

    def _generate_doc_id(self) -> str:
        """Generate a unique document ID."""
        self._doc_counter += 1
        return f"DOC{self._doc_counter:03d}"

    def ingest_file(self, file_path: Path) -> InputDocument:
        """Ingest a single file and return an InputDocument."""
        suffix = file_path.suffix.lower()
        doc_id = self._generate_doc_id()

        # Route to appropriate processor based on file type
        if suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            # Image - use OCR
            return self.ocr.process_document(
                file_path,
                doc_id,
                SourceType.HANDWRITTEN,
                file_path.name,
            )
        elif suffix in [".wav", ".mp3", ".m4a", ".flac", ".ogg"]:
            # Audio - use ASR
            return self.audio.process_document(
                file_path,
                doc_id,
                file_path.name,
            )
        elif suffix in [".txt", ".md"]:
            # Plain text
            return self.text.process_document(
                file_path,
                doc_id,
                filename=file_path.name,
            )
        else:
            # Try as text
            return self.text.process_document(
                file_path,
                doc_id,
                filename=file_path.name,
            )

    def ingest_text(self, text: str, source_type: SourceType = SourceType.TYPED_NOTE) -> InputDocument:
        """Ingest raw text directly."""
        doc_id = self._generate_doc_id()
        return self.text.process_document(text, doc_id, source_type)

    def compile(
        self,
        input_paths: list[Path],
        mode: str = "general",
    ) -> ClinicalSnapshot:
        """Compile multiple input files into a clinical snapshot."""
        documents = []
        for path in input_paths:
            if path.is_file():
                doc = self.ingest_file(path)
                documents.append(doc)
                print(f"Ingested: {path.name} -> {doc.id} ({doc.source_type.value})")
            elif path.is_dir():
                for file_path in path.iterdir():
                    if file_path.is_file() and not file_path.name.startswith("."):
                        doc = self.ingest_file(file_path)
                        documents.append(doc)
                        print(f"Ingested: {file_path.name} -> {doc.id}")

        if not documents:
            raise ValueError("No documents could be ingested")

        return self.compiler.compile(documents, mode)


def run_demo():
    """Run a demonstration with sample clinical notes."""
    print("=" * 60)
    print("Clinical State Compiler - Demo")
    print("=" * 60)
    print()

    app = ClinicalStateCompilerApp()

    # Sample clinical documents
    sample_documents = [
        {
            "type": SourceType.TYPED_NOTE,
            "content": """Ward Round Note - 15/01/2026
68M admitted with chest pain and SOB.
PMHx: T2DM, HTN, Previous MI 2019
Current problems:
1. NSTEMI - troponin rising, cardiology review requested
2. Acute kidney injury - Cr 180 (baseline 95), likely pre-renal
3. Hospital acquired pneumonia - started co-amox

Plan:
- Await cardiology opinion for ?angiogram
- IV fluids, monitor UO
- Chase blood cultures
- ECHO requested - not yet done

Risks: High bleeding risk on anticoagulation given AKI
?Allergic to penicillin - need to clarify with family
"""
        },
        {
            "type": SourceType.TYPED_NOTE,
            "content": """Nursing Handover 16/01/2026 07:00
Mr Smith, Bed 4
Stable overnight. O2 sats 94% on 2L.
BP 110/70, HR 88
Urine output improved - 45ml/hr overnight
Awaiting cardiology review - not seen yesterday
Blood cultures back - no growth so far
Family visiting at 10am - want update on plan
ECHO still outstanding
"""
        },
        {
            "type": SourceType.LAB_RESULT,
            "content": """Lab Results 16/01/2026
Sodium: 138 (135-145)
Potassium: 4.2 (3.5-5.0)
Creatinine: 165 (was 180 yesterday)
Troponin: 850 (was 650, rising)
CRP: 120 (was 95)
WBC: 14.2 (4-11) - elevated
"""
        },
    ]

    # Ingest documents
    print("Ingesting sample documents...")
    documents = []
    for i, sample in enumerate(sample_documents, 1):
        doc = app.ingest_text(sample["content"], sample["type"])
        documents.append(doc)
        print(f"  [{doc.id}] {doc.source_type.value}")

    print()
    print("Compiling clinical snapshot...")
    print("-" * 40)

    # For demo without actual MedGemma, create a mock snapshot
    # In production, this would call: app.compiler.compile(documents, "handover")

    demo_snapshot = create_demo_snapshot(documents)
    print()
    print(demo_snapshot.to_markdown())

    return demo_snapshot


def create_demo_snapshot(documents: list[InputDocument]) -> ClinicalSnapshot:
    """Create a demo snapshot without requiring MedGemma."""
    from .models import ExtractedItem, PendingItem, RiskFlag, UnclearItem

    return ClinicalSnapshot(
        generated_at=datetime.now(),
        input_document_count=len(documents),
        mode="handover",
        active_problems=[
            ExtractedItem(
                text="NSTEMI - troponin rising (850, was 650)",
                category="problem",
                source_id="DOC001",
                source_excerpt="NSTEMI - troponin rising",
                confidence="high",
                is_current=True,
            ),
            ExtractedItem(
                text="Acute kidney injury - Cr 165, improving (was 180)",
                category="problem",
                source_id="DOC001",
                source_excerpt="Acute kidney injury",
                confidence="high",
                is_current=True,
            ),
            ExtractedItem(
                text="Hospital acquired pneumonia - on co-amoxiclav",
                category="problem",
                source_id="DOC001",
                source_excerpt="Hospital acquired pneumonia",
                confidence="high",
                is_current=True,
            ),
        ],
        current_status="68M with NSTEMI (rising troponin), improving AKI, and HAP. Clinically stable overnight. Awaiting cardiology review and ECHO. Family visiting today.",
        key_events=[
            ExtractedItem(
                text="Admitted with chest pain and SOB",
                category="event",
                source_id="DOC001",
                source_excerpt="admitted with chest pain and SOB",
            ),
            ExtractedItem(
                text="Urine output improved overnight",
                category="event",
                source_id="DOC002",
                source_excerpt="Urine output improved - 45ml/hr",
            ),
        ],
        pending_tasks=[
            PendingItem(
                description="Cardiology review - requested but not yet seen",
                source_id="DOC001",
                source_excerpt="cardiology review requested",
                urgency="urgent",
            ),
            PendingItem(
                description="ECHO - requested, not yet done",
                source_id="DOC001",
                source_excerpt="ECHO requested - not yet done",
                urgency="soon",
            ),
            PendingItem(
                description="Chase blood culture results",
                source_id="DOC001",
                source_excerpt="Chase blood cultures",
                urgency="routine",
            ),
            PendingItem(
                description="Update family at 10am visit",
                source_id="DOC002",
                source_excerpt="Family visiting at 10am - want update",
                urgency="soon",
            ),
        ],
        risks=[
            RiskFlag(
                description="High bleeding risk if anticoagulated (given AKI)",
                source_id="DOC001",
                source_excerpt="High bleeding risk on anticoagulation given AKI",
                severity="high",
            ),
        ],
        unclear_items=[
            UnclearItem(
                description="Penicillin allergy status - needs clarification with family",
                reason="ambiguous",
                source_ids=["DOC001"],
                source_excerpts=["?Allergic to penicillin - need to clarify"],
            ),
        ],
        sources={
            "DOC001": "typed_note: Ward Round 15/01",
            "DOC002": "typed_note: Nursing Handover 16/01",
            "DOC003": "lab_result: Labs 16/01",
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Clinical State Compiler - Transform fragmented clinical information into a unified snapshot"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demonstration with sample data")

    # Compile command
    compile_parser = subparsers.add_parser("compile", help="Compile clinical documents")
    compile_parser.add_argument(
        "--input", "-i",
        nargs="+",
        type=Path,
        required=True,
        help="Input files or directories",
    )
    compile_parser.add_argument(
        "--mode", "-m",
        choices=["handover", "discharge", "gp_summary", "general"],
        default="general",
        help="Output mode",
    )
    compile_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file (markdown)",
    )

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "compile":
        app = ClinicalStateCompilerApp()
        snapshot = app.compile(args.input, args.mode)

        output_md = snapshot.to_markdown()
        if args.output:
            args.output.write_text(output_md)
            print(f"Snapshot saved to: {args.output}")
        else:
            print(output_md)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
