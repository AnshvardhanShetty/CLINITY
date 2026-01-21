"""
Full Pipeline Test for Clinical State Compiler

Tests:
1. Text document ingestion
2. OCR on handwritten note image
3. MedGemma extraction
4. Clinical snapshot compilation
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def test_ocr_only():
    """Test OCR on the handwritten note."""
    print("\n" + "=" * 60)
    print("TEST 1: OCR Pipeline")
    print("=" * 60)

    from src.ingestion.ocr import OCRProcessor

    ocr = OCRProcessor()
    image_path = Path("test_data/handwritten_note.png")

    print(f"Processing: {image_path}")
    text, confidence = ocr.extract_text(image_path)

    print(f"\nExtracted text (confidence: {confidence:.2f}):")
    print("-" * 40)
    print(text)
    print("-" * 40)

    return text, confidence


def test_full_pipeline():
    """Test the full compilation pipeline with real MedGemma."""
    print("\n" + "=" * 60)
    print("TEST 2: Full Pipeline with MedGemma")
    print("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from src.models import InputDocument, SourceType, ClinicalSnapshot, ExtractedItem, PendingItem, RiskFlag, UnclearItem
    from src.ingestion import TextProcessor, OCRProcessor
    from datetime import datetime

    # Initialize processors
    text_processor = TextProcessor()
    ocr_processor = OCRProcessor()

    # Load all test documents
    print("\nLoading test documents...")
    documents = []
    doc_counter = 0

    test_files = [
        ("test_data/ward_round_note.txt", SourceType.TYPED_NOTE),
        ("test_data/nursing_handover.txt", SourceType.TYPED_NOTE),
        ("test_data/lab_results.txt", SourceType.LAB_RESULT),
        ("test_data/radiology_report.txt", SourceType.RADIOLOGY_REPORT),
    ]

    for filepath, source_type in test_files:
        doc_counter += 1
        doc = text_processor.process_document(
            Path(filepath),
            f"DOC{doc_counter:03d}",
            source_type,
        )
        documents.append(doc)
        print(f"  [{doc.id}] {source_type.value}: {filepath}")

    # Add OCR document
    doc_counter += 1
    ocr_doc = ocr_processor.process_document(
        Path("test_data/handwritten_note.png"),
        f"DOC{doc_counter:03d}",
        SourceType.HANDWRITTEN,
    )
    documents.append(ocr_doc)
    print(f"  [{ocr_doc.id}] handwritten (OCR conf: {ocr_doc.confidence:.2f}): handwritten_note.png")

    print(f"\nTotal documents: {len(documents)}")

    # Load MedGemma
    print("\nLoading MedGemma...")
    token = os.environ.get("HF_TOKEN")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model_id = "google/medgemma-4b-it"
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    print(f"✓ MedGemma loaded on {device}")

    # Process each document with MedGemma
    print("\nExtracting clinical information...")
    all_extractions = []

    for doc in documents:
        print(f"\n  Processing {doc.id} ({doc.source_type.value})...")

        prompt = f"""You are a clinical information extraction system. Extract structured information from this clinical document.

DOCUMENT TYPE: {doc.source_type.value}

DOCUMENT CONTENT:
{doc.content[:2000]}

Extract and list:
1. ACTIVE PROBLEMS (current medical issues)
2. PENDING TASKS (outstanding investigations or actions)
3. RISKS (safety concerns or red flags)
4. UNCLEAR ITEMS (ambiguous or missing information)

Format your response as:
ACTIVE_PROBLEMS:
- [item]

PENDING:
- [item]

RISKS:
- [item]

UNCLEAR:
- [item]

If a section has no items, write "None identified"."""

        inputs = processor.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
            )

        response = processor.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        print(f"    Response preview: {response[:100]}...")

        all_extractions.append({
            "doc_id": doc.id,
            "doc_type": doc.source_type.value,
            "response": response,
        })

    # Synthesize into clinical snapshot
    print("\n" + "-" * 40)
    print("Synthesizing Clinical Snapshot...")
    print("-" * 40)

    # Combine all document contents for synthesis
    combined_context = "\n\n".join([
        f"[{doc.id} - {doc.source_type.value}]:\n{doc.content[:1000]}"
        for doc in documents
    ])

    synthesis_prompt = f"""Based on the following clinical documents for the same patient, create a concise clinical summary.

{combined_context}

Create a summary with these sections:
1. PATIENT SUMMARY (1-2 sentences: who is this patient and why are they here)
2. ACTIVE PROBLEMS (numbered list of current issues)
3. CURRENT STATUS (brief description of how they are right now)
4. PENDING ACTIONS (what needs to be done)
5. KEY RISKS (safety concerns)
6. UNCLEAR/MISSING (information gaps)

Be concise and focus on what matters for clinical handover."""

    inputs = processor.tokenizer(synthesis_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.2,
            do_sample=True,
        )

    synthesis = processor.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Build final output
    print("\n" + "=" * 60)
    print("CLINICAL STATE COMPILER OUTPUT")
    print("=" * 60)

    output = f"""# Clinical Snapshot
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Mode: handover | Sources: {len(documents)}*

{synthesis}

---
## Source Documents
"""
    for doc in documents:
        conf_str = f" (OCR conf: {doc.confidence:.0%})" if doc.source_type == SourceType.HANDWRITTEN else ""
        output += f"- `[{doc.id}]` {doc.source_type.value}: {doc.filename or 'unnamed'}{conf_str}\n"

    print(output)

    # Save output
    output_path = Path("test_data/output_snapshot.md")
    output_path.write_text(output)
    print(f"\n✓ Output saved to: {output_path}")

    return output


def main():
    print("=" * 60)
    print("Clinical State Compiler - Full Pipeline Test")
    print("=" * 60)

    # Test 1: OCR
    ocr_text, ocr_confidence = test_ocr_only()

    # Test 2: Full pipeline
    snapshot = test_full_pipeline()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
