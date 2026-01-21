"""
Production-quality Clinical State Compiler Pipeline

Features:
- Vision-first document processing
- Multi-pass extraction with verification
- Safety-critical item detection
- Confidence scoring
- Provenance tracking
"""

import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def run_production_pipeline():
    print("=" * 70)
    print("Clinical State Compiler - Production Pipeline")
    print("=" * 70)

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from PIL import Image

    from src.ingestion import TextProcessor
    from src.models import SourceType, InputDocument
    from src.compiler.safety import SafetyChecker
    from src.compiler.multipass import MultiPassExtractor

    # Load model once
    token = os.environ.get("HF_TOKEN")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "google/medgemma-4b-it"

    print(f"\nLoading MedGemma on {device}...")
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    print("✓ Model loaded")

    # Initialize components
    text_proc = TextProcessor()
    safety_checker = SafetyChecker()
    extractor = MultiPassExtractor(model, processor, device)

    documents = []
    all_extractions = []
    all_safety_alerts = []
    all_missing = []

    # === STEP 1: Process the real handover image with vision ===
    print("\n" + "=" * 50)
    print("STEP 1: Vision Processing - Handover Image")
    print("=" * 50)

    image_path = Path("test_data/real_handover.jpg")
    image = Image.open(image_path)

    vision_prompt = """Extract all patient information from this clinical handover sheet.

For EACH patient (HDU1, HDU2, HDU3), extract:
- Patient identifier and demographics (age, sex, day)
- Admission date and reason
- Past medical history (PMH)
- Current status and issues
- Lines and access
- Allergies
- Any pending tasks or plans
- Any handwritten annotations

Be thorough. Include both typed and handwritten text."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": vision_prompt}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    print("Processing image...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
            temperature=0.1,
            do_sample=True,
        )

    vision_content = processor.tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    # Create document from vision output
    handover_doc = InputDocument(
        id="DOC001",
        source_type=SourceType.HANDWRITTEN,
        content=vision_content,
        filename="real_handover.jpg",
        confidence=0.85,
    )
    documents.append(handover_doc)
    print(f"✓ Extracted {len(vision_content)} chars from handover image")

    # Run safety check on vision output
    safety_alerts = safety_checker.check_text(vision_content, "DOC001")
    all_safety_alerts.extend(safety_alerts)
    print(f"✓ Found {len(safety_alerts)} safety alerts")

    # === STEP 2: Process text documents with multi-pass extraction ===
    print("\n" + "=" * 50)
    print("STEP 2: Multi-Pass Extraction - Text Documents")
    print("=" * 50)

    text_files = [
        ("test_data/ward_round_note.txt", SourceType.TYPED_NOTE, "DOC002"),
        ("test_data/nursing_handover.txt", SourceType.TYPED_NOTE, "DOC003"),
        ("test_data/lab_results.txt", SourceType.LAB_RESULT, "DOC004"),
    ]

    for filepath, source_type, doc_id in text_files:
        print(f"\nProcessing: {filepath}")
        doc = text_proc.process_document(Path(filepath), doc_id, source_type)
        documents.append(doc)

        # Multi-pass extraction
        extraction_result = extractor.extract(doc)
        all_extractions.append(extraction_result)
        all_safety_alerts.extend(extraction_result.safety_alerts)
        all_missing.extend(extraction_result.missing_mandatory)

        print(f"  ✓ {len(extraction_result.extractions)} extractions")
        print(f"  ✓ {len(extraction_result.safety_alerts)} safety alerts")

    # === STEP 3: Compile Final Snapshot ===
    print("\n" + "=" * 50)
    print("STEP 3: Compiling Clinical Snapshot")
    print("=" * 50)

    # Deduplicate safety alerts
    seen_alerts = set()
    unique_alerts = []
    for alert in all_safety_alerts:
        key = (alert.category, alert.description)
        if key not in seen_alerts:
            seen_alerts.add(key)
            unique_alerts.append(alert)

    # Build the final output
    output_lines = [
        "# Clinical Snapshot",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Sources: {len(documents)}*",
        "",
    ]

    # Safety alerts box (always first)
    if unique_alerts:
        output_lines.append("## ⚠️ SAFETY ALERTS")
        output_lines.append("")
        for alert in sorted(unique_alerts, key=lambda a: {"critical": 0, "high": 1, "medium": 2}[a.severity]):
            severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡"}[alert.severity]
            output_lines.append(f"- {severity_icon} **{alert.description}** `[{alert.source_id}]`")
        output_lines.append("")

    # Missing mandatory items
    unique_missing = list(set(all_missing))
    if unique_missing:
        output_lines.append("## ⚡ MISSING DOCUMENTATION")
        output_lines.append("")
        for item in unique_missing:
            output_lines.append(f"- ❓ {item}")
        output_lines.append("")

    # Vision extraction (patient summaries from handover)
    output_lines.append("## Patient Information (from Handover Sheet)")
    output_lines.append("")
    output_lines.append(vision_content[:2000])  # Truncate if needed
    output_lines.append("")

    # Verified extractions by category
    output_lines.append("## Verified Extractions")
    output_lines.append("")

    for result in all_extractions:
        if result.extractions:
            output_lines.append(f"### From {result.document_id} ({result.document_type})")
            output_lines.append("")

            # Group by category
            by_category = {}
            for ext in result.extractions:
                by_category.setdefault(ext.category, []).append(ext)

            for category, items in by_category.items():
                output_lines.append(f"**{category.title()}s:**")
                for item in items:
                    conf_bar = "█" * int(item.confidence * 5) + "░" * (5 - int(item.confidence * 5))
                    conf_pct = f"{item.confidence*100:.0f}%"
                    verified = "✓" if item.is_verified else "?"
                    output_lines.append(f"- {verified} {item.text} [{conf_bar} {conf_pct}] `[{item.source_id}]`")
                output_lines.append("")

    # Source documents
    output_lines.append("---")
    output_lines.append("## Source Documents")
    output_lines.append("")
    for doc in documents:
        output_lines.append(f"- `[{doc.id}]` {doc.source_type.value}: {doc.filename or 'unnamed'}")

    output = "\n".join(output_lines)

    print("\n" + "=" * 70)
    print("CLINICAL STATE COMPILER OUTPUT")
    print("=" * 70)
    print(output)

    # Save
    output_path = Path("test_data/production_output.md")
    output_path.write_text(output)
    print(f"\n✓ Saved to: {output_path}")

    return output


if __name__ == "__main__":
    run_production_pipeline()
