"""
Test the vision-first pipeline on real clinical documents.
"""

import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def run_vision_pipeline():
    print("=" * 70)
    print("Clinical State Compiler - Vision-First Pipeline")
    print("=" * 70)

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from src.ingestion import HybridProcessor, TextProcessor
    from src.models import SourceType

    # Initialize processors
    hybrid = HybridProcessor()
    text_proc = TextProcessor()

    # Collect all documents
    documents = []
    doc_id = 0

    # 1. Process the real handover image with vision
    print("\n[1] Processing real handover image with MedGemma Vision...")
    print("-" * 50)

    doc_id += 1
    handover_doc = hybrid.process(
        Path("test_data/real_handover.jpg"),
        f"DOC{doc_id:03d}",
        SourceType.HANDWRITTEN,
        use_vision=True,
    )
    documents.append(handover_doc)
    print(f"✓ [{handover_doc.id}] Handover sheet processed")
    print(f"  Extracted {len(handover_doc.content)} chars")

    # 2. Add lab results (text)
    print("\n[2] Processing lab results (text)...")
    doc_id += 1
    lab_doc = text_proc.process_document(
        Path("test_data/lab_results.txt"),
        f"DOC{doc_id:03d}",
        SourceType.LAB_RESULT,
    )
    documents.append(lab_doc)
    print(f"✓ [{lab_doc.id}] Lab results processed")

    # 3. Add radiology report (text)
    print("\n[3] Processing radiology report (text)...")
    doc_id += 1
    rad_doc = text_proc.process_document(
        Path("test_data/radiology_report.txt"),
        f"DOC{doc_id:03d}",
        SourceType.RADIOLOGY_REPORT,
    )
    documents.append(rad_doc)
    print(f"✓ [{rad_doc.id}] Radiology report processed")

    print(f"\n✓ Total documents: {len(documents)}")

    # Now compile with MedGemma
    print("\n" + "=" * 70)
    print("Compiling Clinical Snapshot")
    print("=" * 70)

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

    # Build compilation prompt
    combined_content = "\n\n---\n\n".join([
        f"[{doc.id} - {doc.source_type.value}]\n{doc.content[:2500]}"
        for doc in documents
    ])

    compile_prompt = f"""You are a Clinical State Compiler. Your job is to take multiple clinical documents and compile them into a single, authoritative clinical snapshot.

INPUT DOCUMENTS:
{combined_content}

Create a CLINICAL SNAPSHOT with these sections:

## Active Problems
List each active medical problem with current status. Number them.

## Current Status
2-3 sentence summary of where the patient is right now.

## Pending Actions
Bullet list of outstanding tasks, investigations, or decisions. Mark urgent items with [URGENT].

## Risks & Safety Concerns
Any flagged risks, allergies, or safety issues.

## Unclear / Missing Information
Anything ambiguous, conflicting, or not documented.

Be concise. Focus on what matters for clinical handover. Cite source documents where relevant."""

    print("\nGenerating clinical snapshot...")
    inputs = processor.tokenizer(compile_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1200,
            temperature=0.2,
            do_sample=True,
        )

    snapshot_content = processor.tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    # Build final output
    output = f"""# Clinical Snapshot
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Sources: {len(documents)}*

{snapshot_content}

---
## Source Documents
"""
    for doc in documents:
        output += f"- `[{doc.id}]` {doc.source_type.value}: {doc.filename or 'unnamed'}\n"

    print("\n" + "=" * 70)
    print("CLINICAL STATE COMPILER OUTPUT")
    print("=" * 70)
    print(output)

    # Save
    output_path = Path("test_data/vision_pipeline_output.md")
    output_path.write_text(output)
    print(f"\n✓ Saved to: {output_path}")

    return output


if __name__ == "__main__":
    run_vision_pipeline()
