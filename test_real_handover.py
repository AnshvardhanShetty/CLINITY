"""Test OCR and MedGemma on real clinical handover image."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def test_real_handover():
    print("=" * 70)
    print("Testing Real Clinical Handover Image")
    print("=" * 70)

    # Step 1: OCR
    print("\n[STEP 1] Running OCR...")
    print("-" * 50)

    from src.ingestion.ocr import OCRProcessor
    from PIL import Image

    ocr = OCRProcessor()
    image_path = Path("test_data/real_handover.jpg")

    # Run OCR
    text, confidence = ocr.extract_text(image_path, preprocess=True)

    print(f"OCR Confidence: {confidence:.1%}")
    print(f"\nExtracted Text ({len(text)} chars):")
    print("-" * 50)
    print(text)
    print("-" * 50)

    # Step 2: MedGemma Analysis
    print("\n[STEP 2] MedGemma Analysis...")
    print("-" * 50)

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

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
    print(f"MedGemma loaded on {device}")

    # Analyze OCR text
    prompt = f"""You are a clinical information extraction system. This is OCR output from a handwritten clinical handover sheet. Extract all patient information you can identify.

OCR TEXT:
{text}

For each patient you can identify, extract:
1. Patient identifier (bed/HDU number)
2. Demographics (age, sex)
3. Admission reason
4. Past medical history
5. Current issues
6. Pending tasks
7. Lines/access

Format clearly. Note any text that is unclear or potentially misread by OCR."""

    inputs = processor.tokenizer(prompt, return_tensors="pt").to(device)

    print("Generating analysis...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.1,
            do_sample=True,
        )

    response = processor.tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    print("\nMedGemma Extraction from OCR:")
    print("=" * 50)
    print(response)

    # Step 3: Compare with ground truth (what I can read from the image)
    print("\n" + "=" * 70)
    print("GROUND TRUTH (manually read from image)")
    print("=" * 70)

    ground_truth = """
HDU1: Male, 84yo, Day 2
- Admitted 01/01/2026 17:49 from A&E
- Chest pain 3/7, SOB, feeling unwell, lethargic
- Slip from chair onto floor
- BP very low 68/35 - admitted for inotropes
- PMH: AF, HTN, T2DM, IHD (post cardiac stents 20yrs ago), Heart failure, High BMI, OSA, CKD stage 4, COPD
- Lines: Art and CVC
- Handwritten: ECHO, USS-HB, Furosemide, various notes

HDU2: Female, 73yo, Day 1
- Admitted 02/01/2026 01:23 from A&E
- 1/7 history of SOB, dry cough, fever
- PMH: Recent MI with PCI @ St Georges, chest sepsis October, COPD, HFrEF (LVEF 30%), T2DM, CABG 2022, Limb ischaemia R leg (previous stents), HTN, ACS, AKI stage 1, hypercholesterolaemia
- Allergies: Nicorette, Ramipril
- Lines: Arterial line
- Handwritten: Amox/Clari, various notes

HDU3: Male, 61yo, Day 3
- Admitted 31/12/2025 20:26 to HDU from A&E
- DKA - Severe Metabolic Acidosis and AKI
- PMH: HTN, DM type 2 (poorly controlled), Lichen Planus, Liver Ca (on immunotherapy), Left eye CA (proton beam therapy)
- Lines: Art
"""
    print(ground_truth)

    return text, response


if __name__ == "__main__":
    test_real_handover()
