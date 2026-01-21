"""Test MedGemma's multimodal vision capability on clinical handover image."""

import os
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

def test_vision_direct():
    print("=" * 70)
    print("MedGemma Vision Test - Direct Image Analysis")
    print("=" * 70)

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    token = os.environ.get("HF_TOKEN")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "google/medgemma-4b-it"

    print(f"\nLoading MedGemma multimodal on {device}...")

    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    print("✓ Model loaded")

    # Load the real handover image
    image_path = Path("test_data/real_handover.jpg")
    image = Image.open(image_path)
    print(f"✓ Image loaded: {image.size}")

    # Create prompt for clinical extraction
    prompt = """Look at this clinical handover sheet image. Extract all patient information you can see.

For each patient (HDU1, HDU2, HDU3), provide:
1. Demographics (age, sex, day of admission)
2. Reason for admission
3. Past medical history (PMH)
4. Current issues/status
5. Lines and access
6. Any pending tasks or plans (including handwritten notes)
7. Allergies if mentioned

Be thorough and include both typed and handwritten information visible in the image."""

    # Format for multimodal input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Process with the multimodal processor
    print("\nProcessing image with MedGemma vision...")

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    print("Generating analysis (this may take a minute)...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=0.1,
            do_sample=True,
        )

    # Decode response
    response = processor.tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    print("\n" + "=" * 70)
    print("MEDGEMMA VISION OUTPUT")
    print("=" * 70)
    print(response)

    # Ground truth comparison
    print("\n" + "=" * 70)
    print("GROUND TRUTH (for comparison)")
    print("=" * 70)
    print("""
HDU1: Male, 84yo, Day 2
- Admitted 01/01/2026 17:49 from A&E
- Chest pain 3/7, SOB, feeling unwell, lethargic, slip from chair
- BP very low 68/35 - admitted for inotropes
- PMH: AF, HTN, T2DM, IHD (stents 20yrs ago), Heart failure, High BMI, OSA, CKD stage 4, COPD
- Lines: Art and CVC
- Handwritten: ECHO, USS-HB, Furosemide, ?Ca Pancreas, CT, MRCP
- Status: DNAR

HDU2: Female, 73yo, Day 1
- Admitted 02/01/2026 01:23 from A&E
- 1/7 hx SOB, dry cough, fever
- PMH: Recent MI (PCI @ St Georges), chest sepsis Oct, COPD, HFrEF LVEF 30%, T2DM, CABG 2022, Limb ischaemia R leg, HTN, AKI stage 1, hypercholesterolaemia
- Allergies: Nicorette, Ramipril
- Lines: Arterial line
- Handwritten: Amox/Clari (antibiotics)

HDU3: Male, 61yo, Day 3
- Admitted 31/12/2025 20:26 from A&E
- DKA - Severe Metabolic Acidosis and AKI
- PMH: HTN, DM type 2 poorly controlled, Lichen Planus, Liver Ca (immunotherapy), Left eye CA (proton beam therapy)
- Lines: Art
- Handwritten: Diosite?, W1
""")

    return response


if __name__ == "__main__":
    test_vision_direct()
