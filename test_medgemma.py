"""Test MedGemma model loading and basic inference."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_medgemma():
    print("=" * 60)
    print("MedGemma Test")
    print("=" * 60)

    # Check token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found in environment")
        return
    print(f"✓ HF_TOKEN found: {token[:10]}...")

    print("\nLoading MedGemma (this will download ~8GB on first run)...")
    print("This may take several minutes...\n")

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = "google/medgemma-4b-it"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    print("✓ Processor loaded")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device,
    )
    print("✓ Model loaded")

    # Test inference
    print("\n" + "-" * 40)
    print("Running test inference...")
    print("-" * 40)

    sample_note = """Ward round note:
72F admitted with shortness of breath and bilateral leg swelling.
PMHx: Heart failure (EF 35%), AF on warfarin, T2DM
Current meds: Furosemide 40mg BD, Bisoprolol 5mg OD, Ramipril 5mg OD
Impression: Acute decompensated heart failure, likely triggered by AF with fast ventricular rate.
Plan:
- IV furosemide 80mg stat then 40mg BD
- Daily weights and fluid balance
- Echo requested
- Cardiology review for rate control
Pending: INR result, renal function"""

    prompt = f"""You are a clinical assistant. Extract the active problems from this clinical note. List them as bullet points.

Clinical Note:
{sample_note}

Active Problems:"""

    inputs = processor.tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
        )

    input_len = inputs["input_ids"].shape[1]
    response = processor.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    print("\n" + "=" * 40)
    print("INPUT NOTE:")
    print("=" * 40)
    print(sample_note[:200] + "...")

    print("\n" + "=" * 40)
    print("MEDGEMMA RESPONSE:")
    print("=" * 40)
    print(response)

    print("\n✓ Test complete!")


if __name__ == "__main__":
    test_medgemma()
