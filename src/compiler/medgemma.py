"""MedGemma integration for clinical text understanding."""

from typing import Optional
import torch
from PIL import Image
import io

from ..config import config
from ..models import InputDocument


class MedGemmaProcessor:
    """Interface to MedGemma for clinical language understanding."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None

    def _get_device(self) -> str:
        """Determine best available device."""
        if config.model.medgemma_device != "auto":
            return config.model.medgemma_device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self):
        """Load MedGemma model."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoProcessor

        self._device = self._get_device()
        model_id = config.model.medgemma_model_id

        print(f"Loading MedGemma from {model_id} on {self._device}...")

        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            token=config.hf_token,
        )

        # Load model with quantization if requested
        model_kwargs = {
            "token": config.hf_token,
            "torch_dtype": torch.bfloat16 if self._device != "cpu" else torch.float32,
        }

        if config.model.medgemma_load_in_4bit and self._device == "cuda":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            model_kwargs["device_map"] = self._device

        self._model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        print("MedGemma loaded successfully")

    def _build_prompt(self, system: str, user: str, image: Optional[Image.Image] = None) -> dict:
        """Build a prompt for MedGemma."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        if image is not None:
            # For multimodal, prepend image token to user message
            messages[1]["content"] = [
                {"type": "image", "image": image},
                {"type": "text", "text": user},
            ]

        return messages

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a clinical assistant helping to extract and organize medical information.",
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Generate a response from MedGemma."""
        self.load()

        messages = self._build_prompt(system_prompt, prompt, image)

        # Process inputs
        if image is not None:
            inputs = self._processor(
                text=self._processor.apply_chat_template(messages, tokenize=False),
                images=image,
                return_tensors="pt",
            )
        else:
            inputs = self._processor(
                text=self._processor.apply_chat_template(messages, tokenize=False),
                return_tensors="pt",
            )

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._processor.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        response = self._processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        return response.strip()

    def extract_clinical_info(self, document: InputDocument) -> dict:
        """
        Extract structured clinical information from a document.

        Returns a dict with:
        - problems: list of identified problems/diagnoses
        - medications: list of medications
        - investigations: list of tests/results
        - plans: list of planned actions
        - risks: list of flagged risks
        - pending: list of outstanding items
        """
        system_prompt = """You are a clinical information extraction system. Your task is to extract structured information from clinical documents.

IMPORTANT RULES:
1. Only extract information explicitly stated in the text
2. Never invent or assume information
3. Flag anything unclear or ambiguous
4. Distinguish between CURRENT/ACTIVE issues and HISTORICAL/RESOLVED issues
5. Preserve clinical terminology exactly as written"""

        extraction_prompt = f"""Extract structured clinical information from this document.

DOCUMENT TYPE: {document.source_type.value}
SOURCE ID: {document.id}

DOCUMENT CONTENT:
{document.content}

Respond in this exact format:

ACTIVE_PROBLEMS:
- [problem 1]
- [problem 2]

HISTORICAL_PROBLEMS:
- [resolved/historical problem]

MEDICATIONS:
- [medication with dose if stated]

INVESTIGATIONS:
- [test: result if available]

PLANS:
- [planned action]

PENDING:
- [outstanding task or investigation]

RISKS:
- [any flagged risks or red flags]

UNCLEAR:
- [anything ambiguous or missing]

If a section has no items, write "None identified" for that section."""

        # Check if document has an associated image
        image = None
        if document.raw_content and document.source_type in [
            "medical_image",
            "handwritten",
        ]:
            try:
                image = Image.open(io.BytesIO(document.raw_content))
            except Exception:
                pass

        response = self.generate(extraction_prompt, system_prompt, image)
        return self._parse_extraction_response(response, document.id)

    def _parse_extraction_response(self, response: str, source_id: str) -> dict:
        """Parse the structured extraction response."""
        sections = {
            "active_problems": [],
            "historical_problems": [],
            "medications": [],
            "investigations": [],
            "plans": [],
            "pending": [],
            "risks": [],
            "unclear": [],
        }

        current_section = None
        section_map = {
            "ACTIVE_PROBLEMS": "active_problems",
            "HISTORICAL_PROBLEMS": "historical_problems",
            "MEDICATIONS": "medications",
            "INVESTIGATIONS": "investigations",
            "PLANS": "plans",
            "PENDING": "pending",
            "RISKS": "risks",
            "UNCLEAR": "unclear",
        }

        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check for section header
            header = line.rstrip(":").upper()
            if header in section_map:
                current_section = section_map[header]
                continue

            # Parse item
            if current_section and line.startswith("-"):
                item = line[1:].strip()
                if item.lower() != "none identified" and item:
                    sections[current_section].append({
                        "text": item,
                        "source_id": source_id,
                    })

        return sections

    def synthesize_status(self, documents: list[InputDocument], extracted_data: list[dict]) -> str:
        """
        Synthesize a current status summary from multiple documents.
        """
        system_prompt = """You are a clinical summarization system. Create a brief, focused summary of the patient's CURRENT clinical state.

RULES:
1. Focus on what matters RIGHT NOW
2. Be concise (2-3 sentences max)
3. Prioritize active problems and immediate concerns
4. Do not include historical information unless directly relevant
5. Do not make assumptions or add information not in the sources"""

        # Compile all active problems and recent events
        all_problems = []
        all_pending = []
        for data in extracted_data:
            all_problems.extend([p["text"] for p in data.get("active_problems", [])])
            all_pending.extend([p["text"] for p in data.get("pending", [])])

        prompt = f"""Based on the following extracted information, write a 2-3 sentence summary of the patient's current clinical state.

ACTIVE PROBLEMS:
{chr(10).join(f'- {p}' for p in all_problems) if all_problems else 'None identified'}

PENDING ITEMS:
{chr(10).join(f'- {p}' for p in all_pending) if all_pending else 'None'}

Write a brief current status summary:"""

        return self.generate(prompt, system_prompt, max_new_tokens=200)

    def resolve_conflicts(
        self, items: list[dict], category: str
    ) -> tuple[list[dict], list[dict]]:
        """
        Identify and flag conflicting information.

        Returns:
            Tuple of (resolved_items, conflicts)
        """
        if len(items) <= 1:
            return items, []

        system_prompt = """You are a clinical data reconciliation system. Identify any conflicting or contradictory information between sources."""

        items_text = "\n".join(
            f"[{item['source_id']}]: {item['text']}" for item in items
        )

        prompt = f"""Review these {category} items from different sources and identify any conflicts or contradictions.

ITEMS:
{items_text}

For each item, respond with either:
KEEP: [item text] - if the item should be kept
CONFLICT: [item text] | [conflicting item text] | [reason] - if there's a contradiction

List your decisions:"""

        response = self.generate(prompt, system_prompt, max_new_tokens=500)

        # Parse response (simplified - real implementation would be more robust)
        resolved = []
        conflicts = []

        for item in items:
            if f"CONFLICT:" in response and item["text"] in response:
                conflicts.append(item)
            else:
                resolved.append(item)

        return resolved, conflicts
