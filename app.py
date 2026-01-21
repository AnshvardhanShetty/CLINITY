"""
CLINITY - Clinical Intelligence Report Generator
"""

import gradio as gr
import os
import re
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import tempfile

load_dotenv()

MODEL = None
PROCESSOR = None
DEVICE = None


def load_model():
    global MODEL, PROCESSOR, DEVICE
    if MODEL is not None:
        return MODEL, PROCESSOR, DEVICE

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    token = os.environ.get("HF_TOKEN")
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "google/medgemma-4b-it"

    print(f"Loading MedGemma on {DEVICE}...")
    PROCESSOR = AutoProcessor.from_pretrained(model_id, token=token)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_id, token=token, torch_dtype=torch.bfloat16, device_map=DEVICE,
    )
    print("Model loaded")
    return MODEL, PROCESSOR, DEVICE


def clean_output(text):
    lines = text.split('\n')
    seen = set()
    cleaned = []
    repeat_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped in ['* "No"', '- "No"', '"No"', '* No', '- No', 'No']:
            repeat_count += 1
            if repeat_count > 2:
                continue
        else:
            repeat_count = 0
        line_key = stripped.lower()
        if line_key and line_key in seen and len(line_key) < 50:
            continue
        if stripped:
            seen.add(line_key)
            if len(seen) > 20:
                seen = set(list(seen)[-20:])
        cleaned.append(line)
    return '\n'.join(cleaned)


def parse_patients(content):
    patients = []
    sections = re.split(r'(?=PATIENT\s*[\[\(]?\d|Patient\s*[\[\(]?\d|HDU\s*\d|Bed\s*\d)', content, flags=re.IGNORECASE)

    for section in sections:
        if not section.strip() or len(section.strip()) < 20:
            continue

        patient = {
            'id': '', 'name': '', 'age': '', 'sex': '', 'dob': '',
            'admission_date': '', 'admission_reason': '', 'pmh': [],
            'current_issues': '', 'allergies': '', 'resus_status': '',
            'lines': '', 'medications': '', 'observations': '', 'plan': '',
            'investigations': '', 'fluid_balance': '',
        }

        id_match = re.search(r'(HDU\s*\d+|Patient\s*[\[\(]?\d+[\]\)]?|Bed\s*\d+)', section, re.IGNORECASE)
        if id_match:
            patient['id'] = id_match.group(1).strip()

        name_match = re.search(r'(?:Name|Patient Name)[:\s]+([A-Za-z\s]+?)(?=\n|,|Age|DOB|$)', section, re.IGNORECASE)
        if name_match:
            patient['name'] = name_match.group(1).strip()[:50]

        age_match = re.search(r'Age[:\s]+(\d+)|(\d+)\s*(?:yo|y/?o|years?\s*old)', section, re.IGNORECASE)
        if age_match:
            patient['age'] = age_match.group(1) or age_match.group(2)

        sex_match = re.search(r'Sex[:\s]+([MF])|([Mm]ale|[Ff]emale)', section)
        if sex_match:
            sex_val = sex_match.group(1) or sex_match.group(2)
            patient['sex'] = 'Male' if sex_val and sex_val.lower() in ['m', 'male'] else 'Female'

        reason_match = re.search(r'(?:Admission|Presenting Complaint|Reason|Chief Complaint|PC)[:\s]+(.+?)(?=\n[-*]|\n\n|PMH|Past|Allerg|Meds|$)', section, re.IGNORECASE | re.DOTALL)
        if reason_match:
            patient['admission_reason'] = re.sub(r'\s+', ' ', reason_match.group(1).strip())[:500]

        pmh_match = re.search(r'(?:PMH|Past Medical History|Medical History|Background)[:\s]+(.+?)(?=\n\n|Current|Lines|Allerg|Resus|Meds|Obs|$)', section, re.IGNORECASE | re.DOTALL)
        if pmh_match:
            pmh_items = re.split(r'[,;]|\n[-*•]|\d+\.', pmh_match.group(1))
            patient['pmh'] = [item.strip() for item in pmh_items if item.strip() and len(item.strip()) > 2][:15]

        allergy_match = re.search(r'Allerg(?:y|ies)[:\s]+(.+?)(?=\n|Resus|Lines|Meds|PMH|$)', section, re.IGNORECASE)
        if allergy_match:
            patient['allergies'] = allergy_match.group(1).strip()[:300]

        resus_match = re.search(r'(?:Resus|DNAR|DNR|Escalation|Code Status|Ceiling)[:\s]*(.+?)(?=\n|$)', section, re.IGNORECASE)
        if resus_match:
            patient['resus_status'] = resus_match.group(1).strip()[:200]

        if 'DNAR' in section.upper() or 'DNR' in section.upper():
            if not patient['resus_status']:
                patient['resus_status'] = 'DNAR'

        lines_match = re.search(r'(?:Lines?|IV Access|Access|Cannula)[:\s]+(.+?)(?=\n\n|Allerg|Tasks|Meds|Obs|$)', section, re.IGNORECASE | re.DOTALL)
        if lines_match:
            patient['lines'] = re.sub(r'\n+', ', ', lines_match.group(1).strip())[:300]

        issues_match = re.search(r'(?:Current Issues?|Key Issues?|Active Problems?|Problems?|Issues)[:\s]+(.+?)(?=\n\n|Tasks|Plan|Meds|$)', section, re.IGNORECASE | re.DOTALL)
        if issues_match:
            patient['current_issues'] = re.sub(r'\n+', '; ', issues_match.group(1).strip())[:500]

        meds_match = re.search(r'(?:Meds|Medications?|Current Meds|Drugs?|Rx)[:\s]+(.+?)(?=\n\n|Plan|Tasks|Obs|$)', section, re.IGNORECASE | re.DOTALL)
        if meds_match:
            patient['medications'] = re.sub(r'\n+', ', ', meds_match.group(1).strip())[:500]

        obs_match = re.search(r'(?:Obs|Observations?|Vitals?|NEWS|EWS)[:\s]+(.+?)(?=\n\n|Plan|Meds|$)', section, re.IGNORECASE | re.DOTALL)
        if obs_match:
            patient['observations'] = re.sub(r'\n+', ', ', obs_match.group(1).strip())[:300]

        plan_match = re.search(r'(?:Plan|Tasks?|To Do|Action|Jobs|Outstanding)[:\s]+(.+?)(?=\n\n[-]|$)', section, re.IGNORECASE | re.DOTALL)
        if plan_match:
            patient['plan'] = re.sub(r'\n+', '; ', plan_match.group(1).strip())[:500]

        inv_match = re.search(r'(?:Investigations?|Inv|Bloods?|Results?)[:\s]+(.+?)(?=\n\n|Plan|$)', section, re.IGNORECASE | re.DOTALL)
        if inv_match:
            patient['investigations'] = re.sub(r'\n+', ', ', inv_match.group(1).strip())[:400]

        if patient['age'] or patient['admission_reason'] or patient['pmh'] or patient['current_issues']:
            patients.append(patient)

    if not patients and content.strip():
        return [{
            'id': 'Patient 1', 'name': '', 'age': '', 'sex': '',
            'admission_reason': content[:800],
            'pmh': [], 'allergies': '', 'resus_status': '', 'lines': '',
            'current_issues': '', 'medications': '', 'observations': '',
            'plan': '', 'investigations': '', 'dob': '', 'admission_date': '',
            'fluid_balance': '',
        }]

    return patients


def process_image(image):
    import torch
    model, processor, device = load_model()

    prompt = """You are a medical transcription expert. Transcribe EVERY piece of text from this clinical handover document.
Include ALL: patient identifiers, demographics, dates, diagnoses, medications (with doses), allergies, observations, vital signs, investigation results, plans, and any other clinical information.
Be thorough and accurate. Do not skip or summarize anything."""

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=3000, temperature=0.1, do_sample=True, repetition_penalty=1.1)
    raw_text = processor.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    structure_prompt = f"""From the following clinical text, extract and organize ALL information for EACH patient:

{raw_text}

Format for EACH patient:
PATIENT [Number/ID]:
- Name: [if available]
- Age: [years]
- Sex: [Male/Female]
- Presenting Complaint/Admission Reason: [full details]
- Past Medical History: [complete list]
- Allergies: [list all, or NKDA if none]
- Resuscitation Status: [Full/DNAR/etc]
- Current Medications: [list with doses if available]
- IV Lines/Access: [describe all]
- Observations/Vitals: [all values]
- Current Issues/Problems: [list all active issues]
- Investigations/Results: [any blood results, imaging, etc]
- Plan/Tasks: [all planned actions]

Extract EVERYTHING. Do not summarize or omit details."""

    inputs2 = processor.tokenizer(structure_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs2 = model.generate(**inputs2, max_new_tokens=2500, temperature=0.1, do_sample=True, repetition_penalty=1.2)
    return clean_output(processor.tokenizer.decode(outputs2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True))


def process_text(text_content):
    import torch
    model, processor, device = load_model()

    prompt = f"""From the following clinical text, extract and organize ALL information for EACH patient:

{text_content[:6000]}

Format for EACH patient:
PATIENT [Number/ID]:
- Name: [if available]
- Age: [years]
- Sex: [Male/Female]
- Presenting Complaint/Admission Reason: [full details]
- Past Medical History: [complete list]
- Allergies: [list all, or NKDA if none]
- Resuscitation Status: [Full/DNAR/etc]
- Current Medications: [list with doses]
- IV Lines/Access: [describe all]
- Observations/Vitals: [all values]
- Current Issues/Problems: [list all]
- Investigations/Results: [bloods, imaging, etc]
- Plan/Tasks: [all actions]

Extract EVERYTHING. Do not summarize."""

    inputs = processor.tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2500, temperature=0.1, do_sample=True, repetition_penalty=1.2)
    return clean_output(processor.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


def process_pdf(pdf_path):
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n\n"
        doc.close()
        if text.strip():
            return process_text(text)
        doc = fitz.open(pdf_path)
        pix = doc[0].get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return process_image(img)
    except Exception as e:
        return f"Error processing PDF: {str(e)}"


def generate_html(patients):
    timestamp = datetime.now().strftime("%d %B %Y at %H:%M")

    html = f'''<div style="font-family:'Inter',system-ui,sans-serif;background:rgba(255,255,255,0.02);border:1px solid #333;border-radius:20px;padding:40px;margin:20px auto;max-width:900px;">
    <div style="text-align:center;border-bottom:1px solid #333;padding-bottom:30px;margin-bottom:30px;">
        <div style="font-size:11px;font-weight:600;letter-spacing:6px;color:#666;margin-bottom:10px;">CLINITY</div>
        <h1 style="font-size:28px;font-weight:600;color:#fff;margin:0;">Clinical Handover Summary</h1>
        <p style="font-size:13px;color:#666;margin-top:10px;">Generated {timestamp} • {len(patients)} patient(s)</p>
    </div>'''

    for i, p in enumerate(patients):
        pid = p.get('id') or p.get('name') or f'Patient {i+1}'
        age = p.get('age', '')
        sex = p.get('sex', '')
        demo = f"{age}yo {sex}" if age else sex

        html += f'''<div style="background:rgba(255,255,255,0.02);border:1px solid #333;border-radius:16px;margin-bottom:20px;overflow:hidden;">
        <div style="background:rgba(255,255,255,0.05);padding:16px 20px;border-bottom:1px solid #333;display:flex;align-items:center;gap:14px;">
            <span style="background:#fff;color:#000;width:36px;height:36px;border-radius:10px;display:inline-flex;align-items:center;justify-content:center;font-weight:700;">{i+1}</span>
            <span style="font-size:18px;font-weight:600;color:#fff;">{pid}</span>
            <span style="margin-left:auto;color:#888;font-size:13px;">{demo}</span>
        </div>'''

        # Critical alerts
        allergy = p.get('allergies', '')
        if allergy and allergy.lower() not in ['none', 'nkda', 'nil', 'no known allergies', '']:
            html += f'<div style="background:rgba(239,68,68,0.15);color:#f87171;padding:12px 20px;font-weight:600;font-size:13px;border-bottom:1px solid #333;">⚠️ ALLERGIES: {allergy}</div>'

        resus = p.get('resus_status', '')
        if resus:
            color = '#fbbf24' if 'dnar' in resus.lower() or 'dnr' in resus.lower() else '#60a5fa'
            bg = 'rgba(251,191,36,0.15)' if 'dnar' in resus.lower() or 'dnr' in resus.lower() else 'rgba(96,165,250,0.15)'
            html += f'<div style="background:{bg};color:{color};padding:12px 20px;font-weight:600;font-size:13px;border-bottom:1px solid #333;">🏥 RESUS STATUS: {resus}</div>'

        # All fields
        fields = [
            ('PRESENTING COMPLAINT', p.get('admission_reason', '')),
            ('PAST MEDICAL HISTORY', ', '.join(p.get('pmh', [])) if p.get('pmh') else ''),
            ('CURRENT ISSUES', p.get('current_issues', '')),
            ('MEDICATIONS', p.get('medications', '')),
            ('IV ACCESS / LINES', p.get('lines', '')),
            ('OBSERVATIONS', p.get('observations', '')),
            ('INVESTIGATIONS', p.get('investigations', '')),
            ('PLAN / TASKS', p.get('plan', '')),
        ]

        for label, value in fields:
            if value and value.strip():
                html += f'''<div style="display:flex;padding:14px 20px;border-bottom:1px solid #262626;">
                    <span style="width:140px;flex-shrink:0;font-size:10px;font-weight:600;color:#666;letter-spacing:1px;">{label}</span>
                    <span style="flex:1;font-size:14px;color:#e0e0e0;line-height:1.6;">{value}</span>
                </div>'''

        html += '</div>'

    html += '''<div style="text-align:center;padding-top:20px;border-top:1px solid #333;margin-top:20px;">
        <p style="color:#666;font-size:11px;">Generated by CLINITY • Powered by MedGemma • Decision-support tool only</p>
    </div></div>'''

    return html


def compile_snapshot(files, progress=gr.Progress()):
    if not files:
        return "<div style='text-align:center;padding:80px;color:#666;'>Upload documents to generate summary</div>", None

    progress(0.1, desc="Starting...")
    all_content = []

    for i, f in enumerate(files):
        progress((i + 1) / (len(files) + 2), desc=f"Processing file {i+1}/{len(files)}...")
        path = Path(f.name)
        ext = path.suffix.lower()

        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp']:
            all_content.append(process_image(Image.open(path)))
        elif ext == '.pdf':
            all_content.append(process_pdf(str(path)))
        elif ext in ['.txt', '.md']:
            with open(path, 'r', errors='ignore') as file:
                all_content.append(process_text(file.read()))

    progress(0.85, desc="Structuring data...")
    combined = '\n\n---\n\n'.join(all_content)
    patients = parse_patients(combined)

    progress(0.95, desc="Generating summary...")
    html = generate_html(patients)

    progress(1.0, desc="Complete!")
    return html, json.dumps(patients, indent=2)


def generate_pdf(patients_json):
    if not patients_json:
        return None

    try:
        patients = json.loads(patients_json)
    except:
        return None

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    c = canvas.Canvas(temp.name, pagesize=A4)
    w, h = A4

    margin = 20 * mm
    col_width = w - 2 * margin

    def wrap_text(text, max_width, font_name, font_size):
        """Wrap text to fit within max_width."""
        words = str(text).split()
        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            if pdfmetrics.stringWidth(test, font_name, font_size) <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def check_page(y_pos, needed=30*mm):
        if y_pos < needed:
            c.showPage()
            return h - margin
        return y_pos

    y = h - margin

    # === TITLE ===
    c.setFillColor(colors.HexColor('#1a1a2e'))
    c.setFont('Helvetica-Bold', 28)
    c.drawCentredString(w/2, y, "CLINITY")
    y -= 10*mm

    c.setFont('Helvetica', 12)
    c.setFillColor(colors.HexColor('#666666'))
    c.drawCentredString(w/2, y, "Clinical Handover Summary Report")
    y -= 7*mm

    c.setFont('Helvetica', 10)
    timestamp = datetime.now().strftime("%d %B %Y at %H:%M")
    c.drawCentredString(w/2, y, f"Generated: {timestamp}  |  Patients: {len(patients)}")
    y -= 12*mm

    # Line
    c.setStrokeColor(colors.HexColor('#cccccc'))
    c.setLineWidth(0.5)
    c.line(margin, y, w - margin, y)
    y -= 12*mm

    # === PATIENTS ===
    for idx, p in enumerate(patients):
        y = check_page(y, 60*mm)

        pid = p.get('id') or p.get('name') or f'Patient {idx+1}'
        age = p.get('age', 'Unknown')
        sex = p.get('sex', 'Unknown')

        # Patient header
        box_h = 14*mm
        c.setFillColor(colors.HexColor('#1a1a2e'))
        c.roundRect(margin, y - box_h, col_width, box_h, 3*mm, fill=1, stroke=0)

        c.setFillColor(colors.white)
        c.setFont('Helvetica-Bold', 14)
        c.drawString(margin + 6*mm, y - 9*mm, f"Patient {idx+1}: {pid}")

        c.setFont('Helvetica', 10)
        c.drawRightString(w - margin - 6*mm, y - 9*mm, f"Age: {age}  |  Sex: {sex}")

        y -= box_h + 4*mm

        # Allergies alert
        allergy = p.get('allergies', '')
        if allergy and allergy.lower() not in ['none', 'nkda', 'nil', 'no known allergies', '']:
            y = check_page(y, 15*mm)
            c.setFillColor(colors.HexColor('#fee2e2'))
            c.roundRect(margin, y - 10*mm, col_width, 10*mm, 2*mm, fill=1, stroke=0)
            c.setFillColor(colors.HexColor('#dc2626'))
            c.setFont('Helvetica-Bold', 10)

            allergy_lines = wrap_text(f"ALLERGIES: {allergy}", col_width - 12*mm, 'Helvetica-Bold', 10)
            for line in allergy_lines[:2]:
                c.drawString(margin + 5*mm, y - 7*mm, line)
                y -= 5*mm
            y -= 5*mm

        # Resus status
        resus = p.get('resus_status', '')
        if resus:
            y = check_page(y, 15*mm)
            c.setFillColor(colors.HexColor('#fef3c7'))
            c.roundRect(margin, y - 10*mm, col_width, 10*mm, 2*mm, fill=1, stroke=0)
            c.setFillColor(colors.HexColor('#b45309'))
            c.setFont('Helvetica-Bold', 10)
            c.drawString(margin + 5*mm, y - 7*mm, f"RESUSCITATION STATUS: {resus}")
            y -= 14*mm

        # Clinical sections
        sections = [
            ("PRESENTING COMPLAINT / REASON FOR ADMISSION", p.get('admission_reason', '')),
            ("PAST MEDICAL HISTORY", ', '.join(p.get('pmh', [])) if p.get('pmh') else ''),
            ("CURRENT ACTIVE ISSUES", p.get('current_issues', '')),
            ("CURRENT MEDICATIONS", p.get('medications', '')),
            ("INTRAVENOUS ACCESS / LINES", p.get('lines', '')),
            ("OBSERVATIONS / VITAL SIGNS", p.get('observations', '')),
            ("INVESTIGATIONS & RESULTS", p.get('investigations', '')),
            ("PLAN / OUTSTANDING TASKS", p.get('plan', '')),
        ]

        for title, content in sections:
            if content and content.strip():
                y = check_page(y, 25*mm)

                # Section header
                c.setFillColor(colors.HexColor('#f0f0f0'))
                c.roundRect(margin, y - 7*mm, col_width, 7*mm, 1*mm, fill=1, stroke=0)
                c.setFillColor(colors.HexColor('#333333'))
                c.setFont('Helvetica-Bold', 8)
                c.drawString(margin + 4*mm, y - 5*mm, title)
                y -= 10*mm

                # Content
                c.setFillColor(colors.HexColor('#1a1a1a'))
                c.setFont('Helvetica', 10)
                lines = wrap_text(content, col_width - 8*mm, 'Helvetica', 10)
                for line in lines[:8]:  # Max 8 lines per section
                    y = check_page(y, 10*mm)
                    c.drawString(margin + 4*mm, y, line)
                    y -= 4.5*mm
                y -= 4*mm

        # Separator between patients
        y -= 4*mm
        c.setStrokeColor(colors.HexColor('#dddddd'))
        c.setLineWidth(0.3)
        c.line(margin, y, w - margin, y)
        y -= 10*mm

    # Footer
    c.setFont('Helvetica', 7)
    c.setFillColor(colors.HexColor('#999999'))
    c.drawCentredString(w/2, 12*mm, "Generated by CLINITY | Powered by MedGemma | Decision-support tool - Always verify critical information")

    c.save()
    return temp.name


# ============ UI ============

MAIN_HTML = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

@keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeOut { to { opacity: 0; visibility: hidden; pointer-events: none; } }
@keyframes fillCircle { to { stroke-dashoffset: 0; } }
@keyframes pulse { 0%, 100% { opacity: 0.6; } 50% { opacity: 1; } }

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

/* SPLASH SCREEN */
#splashScreen {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: 100000;
    background: linear-gradient(135deg, #0a0a0a, #1a1a2e, #0a0a0a);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    animation: fadeOut 0.8s ease 2.5s forwards;
}
#splashScreen .title {
    font-size: 80px; font-weight: 700; letter-spacing: 16px; color: #fff;
    opacity: 0; animation: fadeIn 1s ease 0.2s forwards;
    text-shadow: 0 0 60px rgba(255,255,255,0.3);
}
#splashScreen .sub {
    font-size: 13px; letter-spacing: 8px; color: #666; text-transform: uppercase; margin-top: 20px;
    opacity: 0; animation: fadeIn 1s ease 0.6s forwards;
}

/* LOADING SCREEN */
#loadScreen {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: 99999;
    background: linear-gradient(135deg, #0a0a0a, #1a1a2e, #0a0a0a);
    display: none; flex-direction: column; align-items: center; justify-content: center;
}
#loadScreen.active { display: flex !important; }
#loadScreen .pct {
    font-size: 80px; font-weight: 200; color: #fff; margin-bottom: 40px;
    letter-spacing: 4px;
}
#loadScreen .ring { width: 180px; height: 180px; }
#loadScreen .ring circle { fill: none; stroke-width: 3; }
#loadScreen .ring .bg { stroke: #333; }
#loadScreen .ring .fg {
    stroke: #fff; stroke-linecap: round;
    stroke-dasharray: 440; stroke-dashoffset: 440;
    transform: rotate(-90deg); transform-origin: center;
    transition: stroke-dashoffset 0.3s ease;
}
#loadScreen .status {
    margin-top: 30px; font-size: 11px; letter-spacing: 5px; color: #555;
    text-transform: uppercase; animation: pulse 2s ease infinite;
}

/* Specific button overrides */
#submitBtn, #submitBtn button, button#submitBtn {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    border: none !important;
}
#clearBtn, #clearBtn button, button#clearBtn {
    background: transparent !important;
    color: #ffffff !important;
    border: 1px solid #444 !important;
}
</style>

<div id="splashScreen">
    <div class="title">CLINITY</div>
    <div class="sub">Clinical Intelligence Report Generator</div>
</div>

<div id="loadScreen">
    <div class="pct" id="loadPct">0%</div>
    <svg class="ring" viewBox="0 0 160 160">
        <circle class="bg" cx="80" cy="80" r="70"/>
        <circle class="fg" id="loadRing" cx="80" cy="80" r="70"/>
    </svg>
    <div class="status">Processing Documents</div>
</div>

<script>
(function() {
    const circumference = 2 * Math.PI * 70;
    let loaderStartTime = null;

    function hideLoader() {
        const loadEl = document.getElementById('loadScreen');
        const pctEl = document.getElementById('loadPct');
        const ringEl = document.getElementById('loadRing');

        if (window._clinity_interval) {
            clearInterval(window._clinity_interval);
            window._clinity_interval = null;
        }
        if (window._clinity_checker) {
            clearInterval(window._clinity_checker);
            window._clinity_checker = null;
        }

        if (pctEl) pctEl.textContent = '100%';
        if (ringEl) ringEl.style.strokeDashoffset = '0';

        setTimeout(() => {
            if (loadEl) loadEl.classList.remove('active');
            if (pctEl) pctEl.textContent = '0%';
            if (ringEl) ringEl.style.strokeDashoffset = String(circumference);
            loaderStartTime = null;
        }, 500);
    }

    // Continuous checker that runs every 500ms while loader is active
    function startChecker() {
        loaderStartTime = Date.now();

        if (window._clinity_checker) clearInterval(window._clinity_checker);

        window._clinity_checker = setInterval(() => {
            const loadEl = document.getElementById('loadScreen');
            if (!loadEl || !loadEl.classList.contains('active')) {
                clearInterval(window._clinity_checker);
                return;
            }

            // Check for results
            const html = document.body.innerHTML;
            const hasResults = html.includes('Clinical Handover Summary') ||
                             html.includes('CLINITY') && html.includes('patient');
            const hasError = html.includes('Error processing') ||
                           document.querySelector('[class*="error"]') !== null;

            // Timeout after 3 minutes
            const elapsed = Date.now() - loaderStartTime;
            const timedOut = elapsed > 180000;

            if (hasResults || hasError || timedOut) {
                hideLoader();
            }
        }, 500);
    }

    // Expose functions globally
    window._clinity_hideLoader = hideLoader;
    window._clinity_startChecker = startChecker;
})();
</script>
'''

CSS = """
html, body, .gradio-container, .gradio-container-5-29-0, main, .main, .contain, .wrap {
    background: linear-gradient(135deg, #0a0a0a, #1a1a2e, #0a0a0a) !important;
    min-height: 100vh !important;
}

/* White Primary Buttons - Multiple selectors for Gradio 6 */
button.primary,
button[class*="primary"],
.primary,
.gr-button.primary,
.gr-button-primary,
button.lg.primary,
.gradio-container button.primary,
#component-0 button.primary,
button[data-testid="primary-button"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 14px 36px !important;
    border-radius: 30px !important;
    font-size: 14px !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 20px rgba(255,255,255,0.15) !important;
    transition: all 0.3s ease !important;
}

button.primary:hover,
button[class*="primary"]:hover,
.primary:hover {
    background: #f0f0f0 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(255,255,255,0.25) !important;
}

/* Secondary Buttons */
button.secondary,
button[class*="secondary"],
.secondary,
.gr-button.secondary,
.gr-button-secondary {
    background: transparent !important;
    background-color: transparent !important;
    color: #ffffff !important;
    border: 1px solid #444 !important;
    padding: 14px 36px !important;
    border-radius: 30px !important;
    font-size: 14px !important;
    transition: all 0.3s ease !important;
}

button.secondary:hover,
button[class*="secondary"]:hover,
.secondary:hover {
    border-color: #888 !important;
    background: rgba(255,255,255,0.05) !important;
}

.gr-box, .gr-panel, .gr-form, .block { background: transparent !important; border: none !important; }
footer, .footer { display: none !important; }

/* File upload area styling */
#fileUpload {
    width: 100% !important;
}

#fileUpload > div,
#fileUpload .wrap,
#fileUpload .container {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 12px !important;
    width: 100% !important;
}

/* The drop zone */
#fileUpload .drop-zone,
#fileUpload [role="button"],
#fileUpload label {
    min-width: 200px !important;
    max-width: none !important;
    flex: 1 !important;
}

/* File list should be horizontal */
#fileUpload ul,
#fileUpload .file-list,
#fileUpload [class*="file"] ul {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    list-style: none !important;
    padding: 0 !important;
    margin: 0 !important;
    width: 100% !important;
}

#fileUpload li,
#fileUpload .file-item {
    display: inline-flex !important;
    flex-direction: row !important;
    align-items: center !important;
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    gap: 8px !important;
}

/* Hide gradio progress bar when our loader is active */
.progress-bar, .eta-bar {
    display: none !important;
}

/* CRITICAL: Target specific button IDs */
#submitBtn, #submitBtn > button, #component-12 button, [id*="submitBtn"] button {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 600 !important;
}

#clearBtn, #clearBtn > button, [id*="clearBtn"] button {
    background: transparent !important;
    background-color: transparent !important;
    color: #ffffff !important;
    border: 1px solid #444 !important;
}

/* Override Gradio's orange default */
.svelte-cmf5ev, [class*="svelte-"] {
    --button-primary-background-fill: #ffffff !important;
    --button-primary-text-color: #000000 !important;
}
"""


def create_app():
    with gr.Blocks(title="CLINITY") as app:
        gr.HTML(MAIN_HTML)

        gr.HTML('''<div style="text-align:center;padding:40px 20px 20px;font-family:Inter,sans-serif;">
            <div style="font-size:11px;font-weight:600;letter-spacing:8px;color:#555;">CLINITY</div>
            <h1 style="font-size:44px;font-weight:600;color:#fff;margin:12px 0 0;">Clinical Intelligence</h1>
            <p style="color:#666;margin-top:12px;font-size:14px;">Transform clinical documents into structured summaries</p>
        </div>''')

        with gr.Column():
            gr.HTML('<p style="text-align:center;font-size:10px;letter-spacing:3px;color:#555;margin-bottom:16px;">UPLOAD DOCUMENTS</p>')

            files = gr.File(
                file_count="multiple",
                file_types=["image", ".pdf", ".txt"],
                label="Drop files here or click to upload",
                elem_id="fileUpload"
            )

            with gr.Row():
                submit_btn = gr.Button("Generate Summary", variant="primary", size="lg", elem_id="submitBtn")
                clear_btn = gr.Button("Clear", variant="secondary", size="lg", elem_id="clearBtn")

        output_html = gr.HTML('<div style="text-align:center;padding:60px;color:#555;">Upload clinical documents to begin</div>')
        patient_data = gr.State()

        gr.HTML('<div style="height:16px;"></div>')

        with gr.Row():
            pdf_btn = gr.Button("Download PDF Report", variant="secondary")
            pdf_output = gr.File(label="", visible=False)

        # JavaScript to show loading screen
        show_loader_js = """
        () => {
            const loadEl = document.getElementById('loadScreen');
            const pctEl = document.getElementById('loadPct');
            const ringEl = document.getElementById('loadRing');

            if (loadEl) {
                // Clear any existing intervals
                if (window._clinity_interval) clearInterval(window._clinity_interval);
                if (window._clinity_checker) clearInterval(window._clinity_checker);

                loadEl.classList.add('active');
                let pct = 0;
                const circumference = 2 * Math.PI * 70;

                // Reset display
                if (pctEl) pctEl.textContent = '0%';
                if (ringEl) ringEl.style.strokeDashoffset = String(circumference);

                // Start progress animation
                window._clinity_interval = setInterval(() => {
                    if (pct < 95) {
                        pct += Math.random() * 1.5 + 0.3;
                        pct = Math.min(pct, 95);
                        if (pctEl) pctEl.textContent = Math.round(pct) + '%';
                        if (ringEl) ringEl.style.strokeDashoffset = circumference - (pct / 100) * circumference;
                    }
                }, 250);

                // Start the completion checker
                if (window._clinity_startChecker) window._clinity_startChecker();
            }
            return [];
        }
        """

        # JavaScript to hide loading screen
        hide_loader_js = """
        (html, data) => {
            const loadEl = document.getElementById('loadScreen');
            const pctEl = document.getElementById('loadPct');
            const ringEl = document.getElementById('loadRing');
            const circumference = 2 * Math.PI * 70;

            // Clear the interval
            if (window._clinity_interval) {
                clearInterval(window._clinity_interval);
                window._clinity_interval = null;
            }

            // Animate to 100%
            if (pctEl) pctEl.textContent = '100%';
            if (ringEl) ringEl.style.strokeDashoffset = '0';

            // Hide after showing 100%
            setTimeout(() => {
                if (loadEl) loadEl.classList.remove('active');
                // Reset for next use
                if (pctEl) pctEl.textContent = '0%';
                if (ringEl) ringEl.style.strokeDashoffset = String(circumference);
            }, 600);

            return [html, data];
        }
        """

        submit_btn.click(fn=None, inputs=None, outputs=None, js=show_loader_js).then(
            compile_snapshot, inputs=[files], outputs=[output_html, patient_data]
        ).then(fn=None, inputs=[output_html, patient_data], outputs=[output_html, patient_data], js=hide_loader_js)
        clear_btn.click(lambda: (None, '<div style="text-align:center;padding:60px;color:#555;">Upload clinical documents to begin</div>', None), outputs=[files, output_html, patient_data])
        pdf_btn.click(generate_pdf, inputs=[patient_data], outputs=[pdf_output]).then(
            lambda x: gr.File(value=x, visible=True) if x else gr.File(visible=False),
            inputs=[pdf_output], outputs=[pdf_output]
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True, css=CSS)
