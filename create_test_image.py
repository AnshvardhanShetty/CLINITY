"""Create a simulated handwritten clinical note image for testing OCR."""

from PIL import Image, ImageDraw, ImageFont
import os

def create_handwritten_note():
    # Create a white image (like paper)
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    # Try to use a handwriting-like font, fall back to default
    try:
        # Try common fonts that look somewhat handwritten
        font_paths = [
            "/System/Library/Fonts/Noteworthy.ttc",
            "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
            "/System/Library/Fonts/MarkerFelt.ttc",
        ]
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, 24)
                break
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Simulated handwritten clinical note content
    note_lines = [
        "Quick review 19/1",
        "",
        "Mrs Morrison - day 4",
        "",
        "- AKI improving Cr 189 (was 245)",
        "- New confusion overnight - ?delirium",
        "- Still in AF, rate ok",
        "- UTI - day 4/5 abx",
        "",
        "Plan:",
        "- Delirium screen - 4AT",
        "- Chase echo TODAY - v important",
        "- Cont fluids",
        "- Family mtg 2pm",
        "",
        "?ready for d/c in 2-3 days if",
        "echo ok + mobility improves",
        "",
        "- Dr S"
    ]

    # Draw the text with slight variations to simulate handwriting
    y_position = 30
    for line in note_lines:
        # Add slight x variation
        x_offset = 40 + (hash(line) % 10) - 5
        draw.text((x_offset, y_position), line, fill='darkblue', font=font)
        y_position += 28

    # Add some "paper" effects - light lines
    for y in range(50, height, 30):
        draw.line([(30, y), (width - 30, y)], fill='#e0e0e0', width=1)

    # Save the image
    output_path = "/Users/anshshetty/Library/Mobile Documents/com~apple~CloudDocs/MedGamma/test_data/handwritten_note.png"
    image.save(output_path)
    print(f"Created test handwritten note: {output_path}")
    return output_path

if __name__ == "__main__":
    create_handwritten_note()
