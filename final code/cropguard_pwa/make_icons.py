from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, filename):
    # Create a dark green image
    img = Image.new('RGB', (size, size), color='#064e3b')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple leaf shape / icon
    # Just draw a white circle and text for now
    margin = size // 6
    draw.ellipse([margin, margin, size - margin, size - margin], fill='#10b981')
    
    # Try to add "CG" text
    try:
        # Scale text roughly based on size
        font_size = size // 3
        # Use default font
        font = ImageFont.load_default()
        # Draw "CG" in center
        text = "CG"
        # Since default font is small, let's just scale the image later if we must, or just use shapes.
        # Let's draw a few more shapes to look like a plant.
        draw.polygon([(size//2, size//4), (size//2 + size//4, size//2), (size//2, size//2 + size//4), (size//2 - size//4, size//2)], fill='#ecfdf5')
    except:
        pass
        
    img.save(filename)

os.makedirs('icons', exist_ok=True)
create_icon(192, 'icons/icon-192.png')
create_icon(512, 'icons/icon-512.png')
print("Icons generated.")
