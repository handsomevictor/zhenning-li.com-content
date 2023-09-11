from PIL import Image
import os
from reportlab.pdfgen import canvas
from tqdm import tqdm

# Set the directory and find all PNG files
image_folder = "/Users/zhenningli/Downloads/the inflationary spiral"
images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
images.sort(key=lambda x: int(x.split('.')[0]))
print(images)

# Initialize PDF
pdf_path = "/Users/zhenningli/Downloads/the_inflationary_spiral1.pdf"
c = canvas.Canvas(pdf_path)

# Loop through images and add to PDF
for i in tqdm(images, total=len(images)):
    image_path = os.path.join(image_folder, i)
    img = Image.open(image_path)

    # Convert image to RGB (remove alpha channel)
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    img_width, img_height = img.size
    c.setPageSize((img_width, img_height))
    c.drawImage(image_path, 0, 0, width=img_width, height=img_height)
    c.showPage()

# Save PDF
c.save()

print(f"PDF has been saved at {pdf_path}")
