import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def fetch_image_captions(url, output_file="captions.txt"):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_elements = soup.find_all('img')

    with open(output_file, "w") as caption_file:
        for img_element in img_elements:
            img_url = img_element.get('src')
            
            # Skip SVG and 1x1 images
            if 'svg' in img_url or '1x1' in img_url:
                continue

            # Fix relative URLs
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
            elif not img_url.startswith('http://') and not img_url.startswith('https://'):
                continue  
            
            try:
                img_response = requests.get(img_url)
                img = Image.open(BytesIO(img_response.content))

                # Skip small images
                if img.size[0] * img.size[1] < 400:
                    continue
                
                img = img.convert('RGB')

                inputs = processor(img, return_tensors="pt")
                output = model.generate(**inputs, max_new_tokens=50)
                caption = processor.decode(output[0], skip_special_tokens=True)

                caption_file.write(f"{img_url}: {caption}\n")
            except Exception as e:
                print(f"Error processing image {img_url}: {e}")
                continue

# URL of the Wikipedia page to process
wiki_url = "https://en.wikipedia.org/wiki/IBM"
fetch_image_captions(wiki_url)
