from typing import Dict, List
import requests
from bs4 import BeautifulSoup

# Placeholder for agent tool registration
def function_tool(func):
    return func

@function_tool
def scrape_url(url: str) -> Dict:
    """Scrape product title, description, and images from a product page URL."""
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    # Title
    title = soup.find('title').text.strip() if soup.find('title') else ''
    # Description (try meta description, then product description)
    desc = ''
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and meta_desc.get('content'):
        desc = meta_desc['content']
    else:
        # Try to find a product description div
        desc_div = soup.find('div', {'class': 'product__description'})
        if desc_div:
            desc = desc_div.get_text(strip=True)
    # Try to find product gallery images first
    gallery_imgs = []
    # Look for common product gallery containers
    for gallery_class in ["product__media", "product-gallery", "product__media-list", "product__media-wrapper"]:
        gallery = soup.find_all(class_=gallery_class)
        for g in gallery:
            for img in g.find_all("img"):
                src = img.get("src")
                if src:
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        src = 'https://www.uncomfy.store' + src
                    gallery_imgs.append(src)
                data_src = img.get("data-src")
                if data_src:
                    if data_src.startswith('//'):
                        data_src = 'https:' + data_src
                    elif data_src.startswith('/'):
                        data_src = 'https://www.uncomfy.store' + data_src
                    gallery_imgs.append(data_src)
                srcset = img.get("srcset")
                if srcset:
                    first_url = srcset.split(',')[0].strip().split(' ')[0]
                    if first_url.startswith('//'):
                        first_url = 'https:' + first_url
                    elif first_url.startswith('/'):
                        first_url = 'https://www.uncomfy.store' + first_url
                    gallery_imgs.append(first_url)
    # Remove duplicates
    gallery_imgs = list(dict.fromkeys(gallery_imgs))
    # If not enough, supplement with all <img> tags
    all_imgs = set(gallery_imgs)
    if len(gallery_imgs) < 10:
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                if src.startswith('//'):
                    src = 'https:' + src
                elif src.startswith('/'):
                    src = 'https://www.uncomfy.store' + src
                all_imgs.add(src)
            data_src = img.get('data-src')
            if data_src:
                if data_src.startswith('//'):
                    data_src = 'https:' + data_src
                elif data_src.startswith('/'):
                    data_src = 'https://www.uncomfy.store' + data_src
                all_imgs.add(data_src)
            srcset = img.get('srcset')
            if srcset:
                first_url = srcset.split(',')[0].strip().split(' ')[0]
                if first_url.startswith('//'):
                    first_url = 'https:' + first_url
                elif first_url.startswith('/'):
                    first_url = 'https://www.uncomfy.store' + first_url
                all_imgs.add(first_url)
        all_imgs = list(all_imgs)
    else:
        all_imgs = gallery_imgs
    # Filtering
    def is_relevant(img_url):
        skip_keywords = ["icon", "logo", "thumb", "sprite", "favicon", "banner", "arrow", "cart", "star"]
        return not any(kw in img_url.lower() for kw in skip_keywords)
    filtered_images = [img for img in all_imgs if is_relevant(img)]
    filtered_images = filtered_images[:10]
    result = {
        'title': title,
        'description': desc,
        'images': filtered_images,
    }
    return result 