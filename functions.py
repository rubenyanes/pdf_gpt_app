import os
import fitz
import numpy as np
from PIL import Image
import pandas as pd
import openai
import base64
from io import BytesIO
import json
import gc
import re
import unicodedata
from google.cloud import vision
import tempfile

def pdf_to_images(pdf_path):
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    except Exception as e:
        print(f"❌ Errore durante l'elaborazione di {pdf_path}: {e}")
    return images

def normalize_text(text):
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode("utf-8")
    return text.lower()

def keyword_regex_pattern(keyword):
    keyword = re.escape(keyword)
    keyword = keyword.replace(r'\ ', r'[\W_]*')
    return re.compile(keyword, re.IGNORECASE)

def rotate_and_score_pages_lowres(pil_images, keywords, vision_client):
    page_data = []
    regex_patterns = [keyword_regex_pattern(k) for k in keywords]

    for idx, image in enumerate(pil_images):
        print(f"\n📄 Pagina {idx+1}: analisi OCR (Google Vision) in corso...")
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        content = img_byte_arr.getvalue()

        try:
            gimage = vision.Image(content=content)
            response = vision_client.document_text_detection(image=gimage)
            if response.error.message:
                print(f"⚠️ Errore Google Vision nella pagina {idx+1}: {response.error.message}")
                all_text = ""
            else:
                text = response.full_text_annotation.text if response.full_text_annotation else ""
                all_text = normalize_text(text)
        except Exception as e:
            print(f"⚠️ Errore OCR nella pagina {idx+1}: {e}")
            all_text = ""

        match_count = sum(1 for pattern in regex_patterns if pattern.search(all_text))
        print(f"   ✅ {match_count} parole chiave trovate")

        page_data.append({
            "index": idx,
            "original_image": image,
            "text": all_text,
            "keywords_found": match_count
        })

    return page_data

def find_best_rotated_page(pages_data):
    if not pages_data:
        return None, "", -1
    best = max(pages_data, key=lambda x: x["keywords_found"])
    if best["keywords_found"] == 0:
        print("❌ Nessuna parola chiave trovata.")
        return None, "", -1
    print(f"✅ Pagina selezionata: {best['index']+1} con {best['keywords_found']} parole chiave.")
    return best["original_image"], best["text"], best["index"]

def rotate_image_by_ocr_angle(image_pil, vision_client):
    return image_pil  # puedes implementar rotación más adelante si lo deseas

def call_gpt_image_with_text(image_pil, ocr_text, filename, valid_values, client):
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    valori_lista = "\n- " + "\n- ".join(valid_values)

    prompt = f"""
        Questa immagine rappresenta una tabella tecnica con le misure di un recipiente a pressione. 
        Il testo OCR estratto dalla stessa immagine è il seguente:
        {ocr_text}
        
        Usa l'immagine per estrarre i seguenti valori (se presenti) e ti puoi aiutare con il testo OCR:
        
        - **Fasciame Spessore**: il valore numerico (spesorre nominale in mm) che si riferisce a "FASCIAME" o "TRONCHETTI"
        - **Qualità Fasciame**: il tipo di materiale del fasciame. Deve essere uno dei seguenti valori:
        
        {valori_lista}
        
        - **Fondo Spessore**: il valore numerico (spessore nominale in mm) "FONDO" o "FONDI" o "CALOTTA"
        - **Qualità Fondo**: il materiale del fondo. Deve essere uno dei seguenti valori (gli stessi di Qualità Fasciame).
        
        Se vedi un valore **incompleto, abbreviato o scritto male** (es. "Fe52", "Fe 44", "Fe52 /c"), prova a **riconoscere e completare** il valore corretto dalla lista, se è chiaramente deducibile.  

        Tieni presente che lo spessore del fondo è o uguale o simile a quello del fasciame (con una differenza di 0.5 o 0.6 al massimo) e i valori vanno tra 4 mm e 7 mm di spessore
        
        Rispondi solo in formato JSON come questo:
        
        {{
          "Fasciame Spessore": "",
          "Qualita Fasciame": "",
          "Fondo Spessore": "",
          "Qualita Fondo": ""
        }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Sei un assistente che estrae dati tecnici da tabelle complesse."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=1000
        )

        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        data["Nome"] = filename
        if data.get("Qualita Fondo", "").strip() == "" and data.get("Qualita Fasciame", "").strip() != "":
            data["Qualita Fondo"] = data["Qualita Fasciame"]
        return data

    except Exception as e:
        print(f"❌ Error analizando {filename}: {e}")
        return {
            "Nome": filename,
            "Fasciame Spessore": "",
            "Qualita Fasciame": "",
            "Fondo Spessore": "",
            "Qualita Fondo": ""
        }

def process_pdfs_in_folder(uploaded_files, keywords, valid_values, openai_key, progress_callback=None):
    vision_client = vision.ImageAnnotatorClient()
    client = openai.OpenAI(api_key=openai_key)
    final_data = []

    for idx, uploaded_file in enumerate(uploaded_files, 1):
        if progress_callback:
            progress_callback(idx, len(uploaded_files))
        print(f"\n📂 ({idx}/{len(uploaded_files)}) Elaborazione: {uploaded_file.name}")
        
        try:
            # Guarda temporalmente el archivo en disco para que fitz pueda leerlo
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_pdf_path = tmp_file.name

            images = pdf_to_images(tmp_pdf_path)
            if not images:
                raise Exception("Nessuna pagina estratta.")

            pages_data = rotate_and_score_pages_lowres(images, keywords, vision_client)
            target_image, ocr_text, _ = find_best_rotated_page(pages_data)
            if target_image:
                rotated_target = rotate_image_by_ocr_angle(target_image, vision_client)
                extracted = call_gpt_image_with_text(rotated_target, ocr_text, uploaded_file.name, valid_values, client)
            else:
                extracted = {
                    "Nome": uploaded_file.name,
                    "Fasciame Spessore": "",
                    "Qualita Fasciame": "",
                    "Fondo Spessore": "",
                    "Qualita Fondo": ""
                }

            final_data.append(extracted)

        except Exception as e:
            print(f"❌ Errore nel file {uploaded_file.name}: {e}")
            final_data.append({
                "Nome": uploaded_file.name,
                "Fasciame Spessore": "",
                "Qualita Fasciame": "",
                "Fondo Spessore": "",
                "Qualita Fondo": ""
            })

        gc.collect()

    return pd.DataFrame(final_data)
