import os
import fitz
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import pandas as pd
import openai
import base64
from io import BytesIO
import json
import gc
import re
import unicodedata


# Paso 1
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

# Paso 2
def normalize_text(text):
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode("utf-8")
    return text.lower()

def keyword_regex_pattern(keyword):
    keyword = re.escape(keyword)
    keyword = keyword.replace(r'\ ', r'[\W_]*')
    return keyword

def rotate_and_score_pages_lowres(pil_images, keywords, ocr):
    page_data = []
    regex_patterns = [re.compile(keyword_regex_pattern(k), re.IGNORECASE) for k in keywords]

    for idx, original_img in enumerate(pil_images):
        print(f"\n📄 Pagina {idx+1}: analisi OCR in corso...")
        resized = original_img.resize((int(original_img.width * 0.5), int(original_img.height * 0.5)))
        img_np = np.array(resized)

        try:
            result = ocr.ocr(img_np, cls=True)
            raw_text = " ".join([line[1][0] for block in result if isinstance(block, list)
                                  for line in block if isinstance(line, list)]) if result else ""
            all_text = normalize_text(raw_text)
        except Exception as e:
            print(f"⚠️ Errore OCR nella pagina {idx+1}: {e}")
            all_text = ""

        match_count = sum(1 for pattern in regex_patterns if pattern.search(all_text))
        print(f"   ✅ {match_count} parole chiave trovate")

        page_data.append({
            "index": idx,
            "original_image": original_img,
            "text": all_text,
            "keywords_found": match_count
        })

        del img_np, resized
        gc.collect()

    return page_data

# Paso 3
def find_best_rotated_page(pages_data):
    if not pages_data:
        return None, "", -1
    best = max(pages_data, key=lambda x: x["keywords_found"])
    if best["keywords_found"] == 0:
        print("❌ Nessuna parola chiave trovata.")
        return None, "", -1
    print(f"✅ Pagina selezionata: {best['index']+1} con {best['keywords_found']} parole chiave.")
    return best["original_image"], best["text"], best["index"]

# Paso 4
def rotate_image_by_ocr_angle(image_pil, ocr):
    try:
        img_np = np.array(image_pil)
        result = ocr.ocr(img_np, cls=True)
        vertical_lines = 0
        total_lines = 0

        for block in result:
            for line in block:
                (x1, y1), (x2, y2) = line[0][0], line[0][1]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 75 <= angle <= 105:
                    vertical_lines += 1
                total_lines += 1

        if total_lines == 0:
            return image_pil

        return image_pil.rotate(90, expand=True) if (vertical_lines / total_lines) > 0.6 else image_pil

    except Exception as e:
        print(f"⚠️ Errore durante la rotazione: {e}")
        return image_pil
    
# Paso 5
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
    
# Paso 6
def process_pdfs_in_folder(folder_path, keywords, valid_values, openai_key, progress_callback=None):
    ocr = PaddleOCR(use_angle_cls=True, lang='it', show_log=False)
    client = openai.OpenAI(api_key=openai_key)
    final_data = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    for idx, pdf_file in enumerate(pdf_files, 1):
        if progress_callback:
            progress_callback(idx, len(pdf_files))
        print(f"\n📂 ({idx}/{len(pdf_files)}) Elaborazione: {pdf_file}")
        pdf_path = os.path.join(folder_path, pdf_file)
        try:
            images = pdf_to_images(pdf_path)
            if not images:
                raise Exception("Nessuna pagina estratta.")

            pages_data = rotate_and_score_pages_lowres(images, keywords, ocr)
            target_image, ocr_text, _ = find_best_rotated_page(pages_data)
            if target_image:
                rotated_target = rotate_image_by_ocr_angle(target_image, ocr)
                extracted = call_gpt_image_with_text(rotated_target, ocr_text, pdf_file, valid_values, client)
            else:
                extracted = {
                    "Nome": pdf_file,
                    "Fasciame Spessore": "",
                    "Qualita Fasciame": "",
                    "Fondo Spessore": "",
                    "Qualita Fondo": ""
                }

            final_data.append(extracted)

        except Exception as e:
            print(f"❌ Errore nel file {pdf_file}: {e}")
            final_data.append({
                "Nome": pdf_file,
                "Fasciame Spessore": "",
                "Qualita Fasciame": "",
                "Fondo Spessore": "",
                "Qualita Fondo": ""
            })

        gc.collect()

    return pd.DataFrame(final_data)