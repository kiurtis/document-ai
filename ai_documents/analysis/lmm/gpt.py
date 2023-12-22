import os
import base64
import json

import requests
from dotenv import load_dotenv, find_dotenv
from loguru import logger

load_dotenv(find_dotenv())
# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def set_payload_content(content):
    # Build the payload
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 150
    }
    return payload

def build_block_checking_payload(keys, image_path):

    # Read and encode the image in base64 format
    base64_image = encode_image(image_path)

    # Construct the content for each key
    content = [{"type": "text", "text": f'What is the "{key}" value?'} for key in keys]

    # Add instruction for dictionary format at the end
    dict_instruction = 'Give the answer as a dictionary with the keys ' + \
                       ', '.join([f'"{key}"' for key in keys]) + \
                       ' and the corresponding values. Dont write anything else. If you dont find a key on the image, set the value to "<NOT_FOUND>".' \
                       'If you find the key but no value is associated, set the value to "<EMPTY>". No other value is accepted.'
    content.append({"type": "text", "text": dict_instruction})
    logger.info(f"Block 2 content:\n{content}")

    # Add the image part
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
            "detail": "high"
        }
    })

    payload = set_payload_content(content)

    return payload


def number_plate_check_gpt(plate_number, image_path, with_few_shots=False):

    # Read and encode the image in base64 format
    base64_image = encode_image(image_path)

    content = [{"type": "text", "text": f'Analyze the image. Your objective is to check if the value of '
                                        f' "Immatriculé" is "{plate_number}". The value is often different so watch out.'
                                        f'\n - If you see this, write "{plate_number}" and only this.'
                                        f'\n - If you read something different, write what you read, and only this.'
                                        f'\n - Finally if there nothing after the word "Immatriculé" write "<EMPTY>" and only this.'}]

    logger.info(content)
    #Insist on the fact that I don't want a phrase
    # Add the image part
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    })

    payload = set_payload_content(content)

    return payload

def build_overall_quality_checking_payload(image_path):
    # Read and encode the image in base64 format
    base64_image = encode_image(image_path)

    # Construct the content for each key
    content = [{"type": "text", "text": f'''I send you a picture of document. You have to tell me if the document is readable. 
    Answer "No" (and nothing else) if the document is creased, poorly lit, very crumpled, poorly framed or distorted, 
    or any other condition that makes it hard to read. Otherwise answer "Yes" (and nothing else).'''}]

    logger.info(content)

    # Add the image part
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    })

    payload = set_payload_content(content)

    return payload

def request_completion(payload):
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

def build_signature_checking_payload(image_path):
    base64_image = encode_image(image_path)

    # Construct the content for each key
    content = [{"type": "text", "text": '''Are the signature and stamp present on the document? Answer only one word, 
                                            you have only 4 choices, "both", "stamp", "signature", "none".
                                            - If both, answer only "both". 
                                            - If only a stamp, answer only "stamp".
                                            - If only a a signature, answer only "signature".
                                            - If none are present, answer only "none".'''}]
    logger.info(content)

    # Add the image part
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    })
    payload = set_payload_content(content)

    return payload

if __name__ == '__main__':
    # Example usage
    keys = ["Immatriculé", "Kilométrage", "Restitué le", "Numéro de série", "Modèle", "Couleur"]
    image_path = "/data/performances_data/valid_data/fleet_services_images/DM-984-VT_Proces_verbal_de_restitution_page-0001/blocks/DM-984-VT_Proces_verbal_de_restitution_page-0001_block 3.png"
    payload = build_block_checking_payload(keys, image_path)
    print(payload)


    requests_completion = request_completion(payload, headers)

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
