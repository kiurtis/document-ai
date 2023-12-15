import os
import base64
import json

import requests
from dotenv import load_dotenv, find_dotenv

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
        "max_tokens": 300
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
                       'If you find the key but no value is associated, set the value to "<EMPTY>".'
    content.append({"type": "text", "text": dict_instruction})

    # Add the image part
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    })

    payload = set_payload_content(content)

    return payload


def number_plate_check_gpt(plate_number, image_path):

    # Read and encode the image in base64 format
    base64_image = encode_image(image_path)

    # Construct the content:
    content = [{"type": "text", "text": f'Can you read "{plate_number}" on the image file after the word "Immatriculé"? '
                                        f'If yes return "{plate_number}",if something is writen after "Immatriculé" but different, return this word'
                                        f'Finally if there nothing after the word "Immatriculé" return "<EMPTY>".'}]
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
    content = [{"type": "text", "text": f'Is the overall quality of the document ok? Answer "No" (and nothing else)'
                                        f'if the document is very creased, poorly lit, very crumpled, poorly framed or '
                                        f'distorted, otherwise answer "Yes" (and nothing else).'}]


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
    content = [{"type": "text", "text": f'Is the signature and stamp present on the document? Answer only one word, you have only 3 choices; "both", "stamp","signature".'
                                        f'If both, answer only "both"'
                                        f'If only a stamp, answer only "stamp" '
                                        f'If only a a signature, answer only "signature"'}]

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
    image_path = "/Users/amielsitruk/work/terra_cognita/customers/pop_valet/ai_documents/data/performances_data/valid_data/fleet_services_images/DM-984-VT_Proces_verbal_de_restitution_page-0001/blocks/DM-984-VT_Proces_verbal_de_restitution_page-0001_block 3.png"
    payload = build_block_checking_payload(keys, image_path)
    print(payload)


    requests_completion = request_completion(payload,headers)

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
