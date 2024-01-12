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


def build_block_checking_payload_and_signature_and_stamp(keys, image_path):

    # Read and encode the image in base64 format
    base64_image = encode_image(image_path)

    # Construct the content for each key
    content = [{"type": "text", "text": f'What is the "{key}" value?'} for key in keys]

    # Add instruction for dictionary format at the end
    dict_instruction = 'Give the answer as a dictionary with the keys ' + \
                       ', '.join([f'"{key}"' for key in keys]) + \
                       ' and the corresponding values. Dont write anything else. If you dont find a key on the image, set the value to "<NOT_FOUND>".' \
                       'If you find the key but no value is associated, set the value to "<EMPTY>". No other value is accepted.'

    signature_and_stamp_instructions = """
        In a second time, check if a signature and stamp are present on the right part of the document? Answer only one word, 
                                            you have only 4 choices, "both", "stamp", "signature", "none".
                                            - If both, answer only "both". 
                                            - If only a stamp, answer only "stamp".
                                            - If only a a signature, answer only "signature".
                                            - If none are present, answer only "none
        When you have find your answer, add it in the previous dictionary with the key "Signature and stamp"
        """

    content.append({"type": "text", "text": dict_instruction})
    content.append({"type": "text", "text": signature_and_stamp_instructions})
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


def build_block4_checking_payload(image_path):
    # Read and encode the image in base64 format
    base64_image = encode_image(image_path)

    dict_instruction = """
    What is the "Nom et prénom" value? What is the "E-mail" value? What is the "Tél" value? Is "Société" value "Pop Valet"?
    Give the answer as a dictionary with the keys "Nom et prénom", "E-mail", "Tél", "Société" and the corresponding values. 
    If you are not comfortable with giving the value for "Nom et prénom", "E-mail" or "Tél", just use "<FILLED>" instead.  
    If you dont find a key on the image, set the value to "<NOT_FOUND>".
    Dont write anything else.
    If you find the key but no value is associated, set the value to "<EMPTY>". No other value is accepted.
    """
    content = [{"type": "text", "text": dict_instruction}]
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

def build_block2_checking_payload_and_signature_stamp(image_path):
    # Read and encode the image in base64 format
    #base64_image = encode_image(image_path)
    logger.info(f"image path content:\n{image_path}")

    dict_instruction = """
    First do :
    What is the "Immatriculé" value? What is the "Kilométrage" value? What is the "Restitué le" value? What is the "N° de série" value?
    Give the answer as a dictionary with the keys "Immatriculé", "Kilométrage", "Restitué le", "N° de série" and the corresponding values.
    If you are not comfortable with giving the value for "Immatriculé", "Kilométrage", "Restitué le" or "N° de série" just use "<FILLED>" instead.
    If you dont find a key on the image, set the value to "<NOT_FOUND>".Dont write anything else.If you find the key but no value is associated, set the value to "<EMPTY>".
    No other value is accepted.

    In a second time, check if a signature and stamp are present on the right part of the document? Answer only one word,
    you have only 4 choices, "both", "stamp", "signature", "none".                                  
    - If both, answer only "both".                                         
    - If only a stamp, answer only "stamp".                                        
    - If only a a signature, answer only "signature".                                        
    - If none are present, answer only "noneWhen you have find your answer, add it in the previous dictionary with the key "Signature and stamp".
    I just want the dictionary structure, without ``` at the beginning or the end ! 

    To help you I gave you to exemples, for the first image you should return: {"Immatriculé": ‘<EMPTY>',"Kilométrage": '46072',"Restitué le": '21/09/23',"N° de série": 'VF1IL00055729079',"Signature and stamp": 'signature'}

    For the second you should return : {"Immatriculé": ‘EK-112-NP',"Kilométrage": '198206',"Restitué le": '<EMPTY>',
    "N° de série": 'VF15RBFOA57206076',"Signature and stamp": 'stamp'}

    Can you perform the same for the third image ?
    """
    content = [{"type": "text", "text": dict_instruction}]
    logger.info(f"Block 2 content:\n{content}")

    few_shot_block2_1_path = 'data/few_shots_no_sub/block2/Block2_1.jpeg'
    base64_image_few_shot_block2_1 = encode_image(few_shot_block2_1_path)

    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image_few_shot_block2_1}",
            "detail": "high"
        }
    })

    few_shot_block2_2_path = 'data/few_shots_no_sub/block2/Block2_2.jpeg'
    base64_image_few_shot_block2_2 = encode_image(few_shot_block2_2_path)

    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image_few_shot_block2_2}",
            "detail": "high"
        }
    })


    base64_image = encode_image(image_path)

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

def build_block4_checking_payload_and_signature_stamp(image_path):
    # Read and encode the image in base64 format
    #base64_image = encode_image(image_path)
    logger.info(f"image path content:\n{image_path}")

    dict_instruction = """
    First do :
    What is the "Nom et prénom" value? What is the "E-mail" value? What is the "Tél" value? Is "Société" value "Pop Valet"?
    Give the answer as a dictionary with the keys "Nom et prénom", "E-mail", "Tél", "Société" and the corresponding values.
    If you are not comfortable with giving the value for "Nom et prénom", "E-mail" or "Tél", just use "<FILLED>" instead.
    If you dont find a key on the image, set the value to "<NOT_FOUND>".Dont write anything else.If you find the key but no value is associated, set the value to "<EMPTY>".
    No other value is accepted.

    In a second time, check if a signature and stamp are present on the right part of the document? Answer only one word,
    you have only 4 choices, "both", "stamp", "signature", "none".                                  
    - If both, answer only "both".                                         
    - If only a stamp, answer only "stamp".                                        
    - If only a a signature, answer only "signature".                                        
    - If none are present, answer only "noneWhen you have find your answer, add it in the previous dictionary with the key "Signature and stamp".
    I just want the dictionary structure, without ``` at the beginning or the end ! 

    To help you I gave you to exemples, for the first image you should return: {"Nom et prénom": ‘Hemond Sonia',"E-mail": '<EMPTY>',"Tél": '06 13 34 05 33',"Société": 'Merieux',"Signature and stamp": 'signature'}

    For the second you should return : {"Nom et prénom": ‘Freixo Emmanuel',"E-mail": '<EMPTY>',"Tél": '0666443658',"Société": '<EMPTY>',"Signature and stamp": 'both'}

    Can you perform the same for the third image ?
    Don't put any '\n' in your response.
    """
    content = [{"type": "text", "text": dict_instruction}]
    logger.info(f"Block 4 content:\n{content}")

    few_shot_block4_1_path = 'data/few_shots_no_sub/block4/Block4_1.jpeg'
    base64_image_few_shot_block4_1 = encode_image(few_shot_block4_1_path)

    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image_few_shot_block4_1}",
            "detail": "high"
        }
    })

    few_shot_block4_2_path = 'data/few_shots_no_sub/block4/Block4_2.jpeg'
    base64_image_few_shot_block4_2 = encode_image(few_shot_block4_2_path)

    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image_few_shot_block4_2}",
            "detail": "high"
        }
    })


    base64_image = encode_image(image_path)

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
