import os
import time
import base64
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models

from dotenv import load_dotenv, find_dotenv
from loguru import logger

load_dotenv(find_dotenv())

import pathlib
import textwrap

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def generate(path_to_image, question):
  waiting_time = 5
  print(f"Waiting {waiting_time} seconds before generating content to limit quota usage")
  time.sleep(waiting_time)
  model = GenerativeModel("gemini-pro-vision")
  image = encode_image(path_to_image)
  image = Part.from_data(data=base64.b64decode(image), mime_type="image/jpeg")
  responses = model.generate_content(
    [image,
     question],
    generation_config={
        "max_output_tokens": 2048,
        "temperature": 0.4,
        "top_p": 1,
        "top_k": 32
    },
    safety_settings={
          generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
          generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
          generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
          generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    stream=False,
  )
  return responses

def analyse_block2(image_path):
  question = """What is the value of \"Immatriculé\"? 
    What is the value of \"Kilométrage\"? 
    What is the value of \"Restitué le\" ? 
    What is the value of \"N° de série\"?
    Give the answer as a dictionary with the keys \"Immatriculé\", \"Kilométrage\", \"Restitué le\", \"N° de série\" and the corresponding values
    If you dont find a key on the image, set the value to \"<NOT_FOUND>\".Dont write anything else. If you find the key but no value is associated, set the value to \"<EMPTY>\". No other value is accepted.
    
    Then, check if a signature and stamp are present on the right part of the document? Answer only one word, 
                                        you have only 4 choices, \"both\", \"stamp\", \"signature\", \"none\".
                                        - If both, answer only \"both\". 
                                        - If only a stamp, answer only \"stamp\".
                                        - If only a a signature, answer only \"signature\".
                                        - If none are present, answer only \"none
    When you have find your answer, add it in the previous dictionary with the key \"Signature and stamp\"
    Your answer should thus have this format:
    {
        "Immatriculé": "value",
        "Kilométrage": "value",
        "Restitué le": "value",
        "N° de série": "value",
        "Signature and stamp": "value"
    }
    """

  response = generate(image_path, question)
  return response


def analyse_block4(image_path):
    question = '''What is the value of "Nom et prénom"? 
                What is the value of "E-mail"? 
                What is the value of "Tél"? 
                What is the value of "Société"?  Hint: This is the name of the company. it can be "Pop Valet", or something very different.
                Give the answer as a dictionary with the keys "Nom et prénom", "E-mail", "Tél", "Société" and the corresponding values. 
                If you find the key but no value is associated, set the value to "<EMPTY>".
                
                Dont write anything else.
                Your answer should thus have this format:
                {
                    "Nom et prénom": "value",
                    "E-mail": "value",
                    "Tél": "value",
                    "Société": "value"
                }'''
    logger.info(question)

    response = generate(image_path, question)
    return response

def check_plate_number(plate_number, image_path):

    question = f'''Analyze the image. Your objective is to check if the value of "Immatriculé" is "{plate_number}". 
                   The value is often different so watch out.'
                                        \n - If you see this, write "{plate_number}" and only this.'
                                        \n - If you read something different, write what you read, and only this.'
                                        \n - Finally if there nothing after the word "Immatriculé" write "<EMPTY>" and only this.'''
    logger.info(question)

    response = generate(image_path, question)
    return response

def check_overall_quality(image_path):
    # Construct the content for each key
    question = '''I send you a picture of document. You have to tell me if the document is readable. 
    Answer "No" (and nothing else) if the document is creased, poorly lit, very crumpled, poorly framed or distorted, 
    or any other condition that makes it hard to read. Otherwise answer "Yes" (and nothing else).'''

    logger.info(question)

    response = generate(image_path, question)
    print(response)
    return response


def check_signature_and_stamp(image_path):

    # Construct the content for each key
    question = '''Are the signature and stamp present on the document? Answer only one word, 
                                            you have only 4 choices, "both", "stamp", "signature", "none".
                                            - If both, answer only "both". 
                                            - If only a stamp, answer only "stamp".
                                            - If only a a signature, answer only "signature".
                                            - If none are present, answer only "none".'''
    logger.info(question)
    response = generate(image_path, question)

    return response

if __name__ == '__main__':
    # Example usage
    image_path = "data/performances_data/invalid_data/arval_classic_restitution_images/tmp/ED-913-BL_pv_de_restitution__p1/ED-913-BL_pv_de_restitution__p1_block_0.jpeg"

    response = analyse_block2(image_path)

    print(response)
    # Drawn from https://cloud.google.com/vertex-ai/docs/generative-ai/migrate/migrate-palm-to-gemini
    import os
    import json

    import vertexai
    from vertexai.preview.generative_models import GenerativeModel
    from google.oauth2.service_account import Credentials
    from dotenv import load_dotenv

    load_dotenv()

    '''PROJECT = "..."
    LOCATION = "us-central1"

    # Import credentials 
    service_account_json_string = os.getenv('GCP_SERVICE_ACCOUNT_JSON')
    service_account_info = json.loads(service_account_json_string)
    google_credentials = Credentials.from_service_account_info(service_account_info)
    
    # Initialize Vertex AI
    vertexai.init(project=PROJECT, location=LOCATION, credentials=google_credentials)
    
    # Start prediction
    model = GenerativeModel("gemini-pro")

    responses = model.generate_content("The sun's colour is ", stream=True)

    for response in responses:
        print(response.text)'''