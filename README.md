# API for Document Validation

This Flask-based API provides functionalities for uploading and validating documents against specific criteria. 
The API uses LMMs (chatGPT & Gemini currently) document analysis and validation techniques to assess the contents of the uploaded documents.

## Requirements

- Python 3.11
- Flask
- Other dependencies listed in `requirements.txt`

## Setup Instructions

1. **Clone the repository**:
```sh
git clone <repository-url>
cd path/to/api
```
2. **Install dependencies**:
Make sure you have Python 3.x installed on your system. Install the required Python packages using:

```sh
Copy code
pip install -r requirements.txt
```

3. **Set up environment variables**:

Before running the API, ensure the following environment variables are properly set up in your environment. These variables are essential for the API's functionality and integration with various services and hardware optimizations.

- `PLOT_MATCHED_BLOCKS`: Set this to True to enable plotting of matched blocks within the document analysis process. Useful for debugging or visual validation of the algorithm's output.

- `OPENAI_API_KEY`: Your OpenAI API key, required for any interactions with OpenAI services, such as GPT-3. This key is needed for utilizing ChatGPT.

- `GOOGLE_CLOUD_PROJECT`: The ID of your Google Cloud project. This key is needed for utilizing Gemini.

- `DEVICE`: Specify the device used for running the AI models. For example, mps for Apple's Metal Performance Shaders. This allows you to optimize the execution of deep learning models on specific hardware.

- `SAM_MODEL`: The model name for SAM (Sample Adaptive Multimodal). Set this to "vit_b" or any other model identifier you're using. This variable is crucial for defining which AI model the API should use for processing.

- `CREDENTIAL_FILE`: Path to the credentials.ini file containing API access credentials. If not set, credentials.ini in the current directory will be used.
Create a credentials.ini file with the following content, replacing <your_username> and <md5_hashed_password> with your actual credentials:

```
[API]
username = <your_username>
password_hash = <md5_hashed_password>
```

You can generate an MD5 hash of your password using Python: `python -c "import hashlib; print(hashlib.md5('<your_password>'.encode()).hexdigest())"

4. Run the API:

Start the Flask application by running:
```
python api.py
```

## Using the API
### Get API Status
#### Request:
`GET /`

#### Response:

- Status: 200 OK
- Body: {"message": "The API is up and running."}

#### Validate Document
#### Request:
`POST /validate_document`

- Headers: `Authorization: Basic <base64-encoded-username:password>`
- Form-data:
  - file: The document file to upload. 
  - valet_name (optional): Name of the valet. 
  - plate_number (optional): Plate number of the vehicle. 
  - from_concessionaire (optional): Originating concessionaire. 
  - to_concessionaire (optional): Destination concessionaire. 
  - model (optional): Document analysis model, either GPT or GEMINI. Defaults to GPT.

#### Response:

- Status: 200 OK on success, or appropriate error status.
- Body: JSON object containing analysis results, validation details, and any refusal causes.

## Authentication
This API uses basic authentication to secure endpoints. Clients must provide a valid username and password in the request header. Credentials are defined in the credentials.ini file and must be hashed using MD5 for the password.