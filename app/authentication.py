import configparser
import hashlib
import os

from flask_httpauth import HTTPBasicAuth


def hash_password_with_md5(password):
    if password is None:
        return None
    return hashlib.md5(password.encode()).hexdigest()


auth = HTTPBasicAuth()


@auth.verify_password
def verify_password(username, password):
    config = configparser.ConfigParser()
    credential_file = os.getenv("CREDENTIAL_FILE", "credentials.ini")
    config.read(credential_file)
    return (
        username == config["API"]["username"]
        and hash_password_with_md5(password) == config["API"]["password_hash"]
    )
