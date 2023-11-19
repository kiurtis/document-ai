# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import json

class DocumentAnalyzer:
    def __init__(self, document_name, path_to_document):
        self.document_name = document_name
        self.path_to_document = path_to_document
        self.results = {} #To be filled with results of analysis

    def get_blocks(self):
        raise NotImplementedError

    def analyze_block2_text(self):
        raise NotImplementedError

    def analyze_block4_text(self):
        raise NotImplementedError

    def analyze_block2_signature(self):
        raise NotImplementedError

    def analyze_block4_signature(self):
        raise NotImplementedError

    def analyze(self):
        self.get_blocks()
        self.analyze_block2_text()
        self.analyze_block4_text()
        self.analyze_block2_signature()
        self.analyze_block4_signature()


class ResultValidator:
    def __init__(self, result_json):
        with open(result_json) as f:
            self.result = json.load(f)
        self.result = result_json
        self.signature_is_ok = True
        self.stamp_is_ok = True
        self.mileage_is_ok = True
        self.number_plate_is_filled = True
        self.number_plate_is_right = True
        self.block4_is_filled = True
        self.block4_is_filled_by_company = True


    def validate_signatures(self):
        raise NotImplementedError

    def validate_stamp(self):
        raise NotImplementedError

    def validate_mileage(self):
        raise NotImplementedError

    def validate_number_plate_is_filled(self):
        raise NotImplementedError

    def validate_number_plate_is_right(self):
        raise NotImplementedError

    def validate_block4_is_filled_by_company(self):
        raise NotImplementedError

    def validate_block4_is_filled(self):
        raise NotImplementedError

    def validate(self):
        self.validate_signatures()
        self.validate_stamp()
        self.validate_mileage()
        self.validate_number_plate_is_filled()
        self.validate_number_plate_is_right()
        self.validate_block4_is_filled()
        self.validate_block4_is_filled_by_company()

        self.validated = self.signature_is_ok and self.stamp_is_ok and self.mileage_is_ok and self.number_plate_is_filled and self.number_plate_is_right and self.block4_is_filled and self.block4_is_filled_by_company
        return self.validated
# -

documents = {'document1': {'path':'path/to/document1',
                           'validated': False,
                           'cause': 'block4_is_filled_by_company'},
             'document2': {'path':'path/to/document2',
                           'validated': False,
                           'cause': 'block4_is_filled_by_company'}
             }
full_result_analysis = pd.DataFrame(columns=['document_name', 'true_status', 'predicted_status', 'true_cause', 'predicted_cause'])
for name, info in documents:
    document_analyzer = DocumentAnalyzer(name, info['path'])
    document_analyzer.analyze()
    result_validator = ResultValidator(document_analyzer.results)
    result_validator.validate()
    full_result_analysis = pd.concat(full_result_analysis,
                                     pd.DataFrame({'document_name': name,
                                 'true_status': info['validated'],
                                 'predicted_status': result_validator.validated,
                                 'true_cause': info['cause'],
                                 'predicted_cause': result_validator.cause}))
    print(result_validator.validated)
