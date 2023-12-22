from loguru import logger
from Levenshtein import distance as l_distance

from ai_documents.exceptions import ResultValidationError


class ResultValidator:
    def __init__(self, results, plate_number, valet_name=None, from_concessionaire=None, to_concessionaire=None):
        # with open(result_json) as f:
        #   self.result = json.load(f)
        self.result = results
        self.quality_is_ok = True # Only for ArvalClassicGPTDocumentAnalyzer
        self.signatures_are_ok = True
        self.stamps_are_ok = True
        self.mileage_is_filled = True
        self.number_plate_is_filled = True
        self.number_plate_is_right = True
        self.block4_is_filled = True
        self.block4_is_filled_by_company = True
        self.plate_number = plate_number
        self.valet_name = valet_name
        self.from_concessionaire = from_concessionaire
        self.to_concessionaire = to_concessionaire

    def validate_quality(self):
        self.quality_is_ok = self.result['overall_quality'].lower() == 'yes'

    def validate_signatures(self):
        signature_block_2_condition = self.result['signature_and_stamp_block_2'] in ('both', 'signature')
        signature_block_4_condition = self.result['signature_and_stamp_block_4'] in ('both', 'signature')

        if signature_block_2_condition and signature_block_4_condition:
            self.signatures_are_ok = True
        else:
            self.signatures_are_ok = False

    def validate_stamps(self):
        stamp_block_2_condition = self.result['signature_and_stamp_block_2'] in ('both', 'stamp') or (self.from_concessionaire is False) # If not from concessionaire, no stamp needed
        stamp_block_4_condition = self.result['signature_and_stamp_block_4'] in ('both', 'stamp') or (self.to_concessionaire is False) # If not to concessionaire, no stamp needed

        if stamp_block_2_condition and stamp_block_4_condition:
            self.stamps_are_ok = True
        else:
            self.stamps_are_ok = False

    def validate_mileage(self):
        self.mileage_is_filled = self.result['block_2']['Kilométrage'] not in ["<EMPTY>", "<NOT_FOUND>"]

    def validate_restitution_date(self):
        self.restitution_date_is_filled = self.result['block_2']['Restitué le'] not in ["<EMPTY>", "<NOT_FOUND>"]

    def validate_serial_number(self):
        self.serial_number_is_filled = self.result['block_2']['N° de série'] not in ["<EMPTY>", "<NOT_FOUND>"]

    def validate_number_plate_is_filled(self):
        self.number_plate_is_filled = self.result['block_2']['Immatriculé'] not in ["<EMPTY>", "<NOT_FOUND>"]

    def validate_number_plate_is_right(self):
        detected_plate_number = self.result['block_2']['Immatriculé']
        self.number_plate_is_right = l_distance(detected_plate_number, self.plate_number) < 3

    def validate_block2_is_filled(self):
        self.block2_is_filled = self.number_plate_is_filled & self.mileage_is_filled & self.restitution_date_is_filled &\
                                self.serial_number_is_filled

    def validate_block4_is_filled_by_company(self, distance_margin=4):
        company_name = self.result['block_4']['Société']
        driver_name = self.result['block_4']['Nom et prénom']
        self.block4_is_filled_by_company = l_distance(company_name, "Pop Valet") > distance_margin
        if self.valet_name is not None:
            self.block4_is_filled_by_company = self.block4_is_filled_by_company and (driver_name != self.valet_name)

    def validate_block4_is_filled(self):
        self.driver_name_is_filled = self.result['block_4']['Nom et prénom'] not in ["<EMPTY>", "<NOT_FOUND>"]
        self.telephone_is_filled = self.result['block_4']['Tél'] not in ["<EMPTY>", "<NOT_FOUND>"]
        self.mail_is_filled = self.result['block_4']['E-mail'] not in ["<EMPTY>", "<NOT_FOUND>"]

        if self.to_concessionaire:
            concessionaire_condition = self.result['block_4']['Société'] not in ["<EMPTY>", "<NOT_FOUND>"]
        else:
            concessionaire_condition = True

        self.block4_is_filled = (self.telephone_is_filled or self.mail_is_filled) and self.driver_name_is_filled and \
                                concessionaire_condition

    def gather_refused_motivs(self):
        # Initialize an empty list to store the names of variables that are False
        self.refused_causes = []

        # Check each variable and add its name to the list if it's False
        if not self.quality_is_ok:
            self.refused_causes.append('quality_is_not_ok')
        if not self.signatures_are_ok:
            self.refused_causes.append('signatures_are_not_ok')
        if not self.stamps_are_ok:
            self.refused_causes.append('stamps_are_not_ok')
        if not self.mileage_is_filled:
            self.refused_causes.append('mileage_is_not_ok')
        if not self.number_plate_is_filled:
            self.refused_causes.append('number_plate_is_not_filled')
        if not self.number_plate_is_right:
            self.refused_causes.append('number_plate_is_not_right')
        if not self.block4_is_filled:
            self.refused_causes.append('block4_is_not_filled')
        if not self.block4_is_filled_by_company:
            self.refused_causes.append('block4_is_not_filled_by_company')
        if not self.block2_is_filled:
            self.refused_causes.append('block2_is_not_filled')
        if not self.telephone_is_filled: # Not a problem if the email is filled
            self.refused_causes.append('telephone_is_not_filled')
        if not self.block2_is_filled: # Not a problem if the telephone is filled
            self.refused_causes.append('email_is_not_filled')
        if not self.driver_name_is_filled:
            self.refused_causes.append('driver_name_is_not_filled')
    def validate(self):
        try:
            self.validate_quality()
            self.validate_signatures()
            self.validate_stamps()
            self.validate_mileage()
            self.validate_number_plate_is_filled()
            self.validate_number_plate_is_right()
            self.validate_block4_is_filled()
            self.validate_block4_is_filled_by_company()
            self.validate_restitution_date()
            self.validate_serial_number()
            self.validate_block2_is_filled()
            self.gather_refused_motivs()

            logger.info(f"Refused causes: {self.refused_causes}")

            self.validated = self.stamps_are_ok and self.signatures_are_ok and self.mileage_is_filled \
                             and self.number_plate_is_filled and self.number_plate_is_right and self.block4_is_filled \
                             and self.block4_is_filled_by_company and self.restitution_date_is_filled \
                             and self.serial_number_is_filled
        except Exception as e:
            raise ResultValidationError("Error while validating the result") from e
        return self.validated