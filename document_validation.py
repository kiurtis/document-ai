from loguru import logger
from Levenshtein import distance as l_distance


class ResultValidator:
    def __init__(self, results, plate_number, valet_name=None, from_concessionaire=None, to_concessionaire=None):
        # with open(result_json) as f:
        #   self.result = json.load(f)
        self.result = results
        self.quality_is_ok = True # Only for ArvalClassicGPTDocumentAnalyzer
        self.signatures_are_ok = True
        self.stamps_are_ok = True
        self.mileage_is_ok = True
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
        signature_block_2 = self.result['signature_and_stamp_block_2'] in ('both', 'signature')
        signature_block_4 = self.result['signature_and_stamp_block_4'] in ('both', 'signature')

        if signature_block_2 and signature_block_4:
            self.signatures_are_ok = True
        else:
            self.signatures_are_ok = False

    def validate_stamps(self):
        stamp_block_2_condition = self.result['signature_and_stamp_block_2'] in ('both', 'stamp') or (not self.from_concessionaire) # If not from concessionaire, no stamp needed
        stamp_block_4_condition = self.result['signature_and_stamp_block_4'] in ('both', 'stamp') or (not self.to_concessionaire) # If not to concessionaire, no stamp needed

        if stamp_block_2_condition and stamp_block_4_condition:
            self.stamps_are_ok = True
        else:
            self.stamps_are_ok = False

    def validate_mileage(self):
        self.mileage_is_ok = self.result['block_2']['Kilométrage'] not in ["<EMPTY>", "<NOT_FOUND>"]

    def validate_restitution_date(self):
        self.restitution_date_is_ok = self.result['block_2']['Restitué le'] not in ["<EMPTY>", "<NOT_FOUND>"]

    def validate_serial_number(self):
        self.serial_number_is_ok = self.result['block_2']['N° de série'] not in ["<EMPTY>", "<NOT_FOUND>"]

    def validate_number_plate_is_filled(self):
        self.number_plate_is_filled = self.result['block_2']['Immatriculé'] not in ["<EMPTY>", "<NOT_FOUND>"]

    def validate_number_plate_is_right(self):
        detected_plate_number = self.result['block_2']['Immatriculé']
        self.number_plate_is_right = l_distance(detected_plate_number, self.plate_number) < 3

    def validate_block2_is_filled(self):
        self.block2_is_filled = self.number_plate_is_filled & self.mileage_is_ok & self.restitution_date_is_ok & self.serial_number_is_ok

    def validate_block4_is_filled_by_company(self, distance_margin=4):
        company_name = self.result['block_4']['Société']
        driver_name = self.result['block_4']['Nom et prénom']
        self.block4_is_filled_by_company = company_name not in ["<EMPTY>", "<NOT_FOUND>"] \
                                           and l_distance(company_name, "Pop Valet") > distance_margin
        if self.valet_name is not None:
            self.block4_is_filled_by_company = self.block4_is_filled_by_company and (driver_name != self.valet_name)

    def validate_block4_is_filled(self):
        name_condition = self.result['block_4']['Nom et prénom'] not in ["<EMPTY>", "<NOT_FOUND>"]
        coordinates_condition = self.result['block_4']['Tél'] not in ["<EMPTY>", "<NOT_FOUND>"] and \
                                self.result['block_4']['E-mail'] not in ["<EMPTY>", "<NOT_FOUND>"]

        if self.to_concessionaire:
            concessionaire_condition = self.result['block_4']['Société'] not in ["<EMPTY>", "<NOT_FOUND>"]
        else:
            concessionaire_condition = True
        self.block4_is_filled = coordinates_condition and name_condition and concessionaire_condition

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
        if not self.mileage_is_ok:
            self.refused_causes.append('mileage_is_not_ok')
        if not self.number_plate_is_filled:
            self.refused_causes.append('number_plate_is_not_filled')
        if not self.number_plate_is_right:
            self.refused_causes.append('number_plate_is_not_right')
        if not self.block4_is_filled:
            self.refused_causes.append('block4_is_not_filled')
        if not self.block4_is_filled_by_company:
            self.refused_causes.append('block4_is_not_filled_by_company')



    def validate(self):
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

        self.validated = self.stamps_are_ok and self.signatures_are_ok and self.mileage_is_ok \
                         and self.number_plate_is_filled and self.number_plate_is_right and self.block4_is_filled \
                         and self.block4_is_filled_by_company and self.restitution_date_is_ok and self.serial_number_is_ok
        return self.validated