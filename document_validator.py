from Levenshtein import distance as l_distance


class ResultValidator:
    def __init__(self, results, plate_number):
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

    def validate_quality(self):
        self.quality_is_ok = self.result['overall_quality'].lower() == 'yes'
    def validate_signatures(self):
        signature_block_2 = self.result['signature_and_stamp_block_2'] in ('both', 'signature')
        signature_block_4 = self.result['signature_and_stamp_block_4'] in ('both', 'signature')

        if signature_block_2 and signature_block_4:
            self.stamps_are_ok = True
        else:
            self.stamps_are_ok = False

    def validate_stamps(self):
        stamp_block_2 = self.result['signature_and_stamp_block_2'] in ('both', 'signature')
        stamp_block_4 = self.result['signature_and_stamp_block_4'] in ('both', 'signature')

        if stamp_block_2 and stamp_block_4:
            self.signatures_are_ok = True
        else:
            self.signatures_are_ok = False

    def validate_mileage(self):
        self.mileage_is_ok = self.result['block_2']['Kilométrage'].isdigit()

    def validate_number_plate_is_filled(self):
        self.number_plate_is_filled = self.result['block_2']['Immatriculé'] != "<EMPTY>"

    def validate_number_plate_is_right(self):
        detected_plate_number = self.result['block_2']['Immatriculé']
        self.number_plate_is_right = l_distance(detected_plate_number, self.plate_number) < 3

    def validate_block4_is_filled_by_company(self, distance_margin=4):
        company_name = self.result['block_4']['Société']
        self.block4_is_filled_by_company = company_name not in ["<EMPTY>", "<NOT_FOUND>"] \
                                           and l_distance(company_name, "Pop Valet") > distance_margin


    def validate_block4_is_filled(self):
        # TODO: Check how we want to define this function
        self.block4_is_filled = any(
            value not in ["<EMPTY>", "<NOT_FOUND>"] for value in self.result['block_4'].values()
        )




    def gather_refused_motivs(self):
        # Initialize an empty list to store the names of variables that are False
        self.refused_causes = []

        # Check each variable and add its name to the list if it's False
        if not self.quality_is_ok:
            self.refused_causes.append('quality_is_not_ok')
        if not self.signatures_are_ok:
            self.refused_causes.append('signature_is_ok')
        if not self.stamps_are_ok:
            self.refused_causes.append('stamp_is_ok')
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
        self.gather_refused_motivs()

        print('self.refused_causes')
        print(self.refused_causes)

        self.validated = self.stamps_are_ok and self.stamps_are_ok and self.mileage_is_ok and self.number_plate_is_filled and self.number_plate_is_right and self.block4_is_filled and self.block4_is_filled_by_company
        return self.validated