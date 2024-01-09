import os
import json
import sys

import click
from pathlib import Path
from loguru import logger
from ai_documents.analysis.entities import ArvalClassicGPTDocumentAnalyzer
from ai_documents.validation.entities import ResultValidator


# Define your command with click
@click.command()
@click.option('--valet_name', default=None, help='Valet name.')
@click.option('--plate_number', default=None, help='Plate number.')
@click.option('--from_concessionaire', default=None, help='From concessionaire.')
@click.option('--to_concessionaire', default=None, help='To concessionaire.')
@click.option('--input_file_path', help='Path to the input file.')
def main(valet_name, plate_number, from_concessionaire, to_concessionaire, input_file_path):

    path = Path(input_file_path)
    file_name = path.name

    try:

        document_analyzer = ArvalClassicGPTDocumentAnalyzer(file_name, input_file_path,
                                                            #hyperparameters=None
                                                            )

        document_analyzer.analyze()

        result_validator = ResultValidator(document_analyzer.results, plate_number=plate_number,
                                               valet_name=valet_name, from_concessionaire=from_concessionaire,
                                               to_concessionaire=to_concessionaire)
        result_validator.validate()

        logger.info(f"Result: {document_analyzer.results}")
        dict_to_save = document_analyzer.results.copy()
        dict_to_save['refused_causes'] = result_validator.refused_causes
        dict_to_save['Validated'] = result_validator.validated
        # Save results as JSON
        output_directory = Path("results")
        json_output_path = output_directory / f"{file_name}_results.json"

        logger.info('*************** FINAL RESULTS ***************')
        logger.info(f"Refused causes: {result_validator.refused_causes}")
        logger.info(f"Document valid: {result_validator.validated}")

        with open(json_output_path, 'w') as json_file:
            json.dump(dict_to_save, json_file, indent=4)
            logger.info(f"Results saved to {json_output_path}")



    except Exception as e:
        logger.error(f"Error {e} while analyzing {file_name}")
        raise e

# Entry point for the script
if __name__ == '__main__':
    main()

