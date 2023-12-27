import os
import click
from pathlib import Path
import json
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
    plate_number = file_name.split('_')[0]

    # Random hyperparameters (TODO: Remove or modify as needed)
    hyperparameters = {
        'det_arch': "db_resnet50",
        'reco_arch': "crnn_mobilenet_v3_large",
        'pretrained': True,
        'distance_margin': 5,  # find_next_right_word for words_similarity
        'max_distance':  400,  # find_next_right_word
        'minimum_overlap': 10  # find_next_right_word for _has_overlap
    }

    try:

        document_analyzer = ArvalClassicGPTDocumentAnalyzer(file_name, input_file_path,
                                                            hyperparameters
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
        output_directory = Path(input_file_path).parent
        json_output_path = output_directory / f"{file_name}_json.json"

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
