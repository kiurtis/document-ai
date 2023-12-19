import os
import click
from pathlib import Path
from loguru import logger
from ai_documents.analysis.entities import ArvalClassicGPTDocumentAnalyzer
from ai_documents.validation.entities import ResultValidator

# Define your command with click
@click.command()
@click.option('--valet_name', default=None, help='Valet name.')
@click.option('--from_concessionaire', default=None, help='From concessionaire.')
@click.option('--to_concessionaire', default=None, help='To concessionaire.')
@click.option('--input_file_path', default='data/performances_data/invalid_data/arval_classic_restitution_images/FY-915-LM_PVderestitution.jpeg', help='Path to the input file.')

def main(valet_name, from_concessionaire, to_concessionaire, input_file_path):
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
        document_analyzer = ArvalClassicGPTDocumentAnalyzer(file_name, input_file_path, hyperparameters)
        document_analyzer.analyze()
        logger.info(f"Result: {document_analyzer.results}")

        result_validator = ResultValidator(document_analyzer.results, plate_number=plate_number, valet_name=valet_name, from_concessionaire=from_concessionaire, to_concessionaire=to_concessionaire)
        result_validator.validate()

    except Exception as e:
        logger.error(f"Error {e} while analyzing {file_name}")
        raise e

# Entry point for the script
if __name__ == '__main__':
    main()
