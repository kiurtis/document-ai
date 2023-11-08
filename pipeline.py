from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from document_parsing import get_words_coordinates, remove_word, merge_word_with_following, convert_to_cartesian,find_next_right_word
def run_doctr_model(img_path, **kwargs):
    ''' Load an image and run the model on it.'''
    img = DocumentFile.from_images(img_path)
    model = ocr_predictor(**kwargs)
    result = model(img)
    output = result.export()
    return output


def get_processed_boxes_and_words(img_path, block, verbose=False, **kwargs):
    ''' Load an image, run the model, get the box words pairs and preprocess the needed words.
    kwargs: arguments for ocr_predictor'''
    output = run_doctr_model(img_path, **kwargs)
    image_dims = (
    output['pages'][0]["dimensions"][1], output['pages'][0]["dimensions"][0])

    graphical_coordinates, text_coordinates_and_word = get_words_coordinates(output, verbose)
    converted_boxes = convert_to_cartesian(text_coordinates_and_word, image_dims)
    return converted_boxes


def postprocess_boxes_and_words(converted_boxes, block, safe, verbose, **kwargs):
    """
    kwargs: arguments for merge_word_with_following
    """
    converted_boxes = remove_word(converted_boxes, ":")
    if block == 'block_2':
        converted_boxes = merge_word_with_following(converted_boxes=converted_boxes,
                                                    key_word='Restitué',
                                                    safe=safe,
                                                    verbose=verbose,
                                                    **kwargs)
        converted_boxes = merge_word_with_following(converted_boxes=converted_boxes,
                                                    key_word='Lieu',
                                                    safe=safe,
                                                    verbose=verbose,
                                                    **kwargs)
    if block == 'block_5':
        converted_boxes = merge_word_with_following(converted_boxes=converted_boxes,
                                                    key_word='Nom',
                                                    safe=safe,
                                                    verbose=verbose,
                                                    **kwargs)
    return converted_boxes


#Pipeline functions for unguided bloc division: 


def get_processed_boxes_and_words_unguided_bloc(img_path, verbose=False, **kwargs):
    ''' Load an image, run the model, get the box words pairs and preprocess the needed words.
    kwargs: arguments for ocr_predictor'''
    output = run_doctr_model(img_path, **kwargs)
    image_dims = (
    output['pages'][0]["dimensions"][1], output['pages'][0]["dimensions"][0])

    graphical_coordinates, text_coordinates_and_word = get_words_coordinates(output, verbose)
    converted_boxes = convert_to_cartesian(text_coordinates_and_word, image_dims)
    return converted_boxes,image_dims


def postprocess_boxes_and_words_unguided_bloc(converted_boxes, safe, verbose, **kwargs):
    """
    kwargs: arguments for merge_word_with_following
    """
    converted_boxes = remove_word(converted_boxes, ":")

    if find_next_right_word(converted_boxes,'Restitué') not in ("<NOT_FOUND>", "<EMPTY>"): 
        converted_boxes = merge_word_with_following(converted_boxes=converted_boxes,
                                                    key_word='Restitué',
                                                    safe=safe,
                                                    verbose=verbose,
                                                    **kwargs)
    if find_next_right_word(converted_boxes,'Lieu') not in ("<NOT_FOUND>", "<EMPTY>"): 
        converted_boxes = merge_word_with_following(converted_boxes=converted_boxes,
                                                    key_word='Lieu',
                                                    safe=safe,
                                                    verbose=verbose,
                                                    **kwargs)
    if find_next_right_word(converted_boxes,'Nom') not in ("<NOT_FOUND>", "<EMPTY>"): 
        converted_boxes = merge_word_with_following(converted_boxes=converted_boxes,
                                                    key_word='Nom',
                                                    safe=safe,
                                                    verbose=verbose,
                                                    **kwargs)
    return converted_boxes

