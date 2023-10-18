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

# + jupyter={"outputs_hidden": false} pycharm={"is_executing": true}
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from Levenshtein import distance

# +
from pdf2image import convert_from_path

def convert_pdf_to_jpg(pdf_path, output_folder=None, dpi=300):
    # Convert PDF to list of images
    images = convert_from_path(pdf_path, dpi)

    # Save images to the output directory
    for i, image in enumerate(images):
        image_name = f"output_page_{i + 1}.jpg"
        if output_folder:
            image_name = f"{output_folder}/{image_name}"
        image.save(image_name, "JPEG")

# Example usage
pdf_path = "/Users/pipobimbo/Desktop/Caliente_work/OCR_test/doctr/pdf/pvl_GM266PC.pdf"
convert_pdf_to_jpg(pdf_path)
# !pwd
img_path = "/Users/pipobimbo/Desktop/Caliente_work/OCR_test/doctr/output_page_1.jpg" #Specify your image path here
img = DocumentFile.from_images(img_path)
result = model(img)
output = result.export()
# -

result.show(img)

# +
import math

def convert_coordinates(geometry, page_dim):
    len_x = page_dim[1]
    len_y = page_dim[0]
    (x_min, y_min) = geometry[0]
    (x_max, y_max) = geometry[1]
    x_min = math.floor(x_min * len_x)
    x_max = math.ceil(x_max * len_x)
    y_min = math.floor(y_min * len_y)
    y_max = math.ceil(y_max * len_y)
    return [x_min, x_max, y_min, y_max]

def get_coordinates(output):
    page_dim = output['pages'][0]["dimensions"]
    text_coordinates = []
    i = 0
    for obj1 in output['pages'][0]["blocks"]:              
        converted_coordinates = convert_coordinates(
                                           obj1["geometry"],page_dim
                                          )
        print("{}: {}".format(converted_coordinates,
                                      i
                                      )
                     )
        text_coordinates.append(converted_coordinates)
        i+=1
    return text_coordinates

graphical_coordinates = get_coordinates(output)

# +
import PIL
from PIL import ImageDraw
import matplotlib.pyplot as plt

def draw_bounds(image, bound):
    draw = ImageDraw.Draw(image)
    for b in bound:
        p0, p1, p2, p3 = [b[0],b[2]], [b[1],b[2]], \
                         [b[1],b[3]], [b[0],b[3]]
        draw.line([*p0,*p1,*p2,*p3,*p0], fill='blue', width=2)
    return image

image = PIL.Image.open(img_path)
result_image = draw_bounds(image, graphical_coordinates)
plt.figure(figsize=(15,15))
plt.imshow(result_image)
# -


