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

# # First Test

# +
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import PIL
from PIL import ImageDraw
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from Levenshtein import distance
import math

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
#pdf_path = "/Users/pipobimbo/Desktop/Caliente_work/OCR_test/doctr/pdf/pvl_GM266PC.pdf"
#convert_pdf_to_jpg(pdf_path)


def get_block_coordinates(output):
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

def get_words_coordinates(output):
    page_dim = output['pages'][0]["dimensions"]
    text_coordinates = []
    text_coordinates_and_word = []
    for obj1 in output['pages'][0]["blocks"]:
        for obj2 in obj1["lines"]:
            for obj3 in obj2["words"]:                
                converted_coordinates = convert_coordinates(obj3["geometry"],page_dim)
                
                print("{}: {}".format(converted_coordinates,obj3["value"]))
                text_coordinates.append(converted_coordinates)
                text_coordinates_and_word.append((converted_coordinates,obj3["value"]))
    return text_coordinates,text_coordinates_and_word

def draw_bounds(image, bound):
    draw = ImageDraw.Draw(image)
    for b in bound:
        p0, p1, p2, p3 = [b[0],b[2]], [b[1],b[2]], \
                         [b[1],b[3]], [b[0],b[3]]
        draw.line([*p0,*p1,*p2,*p3,*p0], fill='blue', width=2)
    return image

def plot_centers2(bound, img_dims):
    # Extract the dimensions of the image
    img_width, img_height = img_dims
    
    for b in bound:
        b1 = b[0]
        c = [(b1[0] + b1[1])/2, (b1[2] + b1[3])/2]
        plt.scatter(c[0], c[1], color='red')
        
    plt.xlim(0, img_width)
    plt.ylim(0,img_height)  # Start the y-axis at the top to match image coordinates
    plt.gca().set_aspect('equal', adjustable='box')  # Keep the aspect ratio square
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Center of Words')
    plt.show()

def convert_to_cartesian(bound, img_dims):
    _, img_height = img_dims
    cartesian_bound = []

    for box, word in bound:
        # Convert y-values by subtracting them from the image height
        new_box = [box[0], box[1], img_height - box[3], img_height - box[2]]
        cartesian_bound.append((new_box, word))

    return cartesian_bound



def find_word_to_right(bounding_boxes, search_word):
    # Find the bounding box for the search word
    search_box = None
    for box, word in bounding_boxes:
        if word == search_word:
            search_box = box
            break
    
    # If the word isn't found, return None
    if search_box is None:
        return None

    # Variables to store the closest word and its distance
    closest_word = None
    closest_distance = float('inf')

    for box, word in bounding_boxes:
        # Check if the word's y-coordinates overlap with the search word's
        if not (box[3] < search_box[2] or box[2] > search_box[3]):
            # Check if the word is to the right of the search word
            if box[0] > search_box[1]:
                distance = box[0] - search_box[1]  # Calculate distance to the right
                if distance < closest_distance:
                    closest_distance = distance
                    closest_word = word

    return closest_word


# +
img_path = "/Users/pipobimbo/Desktop/Caliente_work/OCR_test/doctr/test2.png"
img = DocumentFile.from_images(img_path)

model = ocr_predictor(det_arch = 'db_resnet50',reco_arch = 'crnn_vgg16_bn',pretrained = True)

result = model(img)
output = result.export()

# Get and print the blocs :
graphical_coordinates = get_block_coordinates(output)


image = PIL.Image.open(img_path)
result_image = draw_bounds(image, graphical_coordinates)
plt.figure(figsize=(15,15))
plt.imshow(result_image)


# +
#getting all the world and converting them to cartesian coords
graphical_coordinates,text_coordinates_and_word = get_words_coordinates(output)
image_dims = (output['pages'][0]["dimensions"][1], output['pages'][0]["dimensions"][0])  # Replace with your image dimensions
print(image_dims)

converted_boxes = convert_to_cartesian(text_coordinates_and_word, image_dims)

print(converted_boxes)
#print(text_coordinates_and_word)
# -

search_word = "Couleur"
print(find_word_to_right(converted_boxes, search_word))

print(converted_boxes)
plot_centers2(converted_boxes, image_dims)


