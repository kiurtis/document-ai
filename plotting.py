import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import ImageDraw
import PIL

from document_parsing import get_block_coordinates

def plot_boxes_with_text(data):
    """
    Plots given bounding boxes and their associated text.

    Parameters:
    - data (list): A list of tuples, where each tuple contains bounding box coordinates and associated text.

    Example:
    data = [([128, 256, 221, 248], 'LEVEHICULE'), ...]
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    for box, word in data:
        x_min, x_max, y_min, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)
        plt.text(x_min, y_min, word, color='blue', verticalalignment='bottom')

    ax.set_ylim(0, 300)  # adjust these values if necessary
    ax.set_xlim(0, 1200)  # adjust these values if necessary
    plt.show()

def print_blocks_on_document(output, img_path):
    graphical_coordinates = get_block_coordinates(output)
    image = PIL.Image.open(img_path)
    result_image = draw_bounds(image, graphical_coordinates)
    plt.figure(figsize=(15,15))
    plt.imshow(result_image)


def draw_bounds(image, bound):
    draw = ImageDraw.Draw(image)
    for b in bound:
        p0, p1, p2, p3 = [b[0], b[2]], [b[1], b[2]], \
            [b[1], b[3]], [b[0], b[3]]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill='blue', width=2)
    return image


def plot_centroids(bound, img_dims):
    # Extract the dimensions of the image
    img_width, img_height = img_dims

    for b in bound:
        b1 = b[0]
        c = [(b1[0] + b1[1]) / 2, (b1[2] + b1[3]) / 2]
        plt.scatter(c[0], c[1], color='red')

    plt.xlim(0, img_width)
    plt.ylim(0, img_height)  # Start the y-axis at the top to match image coordinates
    plt.gca().set_aspect('equal', adjustable='box')  # Keep the aspect ratio square
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Center of Words')
    plt.show()