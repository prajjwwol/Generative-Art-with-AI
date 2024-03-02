from neural_network import NeuralStyleTransferModel
from PIL import Image

def create_generative_artwork(content_path, style_path):
    # Load content and style images
    content_img = Image.open(content_path).convert('RGB')
    style_img = Image.open(style_path).convert('RGB')

    # Initialize the neural style transfer model
    model = NeuralStyleTransferModel()

    # Perform neural style transfer
    output_img = model.transfer_style(content_img, style_img)

    return output_img
