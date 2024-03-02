import torch
import logging 
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import VGG19_Weights


class NeuralStyleTransferModel:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(self.device).eval()

    def get_features(self, image, model, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '28': 'conv5_1'}
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def gram_matrix(self, tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram 

    def transfer_style(self, content_img, style_img, num_steps=300, style_weight=100000, content_weight=1):
        content_img = transforms.functional.to_tensor(content_img).unsqueeze(0).to(self.device)
        style_img = transforms.functional.to_tensor(style_img).unsqueeze(0).to(self.device)

        content_features = self.get_features(content_img, self.vgg)
        style_features = self.get_features(style_img, self.vgg)
        style_grams = {layer: self.gram_matrix(style_features[layer]) for layer in style_features}

        target = content_img.clone().requires_grad_(True)
        optimizer = optim.Adam([target], lr=0.003)

        for i in range(num_steps):
            target_features = self.get_features(target, self.vgg)
            content_loss = torch.mean((target_features['conv4_1'] - content_features['conv4_1']) ** 2)

            style_loss = 0
            for layer in style_grams:
                target_feature = target_features[layer]
                target_gram = self.gram_matrix(target_feature)
                style_gram = style_grams[layer]
                layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
                style_loss += layer_style_loss / (target_feature.shape[1] * target_feature.shape[2] * target_feature.shape[3])

            total_loss = content_weight * content_loss + style_weight * style_loss
            optimizer.zero_grad()
            if i < num_steps - 1:
                total_loss.backward(retain_graph=True)
            else:
                total_loss.backward()
            optimizer.step()
            logging.info(f"Step {i+1}/{num_steps}: Style transfer in progress...")
            
        logging.info("Style transfer complete.")
        return transforms.functional.to_pil_image(target.squeeze().cpu().detach())

