import torch
from torchvision import transforms
from PIL import Image

# Définir les transformations à appliquer aux images
transform = transforms.Compose([
    transforms.Resize((400, 400)),  # Redimensionner l'image à 256x256 pixels
    transforms.ToTensor()           # Convertir l'image en tenseur PyTorch
])

# Chemin vers l'image à charger
image_path = '0000.png'

# Charger l'image avec PIL
image = Image.open(image_path)

# Appliquer les transformations à l'image
image_tensor = transform(image)

# Ajouter une dimension pour le batch (par exemple, si vous avez une seule image)
image_tensor = image_tensor.unsqueeze(0)

# Afficher la forme du tenseur de l'image
print(image_tensor.shape)