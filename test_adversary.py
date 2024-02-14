import json
import torch
import random
from pathlib import Path
from typing import Union
from tqdm.auto import tqdm

import torchvision
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import list_models, get_model
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from main import create_adversary, convert_tensor_to_PIL


def test_one_image():
    '''Take a random image from Imagenette and convert it to the label Goldfish'''
    working_dir = Path(__file__).parent
    imagenette_images = list(
        (working_dir / "data/imagenette2-320/val").glob("*/*.JPEG")
    )
    image = random.choice(imagenette_images)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model("resnet18", weights="IMAGENET1K_V1")
    model.to(device)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    perb_image = create_adversary(image, model, "goldfish", transform, device)
    
    assert model(perb_image).argmax().cpu().item() == 1
    output_image_path = working_dir / "test_one_image.jpeg"
    perb_image_pil = convert_tensor_to_PIL(perb_image)
    perb_image_pil.save(str(output_image_path))
