import json
from pathlib import Path
from typing import Union

import torch
import torchvision
from PIL import Image
import torch.nn.functional as F
from torchvision.models import list_models, get_model
from torchvision.models import ResNet18_Weights, ResNet50_Weights

# Load imagenet labels
working_dir = Path(__file__).parent
with open(working_dir / "data/imagenet_class_index.json") as f:
    img_label_id, img_label = zip(*json.load(f).values())


available_models = list_models(module=torchvision.models)
model_transforms = {
    "resnet18": ResNet18_Weights.IMAGENET1K_V1.transforms,
    "resnet50": ResNet50_Weights.IMAGENET1K_V1.transforms
}


def create_adversary(image: Union[torch.Tensor, Path], 
                     model_name: str, 
                     target: str, 
                     transform: callable=None, 
                     weights: str="IMAGENET1K_V1",
                     device=None):
    '''
    Image can be a Tensor (in which case no transform will be called)
    Image can also be a string, in which case the user can pass in a custom transform for their model of choice
    Transforms for resnet18 and resnet50 are provided by default. 
    Expectation for a custom transform is to be a standard pytorch style transform/composition object,
    essentially a function/callable that handles any image processing required for your model of choice.
    model_name should be one of the available models in torchvision see https://pytorch.org/vision/stable/models.html
    the code currently assumes your model of choice has pretrained weights available with the Imagenet1k_v1 enum, otherwise you can pass your own.
    '''
    if target not in img_label:
        raise ValueError("Target must be one of the 1k Imagenet labels")
    
    if model_name not in available_models:
        raise ValueError("Model Name must be one of the torchvision available models")
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, weights=weights).to(device)
    model.eval()
    if isinstance(image, Path):
        if not image.exists():
            raise ValueError("Image not found")
        transform = model_transforms.get(model_name, None) if transform is None else transform
        if transform is None:
            raise ValueError("Model is not resnet18/50 and no transform was supplied.")
        transform = transform() # Instantiate the partial
        base_image = Image.open(image)
        transformed_image =  transform(base_image).to(device).unsqueeze(0)
        transformed_image.requires_grad = True
        pre_pertubed_pred = model(transformed_image)
        target = img_label.index(target)
        if pre_pertubed_pred.argmax() == target:
            print("Warning, model already classifies image as target, no perturbation done")
            return base_image
        loss = F.nll_loss(pre_pertubed_pred, torch.tensor(target, dtype=torch.int64).to(device).unsqueeze(0))
        model.zero_grad
        loss.backward()
        epsilon = 0.007
        image_gradient = transformed_image.grad.data
        perturbed_image = transformed_image - epsilon * image_gradient.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return img_label[model(perturbed_image).flatten().argmax().item()]
        
if __name__=="__main__":
    image = Path("/home/karmanya/Documents/repos/Leap/data/imagenette2-320/val/n01440764/ILSVRC2012_val_00009111.JPEG")
    print(create_adversary(image=image, model_name="resnet18", target="stingray"))
        
        