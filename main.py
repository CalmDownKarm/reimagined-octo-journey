import json
import torch
from pathlib import Path
from typing import Union
from tqdm.auto import tqdm


import torchvision
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
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
        epsilon = 1
        image_gradient = transformed_image.grad.data
        perturbed_image = transformed_image - epsilon * image_gradient.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return model(perturbed_image).flatten().argmax()
def perturb_image(image, eta, max_eps):
    eta = torch.clamp(eta, -max_eps, max_eps)
    image = torch.clamp(image + eta, 0, 1)
    return image

def convert_tensor_to_PIL(backup: torch.Tensor, image: torch.Tensor):
    ''' TODO: If an image has been transformed, reverse it.'''
    transform = transforms.ToPILImage()
    return transform(backup.squeeze(0)), transform(image.squeeze(0)) # Remove batch dimension

def run_pgd(image: Union[Path, torch.Tensor], model: torch.nn.Module, target: str,  transform:callable=None, device:torch.device=None, max_eps=5e-4, MAX_ITERS=100):
    '''
    image: can be either a torch tensor (Should already be transformed for the corresponding model.),
           OR it can be a path object to an image file in which case transform should be passed. 
    target: string containing desired target class. Needs to be one of the Imagenet1K classes. 
    transform: Callable, should accept a PIL as input and return a Pytorch Tensor compatible with your model of choice. 
               Ignored if input is a tensor
    device: if specified, will put the model and images on the torch.device
    max_eps: float value between 0 and 1 - that's the maximum epsilon value for the perturbation.
    MAX_ITERS: integer value, that's the maximum number of update steps to take before giving up. 
    '''
    if isinstance(image, Path):
        image = Image.open(str(image))
        if transform is None:
            raise ValueError("Image is not a tensor and Transform not provided")
        image = transform(image).to(device).unsqueeze(0)
        toPil = transforms.ToPILImage()
        toPil(image.squeeze(0)).save("before.jpeg")
    
    backup = image.clone()

    target = torch.tensor(img_label.index(target), dtype=torch.int64).to(device).unsqueeze(0)
    model.eval()
    
    model.requires_grad = False
    random_initialization = torch.clamp(torch.rand_like(image), 0, 1)
    eta_init = (max_eps * 0.5) * random_initialization
    image = perturb_image(image, eta_init, max_eps)
    for _ in tqdm(range(MAX_ITERS), total=MAX_ITERS):
        image = image.clone().detach().requires_grad_(True)
        model_pred = model(image)
        if model_pred.argmax() == target:
            return convert_tensor_to_PIL(backup, image)
        loss = -F.cross_entropy(model_pred, target) # Minimize loss of target label
        loss.backward()
        eta = max_eps* torch.sign(image.grad)
        image = perturb_image(image, eta, max_eps)
    
    print("Max iters reached, and adversarial attack was unsuccessful. Increase MAX_ITERS, or tweak max_eps")    
    return convert_tensor_to_PIL(backup, image)




if __name__=="__main__":
    
    working_dir = Path(__file__).parent
    image = working_dir / "data/imagenette2-320/val/n03445777/ILSVRC2012_val_00008161.JPEG"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model("resnet18", weights="IMAGENET1K_V1")
    model.to(device)
    transform = model_transforms["resnet18"]()
    backup, image = run_pgd(image, model, "goldfish", transform, device)
    backup.save("backup.jpeg")
    image.save("image.jpeg")
        



    



    

    
    
