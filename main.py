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
    img_label = [label.lower() for label in img_label]

def perturb_image(image, eta, max_eps):
    eta = torch.clamp(eta, -max_eps, max_eps)
    image = torch.clamp(image + eta, 0, 1)
    return image


def convert_tensor_to_PIL(image: torch.Tensor):
    """TODO: If an image has been transformed, reverse it."""
    transform = transforms.ToPILImage()
    assert len(image.shape) == 4
    return transform(image.squeeze(0))  # Remove batch dimension


def create_adversary(
    image: Union[Path, torch.Tensor],
    model: torch.nn.Module,
    target: str,
    transform: callable = None,
    device: torch.device = None,
    max_eps=5e-4,
    MAX_ITERS=100,
):
    """
    image: can be either a torch tensor (Should already be transformed for the corresponding model.),
           OR it can be a path object to an image file in which case transform should be passed.
    target: string containing desired target class. Needs to be one of the Imagenet1K classes.
    transform: Callable, should accept a PIL as input and return a Pytorch Tensor compatible with your model of choice.
               Ignored if input is a tensor
    device: if specified, will put the model and images on the torch.device
    max_eps: float value between 0 and 1 - that's the maximum epsilon value for the perturbation.
    MAX_ITERS: integer value, that's the maximum number of update steps to take before giving up.
    """
    if isinstance(image, Path):
        image = Image.open(str(image))
        image.save(str(working_dir / "Pre_transform.jpeg"))
        if transform is None:
            raise ValueError("Image is not a tensor and Transform not provided")
        image = transform(image).to(device).unsqueeze(0)

    backup = image.clone()
    if target.lower() not in img_label:
        raise ValueError("Target should be a imagenet label")
    target = (
        torch.tensor(img_label.index(target), dtype=torch.int64).to(device).unsqueeze(0)
    )
    model.eval()

    model.requires_grad = False

    for _ in tqdm(range(MAX_ITERS), total=MAX_ITERS):
        image = image.clone().detach().requires_grad_(True)
        model_pred = model(image)
        if model_pred.argmax() == target:
            return image
        loss = -F.cross_entropy(model_pred, target)  # Minimize loss of target label
        loss.backward()
        eta = max_eps * torch.sign(image.grad)
        image = perturb_image(image, eta, max_eps)

    print(
        "Max iters reached, and adversarial attack was unsuccessful. Increase MAX_ITERS, or tweak max_eps"
    )
    return image
