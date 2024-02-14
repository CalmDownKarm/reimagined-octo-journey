from pathlib import Path

from main import create_adversary

def test_tfgsm():    
    image = Path("/home/karmanya/Documents/repos/Leap/data/imagenette2-320/val/n01440764/ILSVRC2012_val_00009111.JPEG")
    assert create_adversary(image=image, model_name="resnet18", target="stingray") == "stingray"
        
