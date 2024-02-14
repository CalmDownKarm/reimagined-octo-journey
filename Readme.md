### Readme
Context:  This is a take home assessment- the goal is to create a function/library that generates adversarial examples given an image and a target label. 

I tried 2 approaches first was the targeted fast gradient sign method, however, I wasn't able to get it working reliably. So I switched to a Projected Gradient Descent approach which does work for me.

I tested my approach with Resnet18 and Imagenette but it should work with any model pretrained on Imagenet. 

test_adversary shows an example of how the function in main.py can be used - one caveat, the function returns a Tensor as an output - this is because I haven't correctly handled the transformation logic - at the moment, when an image is passed to the model, it gets transformed (Resized, Cropped and Normalized) to be compatible with Resnet18, and the perturbations are calculated relative to this transformed sample. If I had more time, I would change the function such that it writes to PIL and loads from disk at every iteration of PGD, that way, the perturbations will be robust to writing the image to disk and running through the input pipeline. 