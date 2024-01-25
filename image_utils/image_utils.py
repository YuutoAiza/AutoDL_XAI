import torch
import PIL.Image as IMAGE
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

def apply_transform(image, size=224, means= torch.tensor([0.485, 0.456, 0.406]), stds=torch.tensor([0.229, 0.224, 0.225]), cifar =  False):
    if cifar == True:
        size = 32

    if not isinstance(image, IMAGE.Image):
        image = F.to_pil_image(image)

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    tensor = transform(image).unsqueeze(0)

    tensor.requires_grad = True

    return tensor

def detransform(tensor):
    means, stds = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    denormalized = transforms.Normalize(-1 * means / stds, 1.0 / stds)(tensor)

    return denormalized


def load_image(image_path,rgb=False):
    if rgb:
        return IMAGE.open(image_path).convert('RGB')
    else:
        return IMAGE.open(image_path).convert('L')

def image_to_tensors(opt, cifar = False):

    image=load_image(opt.image,rgb=True)

    image=apply_transform(image, cifar=cifar)

    return image
