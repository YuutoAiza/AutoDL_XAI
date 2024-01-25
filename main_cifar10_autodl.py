import argparse
import os.path as OSPATH
import torchvision.models as models
from metrics import adcc as ADCC
from image_utils import image_utils as IMUT
import torchcam.cams as CAMS
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torch import nn
import cv2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision.transforms as transforms
import torchvision
import torchinfo
import os

class_list = {"airplane":0, "automobile":1, "bird":2, "cat":3, "deer":4, "dog":5, "frog":6, "horse":7, "ship":8, "truck":9}
accuracy_list = {1835:38.33, 2581:90.68, 3894:77.93, 9987:89.12, 12384:90.17, 13458:82.31, 14456:87.87, 15035:89.48, "ResNet_110":90.61}
MAX = 1000
DEVICE = torch.device("cuda")
BATCH_SIZE = 1001
ARCH_NUM = 15035
VISUALIZE = 150
explanation_method = 3

AUTO = True

testloader = torch.utils.data.DataLoader(
torchvision.datasets.CIFAR10('cifar10_data', train=False, transform=transforms.Compose([
                transforms.ToTensor()
            ])),
batch_size=BATCH_SIZE, shuffle=False)
with torch.no_grad():

    for j, inputs in enumerate(testloader, 0):
        _, labels = inputs
        print(labels)
        if j == 0:
            break

def numpy_normalization(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x_scaled = (x - x_min) / (x_max - x_min)
    
    return x_scaled

def ScoreCAM_extracor(image,model,classidx=None):
    scam=CAMS.ScoreCAM(model = model, input_shape = (3, 32 ,32))
    with torch.no_grad(): out = model(image)
    print("output shape : {}".format(np.shape(out)))
    
    sort, idx = torch.sort(out, descending = True)
    print(idx[0])

    if classidx is None:
        classidx=out.max(1)[1].item()
    print("classidx : {}".format(classidx))

    salmap=scam(class_idx=classidx, scores=out)
    return F.interpolate(salmap.unsqueeze(0).unsqueeze(0), (32, 32), mode='bilinear', align_corners=False)

def GradCAM_extracor(image, model, target_layers, classidx=None):
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=image, targets=classidx)
    newaxis_grayscale_cams = grayscale_cams[np.newaxis, :, :, :]
    newaxis_grayscale_cams = torch.from_numpy(newaxis_grayscale_cams.astype(np.float32)).clone()
    
    return newaxis_grayscale_cams

def EigenGradCAM_extracor(image, model, target_layers, classidx=None):
    with EigenGradCAM(model=model, target_layers=target_layers) as cam:
        # grayscale_cams = cam(input_tensor=image, targets=classidx, eigen_smooth = False, aug_smooth = False)
        grayscale_cams = cam(input_tensor=image, targets=classidx)
        print("grayscale_cams", grayscale_cams)
    newaxis_grayscale_cams = grayscale_cams[np.newaxis, :, :, :]
    newaxis_grayscale_cams = torch.from_numpy(newaxis_grayscale_cams.astype(np.float32)).clone()
    
    return newaxis_grayscale_cams

def ScoreCAM_extracor_2(image, model, target_layers, classidx=None):
    with ScoreCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=image, targets=classidx)
    newaxis_grayscale_cams = grayscale_cams[np.newaxis, :, :, :]
    newaxis_grayscale_cams = torch.from_numpy(newaxis_grayscale_cams.astype(np.float32)).clone()

def GradCAMPlusPlus_extracor(image, model, target_layers, classidx=None):
    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=image, targets=classidx)
    newaxis_grayscale_cams = grayscale_cams[np.newaxis, :, :, :]
    newaxis_grayscale_cams = torch.from_numpy(newaxis_grayscale_cams.astype(np.float32)).clone()

    return newaxis_grayscale_cams

def visualize_score(path, correct, pred, adcc, avgdrop, coh, com, out_pred, out_correct, confidence_in, confidence_exp):
    key_correct = [key for key, value in class_list.items() if value == correct]
    key_pred = [key for key, value in class_list.items() if value == pred]
    img = cv2.imread(path)
    np.set_printoptions(precision = 4)
    cv2.putText(img, "correct   : {} - {}".format(key_correct, round(out_correct.item()*100, 2)), (458, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "prediction : {} - {}".format(key_pred, round(out_pred.item()*100, 2)), (458, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "Comlexity : {}".format(f"{com * 100:.5f}"), (458, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Average Drop : {}".format(f"{avgdrop * 100:.5f}"), (458, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "confidence input : {}".format(f"{confidence_in * 100:.5f}"), (458, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 241, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "confidence explain : {}".format(f"{confidence_exp * 100:.5f}"), (458, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 241, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "Coherency : {}".format(f"{coh * 100:.5f}"), (458, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "ADCC : {}".format(f"{adcc * 100:.5f}"), (458, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)

def main(IMAGE_NUM):
    TARGET_CLASS = labels[IMAGE_NUM].item()
    # directory_img_test = "./example/both.png"
    # directory_img_test = "./example/image_resize.png"
    directory_img_test = "./imgs_cifar10/resize_test{}.png".format(IMAGE_NUM)

    img = np.array(Image.open(directory_img_test).convert('RGB'))
    img = np.float32(img) / 255
    input_tensor = preprocess_image(img, mean=[x / 255 for x in [125.3, 123.0, 113.9]], std=[x / 255 for x in [63.0, 62.1, 66.7]])
    input_tensor = input_tensor.to(DEVICE)

    # image=IMUT.image_to_tensors(opt, True)
    # image = image.to(DEVICE)

    pretrained_model = './pretrained_model/model_{}.pth'.format(ARCH_NUM)
    arch = torch.load(pretrained_model)
    arch = arch.to(DEVICE)
    # arch.eval()
    # if IMAGE_NUM == MAX-1:
    #     print(arch)
    #     print(torchinfo.summary(arch, input_size = (1, 3, 32, 32)))
    target_layers = [arch.module.lastact[0]]
    # target_layers = [arch.module.cells[16].layers[4].op[2]]
    output = arch(input_tensor)
    pred = output[1].max(1)[1].item()
    Softmax = torch.nn.Softmax(dim = 1)
    softmax_output = Softmax(output[1])
    softmax_output_pred = softmax_output[0][pred]
    softmax_output_TARGET_CLASS = softmax_output[0][TARGET_CLASS]
    targets = [ClassifierOutputTarget(TARGET_CLASS)]

    if explanation_method == 0:
        saliency_map = GradCAM_extracor(input_tensor, arch, target_layers, targets)
        directory = "GradCAM"
    
    elif explanation_method == 1:
        saliency_map = EigenGradCAM_extracor(input_tensor, arch, target_layers, targets)
        directory = "EigenGradCAM"

    elif explanation_method == 2:
        saliency_map = ScoreCAM_extracor_2(input_tensor, arch, target_layers, targets)
        directory =  "ScoreCAM"

    elif explanation_method == 3:
        saliency_map = GradCAMPlusPlus_extracor(input_tensor, arch, target_layers, targets)
        directory =  "GradCAMPlusPlus"
        
    new_path = "./arch_{}/{}".format(ARCH_NUM, directory)
    if os.path.isdir(new_path) == False:
        os.mkdir(new_path)
    # saliency_map = ScoreCAM_extracor(image, arch)
    # saliency_map = ScoreCAM_extracor(image, arch, TARGET_CLASS)

    numpy_saliency_map = saliency_map.to("cpu").detach().numpy().copy()
    numpy_saliency_map = numpy_saliency_map[0][0]
    
    cam_imge = show_cam_on_image(img, numpy_saliency_map, use_rgb=True)

    saliency_map = saliency_map.to(DEVICE)
    explanation_map=input_tensor*saliency_map    

    #start
    # numpy_input_tensor = input_tensor.to("cpu").detach().numpy().copy()
    # numpy_input_tensor = numpy_input_tensor[0]
    # numpy_input_tensor = numpy_input_tensor.transpose(1 ,2, 0)
    # numpy_input_tensor = numpy_normalization(numpy_input_tensor)
    # im = Image.fromarray((numpy_input_tensor*255).astype(np.uint8))
    # im.save("./arch_{}/input_{}.png".format(ARCH_NUM, IMAGE_NUM))
    # im = Image.open("./arch_{}/input_{}.png".format(ARCH_NUM, IMAGE_NUM))
    # im = im.resize((224, 224))
    # im.save("./arch_{}/input_{}.png".format(ARCH_NUM, IMAGE_NUM))
    #over
    
    if IMAGE_NUM < VISUALIZE:
        numpy_explanation_map = explanation_map.to("cpu").detach().numpy().copy()
        numpy_explanation_map = numpy_explanation_map[0]
        numpy_explanation_map = numpy_explanation_map.transpose(1 ,2, 0)
        numpy_explanation_map = numpy_normalization(numpy_explanation_map)
        im = Image.fromarray((numpy_explanation_map*255).astype(np.uint8))
        im.save("./arch_{}/{}/explanation_map_{}.png".format(ARCH_NUM, directory, IMAGE_NUM))
        im = Image.open("./arch_{}/{}/explanation_map_{}.png".format(ARCH_NUM, directory, IMAGE_NUM))
        im = im.resize((224, 224))
        im.save("./arch_{}/{}/explanation_map_{}.png".format(ARCH_NUM, directory, IMAGE_NUM))
        

        out_numpy_saliency_map = cv2.merge([numpy_saliency_map, numpy_saliency_map, numpy_saliency_map])
        images = np.hstack((np.uint8(255*img), (numpy_explanation_map*255).astype(np.uint8), (out_numpy_saliency_map * 255).astype(np.uint8) , cam_imge.astype(np.uint8)))
        im = Image.fromarray(images)
        im.save("./arch_{}/{}/cifar_{}_grad.png".format(ARCH_NUM, directory, IMAGE_NUM))
        im = Image.open("./arch_{}/{}/cifar_{}_grad.png".format(ARCH_NUM, directory, IMAGE_NUM))
        im = im.resize((896, 224))
        im.save("./arch_{}/{}/cifar_{}_grad.png".format(ARCH_NUM, directory, IMAGE_NUM))
    
    adcc, avgdrop, coh, com, A, B, conf_in, conf_exp = ADCC.ADCC(input_tensor, saliency_map, explanation_map, arch, targets, target_layers, attr_method=GradCAM_extracor, target_class_idx=TARGET_CLASS, debug=True, auto = AUTO)
    return adcc, avgdrop, coh, com, A, B, TARGET_CLASS, pred, softmax_output_pred, softmax_output_TARGET_CLASS, conf_in, conf_exp, directory

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default='example/image.png')
    parser.add_argument("--model", type=str, default='resnet18')

    opt = parser.parse_args()

    assert OSPATH.exists(opt.image), "Image not found"
    adcc_sum = 0
    comlexity_sum = 0
    average_drop_sum = 0
    coherency_sum = 0
    adcc_sum_correct = 0
    comlexity_sum_correct = 0
    average_drop_sum_correct = 0
    coherency_sum_correct = 0
    correct_count = 0

    for i in range(0, MAX):
        print("-----------------------------{}-----------------------------".format(i))
        adcc, avgdrop, coh, com, A, B, correct, pred, out_pred, out_correct, confidence_in, confidence_exp, directory = main(i)
        path = "./arch_{}/{}/cifar_{}_grad.png".format(ARCH_NUM, directory, i)
        adcc_sum += adcc
        comlexity_sum += com
        average_drop_sum += avgdrop
        coherency_sum += coh
        if correct == pred:
            adcc_sum_correct += adcc
            comlexity_sum_correct += com
            average_drop_sum_correct += avgdrop
            coherency_sum_correct += coh
            correct_count += 1
        print("adcc : {}".format(adcc))
        print("100 * ascc : {}".format(adcc * 100))
        print('finish')
        print("\n")
        if i < VISUALIZE:
            visualize_score(path, correct, pred, adcc, avgdrop, coh, com.item(), out_pred, out_correct, confidence_in, confidence_exp)
    average_adcc = (adcc_sum / MAX) * 100
    average_comlexity = (comlexity_sum / MAX) * 100
    average_drop = (average_drop_sum / MAX) * 100
    average_coherency = (coherency_sum / MAX) *100

    average_adcc_correct = (adcc_sum_correct / correct_count) * 100
    average_comlexity_correct = (comlexity_sum_correct / correct_count) * 100
    average_drop_correct = (average_drop_sum_correct / correct_count) * 100
    average_coherency_correct = (coherency_sum_correct / correct_count) *100
    
    print("final average of comlexity : {}".format(average_comlexity))
    print("final average of average drop: {}".format(average_drop))
    print("final average of coherency: {}".format(average_coherency))
    print("final average of adcc : {}".format(average_adcc))
    print("\n")

    print("final average of comlexity coorect : {}".format(average_comlexity_correct))
    print("final average of average drop coorect : {}".format(average_drop_correct))
    print("final average of coherency coorect : {}".format(average_coherency_correct))
    print("final average of adcc coorect : {}".format(average_adcc_correct))
    
    write_directory = "./arch_{}/{}/Score.txt".format(ARCH_NUM, directory)

    accuracy_arch = [value for key, value in accuracy_list.items() if key == ARCH_NUM]

    f = open(write_directory, "w")
    f.write("acurracy = {}\n".format(accuracy_arch))
    f.write("\n")
    f.write("final average of comlexity : {}\n".format(average_comlexity))
    f.write("final average of average drop: {}\n".format(average_drop))
    f.write("final average of coherency: {}\n".format(average_coherency))
    f.write("final average of adcc : {}\n".format(average_adcc))
    f.write("\n")

    f.write("final average of comlexity coorect : {}\n".format(average_comlexity_correct))
    f.write("final average of average drop coorect : {}\n".format(average_drop_correct))
    f.write("final average of coherency coorect : {}\n".format(average_coherency_correct))
    f.write("final average of adcc coorect : {}\n".format(average_adcc_correct))
    f.close()