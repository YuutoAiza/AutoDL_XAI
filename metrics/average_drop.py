import torch
import torch.nn.functional as FF

def average_drop(image, explanation_map, arch, out, class_idx=None, auto = False):

    if auto == True:
        with torch.no_grad():
            out_on_exp = FF.softmax(arch(explanation_map)[1], dim=1)
    else:
        with torch.no_grad():
            out_on_exp = FF.softmax(arch(explanation_map), dim=1)

    if class_idx is None:
        class_idx = out.max(1)[1].item()
        confidence_on_inp = out.max(1)[0].item()
    
    confidence_on_inp = out[:, class_idx][0].item()
    confidence_on_exp = out_on_exp[:,class_idx][0].item()
    # print("confidence_on_inp : {}".format(confidence_on_inp))
    # print("confidence_on_exp : {}".format(confidence_on_exp))
    # print("\n")

    return max(0.,confidence_on_inp-confidence_on_exp)/confidence_on_inp, confidence_on_inp, confidence_on_exp