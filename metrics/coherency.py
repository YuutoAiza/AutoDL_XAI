from scipy import stats as STS
import torch
import numpy as np

def coherency(saliency_map, explanation_map, target_layers, arch, attr_method, out, target_class = None):
    if torch.cuda.is_available():
        explanation_map = explanation_map.cuda()
        arch = arch.cuda()

    # if target_class is None:
    #     target_class = out.max(1)[1].item()
    saliency_map_B=attr_method(image=explanation_map, model=arch, target_layers = target_layers, classidx=target_class)

    A, B = saliency_map.detach(), saliency_map_B.detach()

    '''
    # Pearson correlation coefficient
    # '''
    Asq, Bsq = A.view(1, -1).squeeze(0).cpu(), B.view(1, -1).squeeze(0).cpu()
    # print("Asq : {}".format(Asq))
    # print("Asq var : {}".format(torch.var(Asq)))
    # print("torch.tensor(Asq).isnan().any() : {}".format(torch.tensor(Asq).isnan().any()))
    # print("Bsq : {}".format(Bsq))
    # print("Bsq var : {}".format(torch.var(Bsq)))
    # print("\n")

    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    if torch.tensor(Asq).isnan().any() or torch.tensor(Bsq).isnan().any() or torch.var(Asq) == 0.0 or torch.var(Bsq) == 0.0:
        y = 0.
    else:
        y, _ = STS.pearsonr(Asq, Bsq) #Calculate Pearson Correlation Coefficient.
        y = (y + 1) / 2 #Since the Pearson Correlation Coefficient ranges between -1 and 1, we normalize the Coherency score between 0 and 1.

    return y,A,B

def coherency_2(saliency_map, explanation_map, arch, attr_method, out):
    if torch.cuda.is_available():
        explanation_map = explanation_map.cuda()
        arch = arch.cuda()

    class_idx = out.max(1)[1].item()
    saliency_map_B=attr_method(image=explanation_map, model=arch, classidx=class_idx)

    A, B = saliency_map.detach(), saliency_map_B.detach()

    '''
    # Pearson correlation coefficient
    # '''
    Asq, Bsq = A.view(1, -1).squeeze(0).cpu(), B.view(1, -1).squeeze(0).cpu()

    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    if torch.tensor(Asq).isnan().any() or torch.tensor(Bsq).isnan().any():
        y = 0.
    else:
        y, _ = STS.pearsonr(Asq, Bsq)
        y = (y + 1) / 2
    
    return y,A,B