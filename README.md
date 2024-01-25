# Relationship between AutoML and Explainability

## Note
&nbsp; First, this code is not yet complete. Second, I am not good at English, so it may be difficult to read. I apologize in advance.

## Abstract

&nbsp;In recent years, Auto ML, which can be used without requiring specialized knowledge, has attracted much attention. However, with the development of this method, it has become a black box, and its reliability has been questioned. This study aims to clarify the relationship between Auto ML and explainability by using explainable AI that provides the basis for decisions.  
&nbsp; This code is This code is used to verify the explainability of models created with Auto ML.
Note that this code contains quotes from [ADCC], [AutoDL-Projects] and [pytorch-grad-cam].

## Outline step

1. Create a 10-classification model of cifar-10 using the NAS method provided in [AutoDL-Projects].
2. Save the created model information in the directory "pretrained_model" (e.g. model_12384.pth).
3. Create a class activation map(CAM) for the saved model using the cifar10 dataset.
4. The degree to which the created CAM satisfies the explanatory potential is expressed using the ADCC metrics introduced in [ADCC].
5. Finally, the generated CAM and metrics are saved in your directory (e.g. arch_12384).

## Finaly
If you have any questions, please email "f23c001b@mail.cc.niigata-u.ac.jp".

[ADCC]: https://github.com/aimagelab/ADCC "ADCC"
[AutoDL-Projects]: https://github.com/D-X-Y/AutoDL-Projects "AutoDL-Projects"
[pytorch-grad-cam]: https://github.com/jacobgil/pytorch-grad-cam "pytorch-grad-cam"
