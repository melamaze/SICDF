## SICDF

In this repository, code is for our IEEE ICC 2023 paper [Successive Interference Cancellation Based Defense for Trigger Backdoor in Federated Learning](https://github.com/melamaze/SICDF) 
#

## Introduction
This paper proposes an efficient backdoor trigger defense
framework termed SICDF based on Explainable AI and image
processing. Explainable AI is used to generate the feature
importance of the image and infer the location of the trigger.
Image processing reduces the influence of important features,
allowing the model to make predictions from other features
rather than the trigger. SICDF not only defends against trigger
backdoor attacks in different attack scenarios but also not
affecting the accuracy when the model or the image is not
been attacked.

## Models

| file                  | model        | dataset  | attack ratio | 
|-----------------------|--------------|----------|--------------|
| cifar_densenet_03.pth | densenet121  | Cifar-10 | 0.3          |
| cifar_regnet_03.pth   | regnetY400MF | Cifar-10 | 0.3          |
| cifar_resnet_01.pth   | resnet18     | Cifar-10 | 0.1          |
| cifar_resnet_02.pth   | resnet18     | Cifar-10 | 0.2          |
| cifar_resnet_03.pth   | resnet18     | Cifar-10 | 0.3          |
| clean_cifar.pth       | resnet18     | Cifar-10 | 0.0          |
| clean_densenet.pth    | densenet121  | Cifar-10 | 0.0          |
| clean_gtsrb.pth       | resnet18     | GTSRB    | 0.0          |
| clean_mnist.pth       | resnet18     | MNIST    | 0.0          |
| clean_regnet.pth      | regnetY400MF | Cifar-10 | 0.0          |
| gtsrb_resnet_03.pth   | resnet18     | GTSRB    | 0.3          |
| mnist_resnet_03.pth   | resnet18     | MNIST    | 0.3          |

## How to get the code
```
git clone https://github.com/melamaze/SICDF.git
```

## Model Training 
Training environment and code is placed in `FL_backdoor_model_training` folder.

| --  ``CIFAR-10``

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | -- ``DesnseNet121``

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | -- ``RegNetY400MF``

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | -- ``ResNet18``

| -- ``GTSRB (ResNet18)``

| -- ``MNIST (ResNet18)``

| -- ``requirements.txt (python package requirement)``


## SICDF

Complete version is in this [link](https://github.com/melamaze/pytorch-grad-cam/tree/FINAL_CAM), or you can use this command to get the code: 
```
git clone https://github.com/melamaze/pytorch-grad-cam.git -b FINAL_CAM
```

## Experiment Result

### MNIST
![image](https://i.imgur.com/aSNMNRF.png)

### Cifar-10
![image](https://imgur.com/PWNXC7v.png)

### GTSRB
![image](https://imgur.com/iI32r2E.png)

#
- The first picture in every row is the original image.
- The second picture in every row is the image which is embedded trigger.
- The third picture in every row is heatmap of image which is embedded trigger. We can see the upper right corner is the reddest, which means that the model focuses this position.
- The forth picture in every row is the image after going through our framework.
- The fifth picture in every row is heatmap of the image after going through our framework. We can see the upper right corner is no longer the reddest, which means that the model no longer focuses this position. That is, we successfully decrease model's attention on the trigger.

































