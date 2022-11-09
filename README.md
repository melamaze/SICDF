## SICDF

## How to get the code
```
git clone https://github.com/melamaze/SICDF.git
```

## Model Training 
Training environments and codes are placed in `FL_backdoor_model_training` folder.

| --  ``CIFAR-10``

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | -- ``DesnseNet121``

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | -- ``RegNetY400MF``

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | -- ``ResNet18``

| -- ``GTSRB (ResNet18)``

| -- ``MNIST (ResNet18)``

| -- ``requirements.txt (python package requirement)``


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


## SICDF

Complete version is in this [link](https://github.com/melamaze/pytorch-grad-cam/tree/FINAL_CAM), and you can use this command to get the code: 
```
git clone https://github.com/melamaze/pytorch-grad-cam.git -b FINAL_CAM
```


































