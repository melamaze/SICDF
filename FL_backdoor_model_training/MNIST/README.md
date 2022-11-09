### Model
- ``ResNet18``

### Dataset
- ``MNIST`` 

### How to run the code :

```
# run the code
python3 start.py
# run the code under nohup, and redirect ouput in to log
nohup python3 start.py > log &
```

### How to change training setting

``package`` >> ``config`` >> ``for_FL.py``

you can ***change setting*** (i.e., attack ratio) in **for_FL.py**