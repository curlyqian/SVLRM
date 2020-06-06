# Depth Upsampling
```
In this part, we introduce the implementation of applying SVLRM to the task of Depth image upsampling.
```

# Requirement
```
torch
torchvision
cv2
PIL
numpy
matplotlib
maths
os
```

# Download dataset
```

```

#Usage
```
usage: main.py [-h] --dataset DATASET --crop_size CROP_SIZE
               --upscale_factor UPSCALE_FACTOR [--batch_size BATCH_SIZE]
               [--test_batch_size TEST_BATCH_SIZE] [--epochs EPOCHS] [--lr LR]
               [--step STEP] [--clip CLIP] [--weight-decay WEIGHT_DECAY]
               [--cuda] [--threads THREADS] [--gpuids GPUIDS [GPUIDS ...]]
               [--test] [--model PATH]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset directory name
  --crop_size CROP_SIZE
                        network input size
  --upscale_factor UPSCALE_FACTOR
                        super resolution upscale factor
  --batch_size BATCH_SIZE
                        training batch size
  --test_batch_size TEST_BATCH_SIZE
                        testing batch size
  --epochs EPOCHS       number of epochs to train for
  --lr LR               Learning Rate. Default=0.001
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs, Default: n=10
  --clip CLIP           Clipping Gradients. Default=0.4
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        Weight decay, Default: 1e-4
  --eps                 eps, Default: 1e-4
  --cuda                use cuda?
  --threads THREADS     number of threads for data loader to use
  --gpuids GPUIDS [GPUIDS ...]
                        GPU ID for using
  --crop                whether to crop?
  --test                test mode
  --model PATH          path to test or resume model
```


