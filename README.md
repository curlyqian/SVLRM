# SVLRM
Implementation of CVPR2019 Paper: "Spatially Variant Linear Representation Models for Joint Filterings"(http://openaccess.thecvf.com/content_CVPR_2019/papers/Pan_Spatially_Variant_Linear_Representation_Models_for_Joint_Filtering_CVPR_2019_paper.pdf) in PyTorch

Joint filtering mainly uses an additional guidance image as a prior and transfers its structures to the target image in the filtering process. Different from existing algorithms that rely on locally linear models or hand-designed objective functions to extract the structural information from the guidance image, we propose a new joint filter based on a spatially variant linear representation model (SVLRM), where the target image is linearly represented by the guidance image. However, the SVLRM leads to a highly ill-posed problem. To estimate the linear representation coefficients, we develop an effective algorithm based on a deep convolutional neural network (CNN). The proposed deep CNN (constrained by the SVLRM) is able to estimate the spatially variant linear representation coefficients which are able to model the structural information of both the guidance and input images. We show that the proposed algorithm can be effectively applied to a variety of applications, including depth/RGB image upsampling and restoration, flash/no-flash image deblurring, natural image denoising, scale-aware filtering, etc. Extensive experimental results demonstrate that the proposed algorithm performs favorably against state-of-the-art methods that have been specially designed for each task.  

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

# Implementations Introduction
```
This paper applies the proposed algorithm to several different image restoration tasks. The implementation of these tasks will be introduced in subfolders.
```
