# DINOv3 ConvNext distillation

The main branch of this repository is empty, while individual branches represents experiments and ablations. 

# Branch: feature_scaling
As the base implementation of distillation uses libraries `lightly_train` and `pytorch-image-models`, there is more the 200k source lines of python code which needs to work in order to run the experiments. Modifying the code as recommended proved difficult; writing a class based on the specified interface is insufficient, since the class name is checked at runtime.

The solution is to modify the class instance directly, setting the `forward_features` method to a new function wrapped in `MethodType`. See [https://github.com/GustavMH/DINOv3-ConvNext/blob/feature_scaling/satdistill/distill.py](https://github.com/GustavMH/DINOv3-ConvNext/blob/feature_scaling/satdistill/distill.py)
