# DINOv3 ConvNext distillation

The main branch of this repository is empty, while individual branches represents experiments and ablations. 

# Branch: feature_scaling
As the base implementation of distillation uses libraries `lightly_train` and `pytorch-image-models`, there is more the 200k source lines of python code which needs to work in order to run the experiments. Modifying the code as recommended proved difficult; writing a class based on the specified interface is insufficient, since the class name is checked at runtime.

The solution is to modify the class instance directly, setting the `forward_features` method to a new function wrapped in `MethodType`. See [https://github.com/GustavMH/DINOv3-ConvNext/blob/feature_scaling/satdistill/distill.py](https://github.com/GustavMH/DINOv3-ConvNext/blob/feature_scaling/satdistill/distill.py)

# Branch: training-length-ablation

This branch trains ConvNeXt-Tiny students with web and satellite DINOv3 teachers for 100 epochs. It evaluates checkpoints at epochs 5, 10, 20, 40, 60, 80 and 100 on Maryland test, New York test and LoveDA validation using three probe seeds.

`experiment.py` trains the students and exports checkpoints. `probe_all_checkpoints.py` evaluates them using the baseline segmentation probe.

See the (https://github.com/GustavMH/DINOv3-ConvNext/tree/training-length-ablation) for instructions.
