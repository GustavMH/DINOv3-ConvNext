# DINOv3 ConvNext distillation

The main branch of this repository is empty, while individual branches represents experiments and ablations. 

# Branch: baseline
This branch provides the baseline pipeline on which the other branches are built. It covers the preprocessing of the Chesapeake Land Cover and LoveDA datasets, the distillation of DINOv3 ViT-L/16 teachers into a specified ConvNeXt student using LightlyTrain, and the evaluation of the resulting models with frozen linear segmentation probes. Settings such as the teacher, the distillation method, the number of epochs and the batch size can be adjusted.

The baseline configuration of the study distils the web-pretrained and satellite-pretrained ViT-L/16 teachers into the released DINOv3 ConvNeXt-Tiny, using the DistillationV3 method with LightlyTrain's default augmentation and optimisation settings, a batch size of 64 and 10 epochs on 96,400 Chesapeake crops. Linear probes are trained for 30 epochs and evaluated on the Maryland and New York test sets and the LoveDA validation set.
See [baseline branch](https://github.com/GustavMH/DINOv3-ConvNext/tree/baseline).

# Branch: feature_scaling
As the base implementation of distillation uses libraries `lightly_train` and `pytorch-image-models`, there is more the 200k source lines of python code which needs to work in order to run the experiments. Modifying the code as recommended proved difficult; writing a class based on the specified interface is insufficient, since the class name is checked at runtime.

The solution is to modify the class instance directly, setting the `forward_features` method to a new function wrapped in `MethodType`. See [https://github.com/GustavMH/DINOv3-ConvNext/blob/feature_scaling/satdistill/distill.py](https://github.com/GustavMH/DINOv3-ConvNext/blob/feature_scaling/satdistill/distill.py)

# Branch: training-length-ablation

This branch trains ConvNeXt-Tiny students with web and satellite DINOv3 teachers for 100 epochs. It evaluates checkpoints at epochs 5, 10, 20, 40, 60, 80 and 100 on Maryland test, New York test and LoveDA validation using three probe seeds.

`experiment.py` trains the students and exports checkpoints. `probe_all_checkpoints.py` evaluates them using the baseline segmentation probe.

See the (https://github.com/GustavMH/DINOv3-ConvNext/tree/training-length-ablation) for instructions.
