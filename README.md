# What is our project?
We aim to reunite children with their parents after disaster. By equipping every shelter / disaster relief site with a node in the network and a camera, we use kinship verification technology to match children with their mother and father.

# Hypothesis
The hypothesis of this project is that the age-invariant features of children are inherited from parents. By inferring from these features, we may tell whether a child is related to a father-mother pair.

# Architecture
We utilize transfer learning by extracting age-invariant features from the mother, father and child. We learn a comparison function by the use of fully-connected and concatenation layer. Our approach is to compare the father and child, and the mother and child. Then, these comparisons are concatenated, and are inferred from to obtain a final relatedness score (0-1).

![Network Architecture ](https://docs.google.com/drawings/d/e/2PACX-1vQ8U3VpxiEMBZjwiohKaD9AMxnCiTWgx9hjdq3mOVeJMTNZXeq1EE2O1RuOsnaRiq9kpmHpK7mpxr1E/pub?w=960&h=720)

# results

|Date|Dataset| Model| Validation Acc| Validation Loss| Training Acc| starting LR| Epochs| Batch size| Batches per epoch| Notes| Weights filename|
|-|-|-|-|-|-|-|-|-|-|-|-|
|Sep 5 2019|FIW|`model_v3`|`90.26%`|`0.389`|`91.37%`|`0.0001`| `2410`| `50` F-M-C(+)-C(-) pairs | 20| Reduced LR on plateau after a patience of `75` epochs| `src/ml/saved_model_weights/v3_weights.2409-0.39.hdf5`|
|Sep 1 2019|FIW|`model_v3`|`86.57%`|`0.495`|`99.05%`|`0.0001`| `2134`| `50` F-M-C(+)-C(-) pairs | 20| Reduced LR on plateau after a patience of `75` epochs| `src/ml/saved_model_weights/v3_weights.2134-0.50.hdf5`|

Note: to use the model weights, you must point the model saver (see `train_model_v3.py`) to the location of the weights.

# TODO
Things we need to do:
- [x] Implement more loss functions if needed (e.g. crossentropy?)
- [x] Visualize using tensorboard.
- [x] Finish `train_model.py`  `and model.py`
- [x] Implement data loader class to load images from FIW / other dataset
- [x] Implement algorithm to sample (MF-C) pairs used in triplet pairs
- [x] Fix training bug in tensorflow - (multiple forward passes result in weight update)
- [x] Try reducing LR on plateau after 80 epochs

# Papers used:

Look Across Elapse
```
 @article{zhao2018look,
      title={Look Across Elapse: Disentangled Representation Learning and Photorealistic Cross-Age Face Synthesis for Age-Invariant Face Recognition},
      author={Zhao, Jian and Cheng, Yu and Cheng, Yi and Yang, Yang and Lan, Haochong and Zhao, Fang and Xiong, Lin and Xu, Yan and Li, Jianshu and Pranata, Sugiri and others},
      journal={AAAI},
      year={2019}
      }
```
