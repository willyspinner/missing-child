# What is our project?
We aim to reunite children with their parents after disaster. By equipping every shelter / disaster relief site with a node in the network and a camera, we use kinship verification technology to match children with their mother and father.

# TODO
Things we need to do:
- [x] Implement more loss functions if needed (e.g. crossentropy?)
- [x] Visualize using tensorboard.
- [x]  Finish train_model.py  and model.py
- [x] Implement data loader class to load images from FIW / other dataset
- [x] Implement algorithm to sample (MF-C) pairs used in triplet pairs
- [x] Fix training bug in tensorflow - (multiple forward passes result in weight update)

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
