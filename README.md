# Vision Transformer with Tensorflow2
This is an unofficial implementation of [An Image is worth 16x16 words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929v2.pdf) based on Tensorflow2. <br>

## Introduce
While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks(ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train

## Why it works?

## TODO
- [ ] basic input process `tf.data`
- [ ] basic train process `tf.function`
- [ ] basic inference process


## Usage

### train
```python
python train.py
```

### demo
```python
python demo_image.py
```

## References
- [tensorflow/models/vit](https://github.com/tensorflow/models/blob/master/official/projects/vit/modeling/vit.py)
- [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)
