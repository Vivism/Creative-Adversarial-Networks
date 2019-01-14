# Creative Adversarial Networks

![collage](assets/256_external_collage.png)

_256x256 samples directly from CAN (no cherry picking) with fixed classification network trained on WikiArt_

An implementation of [CAN: Creative Adversarial Networks, Generating "Art"
by Learning About Styles and Deviating from Style Norms](https://arxiv.org/abs/1706.07068) with a variation that improves sample variance and quality significantly.

Repo based on [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow).

<!-- with modifications to reduce checkerboard artifacts according to [this -->
<!-- distill article](https://distill.pub/2016/deconv-checkerboard/) -->

## Setup

### Installation

1. If you don't already have [Pipenv](https://pipenv.readthedocs.io/en/latest/) installed, install it.

```bash
brew install pipenv
```

2. Install the project's dependancies.

```bash
pipenv install
```

### Getting the Dataset

We used the [WikiArt](https://www.wikiart.org/)
[Dataset](https://github.com/cs-chan/ICIP2016-PC/tree/f5d6f6b58a6d8a4bd05aaaedd9688d08c02df8f2/WikiArt%20Dataset).
Its usage is subject to WikiArts's [Terms of Use](https://www.wikiart.org/en/terms-of-use)

```bash
bash data/download_wikiart.sh
```

### Getting pretrained models

Models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/12r4dpxW5j1ouQbn51GkCoc-rcZjEZU1u?usp=sharing).

## Training a CAN model from scratch (architecture used in the paper)

```bash
# must run from the root directory of the project
bash experiments/train_can_paper.sh
```

## Evaluating an existing CAN model

```bash
# make sure that load_dir acts correctly
bash experiments/eval_can_paper.sh
```

# External Style Classification network

We ran an experiment where we trained an inception resnet to classify style (60% accuracy)
and then used this for the style classification loss, removing the need to learn the layers
in the discriminator. We hold the style classification network constant, so the style distribution
doesn't change as the generator improves. We found that this improved the quality and diversity
of our samples.

## Training CAN with External Style Network

```bash
# make sure that `style_net_checkpoint` is set correctly, or you will error out
bash experiment/train_can_external_style.sh
```

## Training the (ImageNet pre-trained) Inception Resnet

Everything you need should be included in the script. The gist is that it converts the wikiart images into tf records
trains the last layer of the model on these images, then fine-tunes the entire model for 100 epochs, at the end of which
you should get roughly 60% validation accuracy. Since we're looking to generate artwork, this gives us a
level of accuracy that is sufficient to try and generate new artwork.

```bash
cd slim/
vim finetune_inception_resnet_v2_on_wikiart.sh # edit INPUT_DATASET_DIR to match the location of where you downloaded wikiart
bash finetune_inception_resnet_v2_on_wikiart.sh
```

## Evaluating CAN with External Style Network

```bash
# make sure that `style_net_checkpoint` and `load_dir` point to the downloaded models.
bash eval_can_external_style.sh
```

## Experiments

We have run a variety of experiments, all of which are available in the `experiments/` directory.

## Authors

[Phillip Kravtsov](https://github.com/phillip-kravtsov)

[Phillip Kuznetsov](https://github.com/philkuz)

## Citation

If you use this implementation in your own work please cite the following

```
@misc{2017cans,
  author = {Phillip Kravtsov and Phillip Kuznetsov},
  title = {Creative Adversarial Networks},
  year = {2017},
  howpublished = {\url{https://github.com/mlberkeley/Creative-Adversarial-Networks}},
  note = {commit xxxxxxx}
}
```
