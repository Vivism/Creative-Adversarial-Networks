# Installation

1. If you don't already have [Pipenv](https://pipenv.readthedocs.io/en/latest/) installed, install it.

```bash
brew install pipenv
```

2. Install the project's dependancies.

```bash
pipenv install
pipenv shell
python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.3.0-py3-none-any.whl
```

## Getting the Dataset

We used the [WikiArt](https://www.wikiart.org/)
[Dataset](https://github.com/cs-chan/ICIP2016-PC/tree/f5d6f6b58a6d8a4bd05aaaedd9688d08c02df8f2/WikiArt%20Dataset).
Its usage is subject to WikiArts's [Terms of Use](https://www.wikiart.org/en/terms-of-use)

```bash
bash data/download_wikiart.sh
```

## Getting pretrained models

Models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/12r4dpxW5j1ouQbn51GkCoc-rcZjEZU1u?usp=sharing). They should be downloaded and extracted to the `/models` directory.