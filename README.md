# Friends-MMC
[![arXiv](https://img.shields.io/badge/arXiv-2412.17295-b31b1b.svg)](https://arxiv.org/abs/2412.17295)
[![Static Badge](https://img.shields.io/badge/🤗Dataset-Friends_MMC-yellow)](https://huggingface.co/datasets/wangyueqian/friends_mmc/tree/main)
[![Static Badge](https://img.shields.io/badge/🤗Model-Friends_MMC_Speaker_Identification-yellow)](https://huggingface.co/wangyueqian/friends_mmc-speaker_identification)

This repository is the official implementation of paper "Friends-MMC: Dataset for Multi-modal Multi-party Conversation Understanding" (AAAI 2025),
which contains the dataset and code for the conversation speaker identification model.

## Dataset
Download Friends-MMC dataset from [🤗wangyueqian/friends_mmc](https://huggingface.co/datasets/wangyueqian/friends_mmc), and rename the base folder of the downloaded dataset as `datasets`.

## Conversation Speaker Identification
For code related to the conversation speaker identification task, please refer to [csi/README.md](csi/README.md).

## Citation
If you find this work useful, please consider citing:
```bibtex
@misc{wang2024friendsmmcdatasetmultimodalmultiparty,
      title={Friends-MMC: A Dataset for Multi-modal Multi-party Conversation Understanding}, 
      author={Yueqian Wang and Xiaojun Meng and Yuxuan Wang and Jianxin Liang and Qun Liu and Dongyan Zhao},
      year={2024},
      eprint={2412.17295},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.17295}, 
}
```