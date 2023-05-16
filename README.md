# CIF-MMIN

This repo implements the CIF Aware Missing Modality Imagination Network (CIF-MMIN)  for the following paper:
"Contrastive Learning based Modality-Invariant Feature Acquisition for Robust Multimodal Emotion Recognition with Missing Modalities" 

# Environment

``` 
python 3.8.0
pytorch >= 1.8.0
```

# Usage

First you should change the data folder path in ```data/config``` and preprocess your data follwing the code in ```preprocess/```.

The preprocess of feature was done handcrafted in several steps, we will make it a automatical running script in the next update. You can download the preprocessed feature to run the code.

+ For Training CIF-MMIN on IEMOCAP:

    First training a model self-supervise model with all audio, visual and lexical modality as the pretrained encoder.

    ```bash
    bash scripts/CAP_utt_self_supervise.sh AVL [num_of_expr] [GPU_index]
    ```

    Then

    ```bash
    bash scripts/our/CAP_CIF_MMIN.sh [num_of_expr] [GPU_index]
    ```

+ For Training CIF-MMIN on MSP-improv: 

    ```bash
    bash scripts/MSP_utt_self_supervise.sh AVL [num_of_expr] [GPU_index]
    ```

    ```
    bash scripts/our/MSP_CIF_MMIN.sh [num_of_expr] [GPU_index]
    ```
    
+ For Training CIF-MMIN on MOSI: 

    ```bash
    bash scripts/MOSI_utt_self_supervise.sh AVL [num_of_expr] [GPU_index]
    ```

    ```
    bash scripts/our/MOSI_CIF_MMIN.sh [num_of_expr] [GPU_index]
    ```
    
Note that you can run the code with default hyper-parameters defined in shell scripts, for changing these arguments, please refer to options/get_opt.py and the ```modify_commandline_options``` method of each model you choose.

# Download the features
Baidu Yun Link
IEMOCAP A V L modality Features
链接: https://pan.baidu.com/s/1WmuqNlvcs5XzLKfz5i4iqQ 提取码: gn6w 

# License
MIT license. 

Copyright (c) 2023 S2Lab, School of Inner Mongolia University.

<!-- # Citation
If you find our paper and this code usefull, please consider cite
```
@inproceedings{zhao2021missing,
  title={Missing modality imagination network for emotion recognition with uncertain missing modalities},
  author={Zhao, Jinming and Li, Ruichen and Jin, Qin},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  pages={2608--2618},
  year={2021}
}
``` -->
