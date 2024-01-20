# DeepMFPP
Multi-Functional Peptides Prediction Based on Data Augmentation and Contrastive Learning

## Introduction
As the important potential therapeutic drugs, multi-functional peptides (MFPs) identification is a major challenge in the field of medicine. For identifying the different functional characteristics, many multi-label computational methods have been proposed. These methods improved the efficiency of MFPs discovery, and achieved some success. However, these methods ignore the influence of the Long-Tail label distribution on the prediction accuracy. To further improve the accuracy of MFPs prediction, we introduced a novel deep learning approach DeepMFPP, which utilized data augmentation techniques and contrastive learning to solve the Tail-End problem. The 1:1 data augmentation based on random replace and random swap was conducted on the certain Tail-End samples. For the MFPs representations, the semantic information obtained from ESM-2 and the positional information are fused with self-attention-based method. Then with the feature optimization and projection modules, we could acquire the robust representation of MFPs. DeepMFPP outperforms the state-of-the-art methods for MFPs prediction, with the key metrics Accuracy is 0.702, and Absolute True is 0.633. On additional external datasets, DeepMFPP exhibits excellent extensibility (Accuracy: 0.802, and Absolute True: 0.795), potentially unifying the paradigm framework for MFPs prediction task.

## Description of relevant files
|Files name      |Description |
|----------------|------------|
|data            | Raw data and DA data used in this study |
|ESM2            | Protein Pretraining Language Model ESM-2 |
|attention.py    | DeepPD and Other algorithms involved in this study |
|config.py       | DeepMFPP configuration file |
|data_process.py | Data preprocessing and format conversion script |
|evaluation.py   | evaluating indicator |
|LossFunction.py | loss functions |
|main.py         | model training script|
|model.py        | DeepMFPP model |
|modules.py      | the necessary model components for building DeepMFPP |
|utils.py        | Some necessary component units |

## Citation
Not available at the moment.