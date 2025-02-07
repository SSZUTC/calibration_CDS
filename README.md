# SOLID - "Calibration of Time-Series Forecasting: Detecting and Adapting Context-Driven Distribution Shift"

We provide the code for the KDD'24 paper - "Calibration of Time-Series Forecasting: Detecting and Adapting Context-Driven Distribution Shift".

## About

Recent years have witnessed the success of introducing deep learning models to time series forecasting. From a data generation perspective, we illustrate that existing models are susceptible to distribution shifts driven by temporal contexts, whether observed or unobserved. Such context-driven distribution shift (CDS) introduces biases in predictions within specific contexts and poses challenges for conventional training paradigms. 

In this paper, we introduce a universal calibration methodology for the detection and adaptation of CDS with a trained model. To this end, we propose a novel CDS detector, termed the "residual-based CDS detector" or "**Reconditionor**", which quantifies the model's vulnerability to CDS by evaluating the mutual information between prediction residuals and their corresponding contexts. A high **Reconditionor** score indicates a severe susceptibility, thereby necessitating model adaptation. In this circumstance, we put forth a straightforward yet potent adapter framework for model calibration, termed the "sample-level contextualized adapter" or "**SOLID**". This framework involves the curation of a contextually similar dataset to the provided test sample and the subsequent fine-tuning of the model's prediction layer with a limited number of steps. 

Our theoretical analysis demonstrates that this adaptation strategy can achieve an optimal bias-variance trade-off. Notably, our proposed **Reconditionor** and **SOLID** are model-agnostic and readily adaptable to a wide range of models. 

Extensive experiments show that SOLID consistently enhances the performance of current forecasting models on real-world datasets, especially on cases with substantial CDS detected by the proposed **Reconditionor**, thus validating the effectiveness of the calibration approach. 

## Setup 

1. `pip install requirements.txt`, run this to install the envrionment.

## How to use?

We use PatchTST and ETTh1 datasets as an example.

1. `sh scripts/PatchTST/train/ETTh1.sh` This is to first train the forecasting models. Here all scripts in scripts/PatchTST/train are simply copied from the original PatchTST repository, but adding an extra `--run_train --run_test`.
2. `sh scripts/PatchTST/detection/ETTh1.sh` This is to obtain the prediction residuals for calculating the **Reconditionor** indicators. Here `--get_data_error --batch_size 1` is used.
3. `python reconditionor/calc_distribution.py` This is to calculate the **Reconditionor** indicators.
4. `sh scripts/PatchTST/adaptation/ETTh1.sh` This is to use **SOLID** to make sample-level adaptations on the forecasting models, thus making better performance. `--test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 10` is used to make adaptation.


## Citation

```bibtex
@inproceedings{2024_calibration,
  title={Calibration of Time-Series Forecasting: Detecting and Adapting Context-Driven Distribution Shift},
  author={Mouxiang Chen and Lefei Shen and Han Fu and Zhuo Li and Jianling Sun and Chenghao Liu}
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2024}
}
```
