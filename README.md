# Heartfelt Words: Automated Report Generation for ECG Signals

This repository contains the implementation of an innovative system for generating clinical reports from 12-lead ECG signals. It combines advanced signal processing techniques with state-of-the-art language models to assist medical professionals in interpreting ECG data.

## Project Overview

Cardiovascular diseases (CVDs) are the leading cause of death worldwide. The 12-lead ECG is a widely used diagnostic tool, but interpreting these signals can be challenging and time-consuming. This project proposes a novel approach using machine learning to generate human-readable ECG reports, aiming to aid clinicians and improve communication with patients.

### Key Features:
- **Encoder-Decoder Architecture**: The model consists of an encoder for extracting key information from ECG signals and a GPT-2-based decoder for generating clinical reports.
- **Multi-label Learning**: Incorporates multi-label training to capture diverse information such as disease type, rhythm, and form of the signal.
- **BLEU-Based Evaluation**: Measures the quality of generated reports using BLEU scores, complemented by qualitative analysis.
- **Multilingual Support**: Preliminary experiments on generating reports in multiple languages.

## Repository Structure

- **`encoder_training/`**: Code for training the encoder models (ResNet, FCN) on ECG classification tasks.
- **`decoder_training/`**: Implementation of the GPT-2-based text generation component.
- **`data_processing/`**: Scripts for preparing datasets, including the PTB-XL dataset and translated captions.
- **`experiments/`**: Jupyter notebooks and scripts for conducting experiments, including testing various encoders and evaluating BLEU scores.
- **`reports/`**: Example generated ECG reports for different encoder-decoder configurations.

## Datasets

The project leverages the **PTB-XL** dataset, a publicly available collection of 21,837 clinical 12-lead ECGs. Each record is annotated with diagnostic, form, and rhythm information, along with textual clinical reports. Translated captions were also utilized to explore multilingual capabilities.

## Key Experiments

1. **Encoder Models**:
   - Tested ResNet and FCN architectures for feature extraction.
   - Found that multi-label training enhances the encoder's ability to represent complex ECG data.

2. **Text Pre-training**:
   - Evaluated the impact of pre-training the GPT-2 decoder on medical texts, which showed limited improvement due to overfitting risks.

3. **Multilingual Captioning**:
   - Explored the use of English and German captions, with limited success due to data constraints.

4. **Evaluation Metrics**:
   - Assessed performance using BLEU-1 and BLEU-4 scores and qualitative analysis for readability.

## Future Work

- Incorporate larger and more diverse datasets to improve model robustness.
- Develop end-to-end training approaches for better integration between encoder and decoder.
- Enhance multilingual support with advanced language models and more balanced datasets.
- Explore interpretability techniques to build trust in ML-generated reports among clinicians and patients.

## References

1. [Let Your Heart Speak in its Mother Tongue: Multilingual Captioning of Cardiac Signals](https://arxiv.org/abs/2103.14626)
2. [GPT-2: Language Models are Unsupervised Multitask Learners](https://openai.com/research/language-unsupervised)
3. [PTB-XL Dataset](https://physionet.org/content/ptb-xl/1.0.1/)

---

This repository showcases a proof-of-concept for generating medical reports from ECG signals and sets a foundation for future research in medical text generation. Contributions and feedback are welcome!
