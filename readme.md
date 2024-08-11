# Fact-driven Storytelling with LLMs 🌐

## Overview 📖
This repository contains the implementation of a system that leverages large language models (LLMs) for generating argument pyramids, focusing on the relevance of claims with arguments, support of arguments with evidences, the overall logical coherence, and the completness with all arguments.
![Process Overview](./pipeline.png)

## Model Weights 📦 
The model weights are hosted on Google Drive due to their size. Access them [here](https://drive.google.com/drive/folders/1UPbiBLuExIKfrYGkbWyj4pBYlNavgjLO?usp=sharing).

### BERT Model(relevance) 📕
- **File**: `model.safetensors`
- **Location**: Download and place it under `models/relevance_model`.
- **Purpose**: Calculates the relevance score between claims and arguments.
- **Source**: Fine-tuned from the model available at [ibm/argument_quality_ranking_30k](https://huggingface.co/datasets/ibm/argument_quality_ranking_30k). This fine-tuning was specifically performed using the MACE-P labeling to enhance the model's capability in evaluating the quality of arguments in the context of claims.

### RoBERTa Model(support) 📗
- **File**: `best_model.pt`
- **Location**: Download and place it under `models/support_model`.
- **Training**: Preprocess the data from "Evidence Convincingness" dataset http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml to create pos_neg_pairs_train.csv, details in support_model_training/data_preparation. Then apply contrastive learning based on the preprocessed data to fine-tune the RoBERTa-based Model.
- **Purpose**: Used to assess the support score between arguments and evidences.

### ALBERT Model(coherence) 📘
- **File**: `pytorch_model.bin`
- **Location**: Download and place it under `models/coherence_model`.
- **Source**: From CoUDA project: "Coherence Evaluation via Unified Data Augmentation"(NAACL 2024) (https://aclanthology.org/2024.naacl-long.55).
- **Purpose**: Evaluates the overall logical coherence score of the pyramid.

### DeBERTa Model(completeness) 📙
- **File**: `model_4.safetensors`
- **Location**: Download and place it under `models/completeness_model`.
- **Dataset and training**: use dataset from https://aclanthology.org/E17-1092/ to fine-tune a pre-trained DeBERTa model on classification task.
- **Purpose**: compute a completeness score of the pyramid, indicating whether the claim has sufficient supporting arguments.
  
## Installation 🛠️
To set up the project environment, execute the following command:

```bash
pip install -r requirements.txt
```
## Setup and Configuration ⚙️
Configure the necessary API keys before use:
1. Navigate to the `config` folder.
2. Enter your OpenAI and Bing API keys for content generation and data fetching.

## Usage 🚀
To generate argument pyramids:
1. Modify the `question` in `main.py` to set your query.
2. Set `num_pyramids` to define how many successful pyramids to generate.
3. Run `main.py`. The system will generate multiple pyramids and select the one with the highest score.

## Note on API Access ✅
If you don't have access to Bing or OpenAI APIs, don't worry! You can still follow along with the demonstration of how the final results are generated. Simply check the end of the `Demo.ipynb` notebook where we've provided a detailed walkthrough of the process, so you can see exactly how the output is produced even without API keys.
   
## Cost and Performance 💰
- Generation uses GPT-4, costing approximately $2.00 and taking about 6 minutes on a GPU-T4.
- Using GPT-3.5 turbo can result in unpredictable quality and may not conform to detailed prompt structures necessary for improvements.

## Citation 🌟

This project was developed at the **Technical University of Munich** under the **Research Group Social Computing of TUM**, in **July 2024**.

| Role        | Names                              |
|-------------|------------------------------------|
| **Authors** | Jiaqi Mo, Ying Hua, Xinyan Guo     |
| **Guidance**| Miriam Anschütz, Simon Malberg     |

![badge](https://img.shields.io/badge/University-TUM-blue)
![badge](https://img.shields.io/badge/Year-2024-red)


## Note ❗
Regular expressions extract claims, arguments, and evidences from GPT-generated content. Format inconsistencies might cause bugs, sometimes requiring a restart of the generation process.

For detailed methodology and computational requirements, see `Poster.pdf` in this repository.
