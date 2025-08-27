---
library_name: peft
license: llama2
base_model: llava-hf/llava-1.5-7b-hf
tags:
- base_model:adapter:llava-hf/llava-1.5-7b-hf
- lora
pipeline_tag: text-generation
model-index:
- name: llava-1.5-7b-animal-lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llava-1.5-7b-animal-lora

This model is a fine-tuned version of [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) on an unknown dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 4
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 2

### Training results



### Framework versions

- PEFT 0.17.0
- Transformers 4.55.2
- Pytorch 2.8.0+cu128
- Datasets 4.0.0
- Tokenizers 0.21.4