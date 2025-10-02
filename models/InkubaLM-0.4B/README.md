---
language:
- en
- sw
- zu
- xh
- ha
- yo
pipeline_tag: text-generation
tags:
- nlp
- InkubaLM
- africanLLM
- africa
- llm
datasets:
- lelapa/Inkuba-Mono
license: cc-by-nc-4.0
---
# InkubaLM-0.4B: Small language model for low-resource African Languages

<!-- Provide a quick summary of what the model is/does. -->


![ ](InkubaLM.png) 

## Model Details
InkubaLM has been trained from scratch using 1.9 billion tokens of data for five African languages, along with English and French data, totaling 2.4 billion tokens of data. 
Similar to the model architecture used for MobileLLM, we trained this InkubaLM with a parameter size of 0.4 billion and a vocabulary size of 61788. 
For detailed information on training, benchmarks, and performance, please refer to our full [blog post](https://medium.com/@lelapa_ai/inkubalm-a-small-language-model-for-low-resource-african-languages-dc9793842dec).
### Model Description

<!-- Provide a longer summary of what this model is. -->



- **Developed by:** [Lelapa AI](https://lelapa.ai/) - Fundamental Research Team.
- **Model type:** Small Language Model (SLM) for five African languages built using the architecture design of LLaMA-7B.
- **Language(s) (NLP):** isiZulu, Yoruba, Swahili, isiXhosa, Hausa, English and French.
- **License:** CC BY-NC 4.0.

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** TBD
- **Paper :** [InkubaLM](https://arxiv.org/pdf/2408.17024)


## How to Get Started with the Model

Use the code below to get started with the model.

``` python
pip install transformers

```
# Running the model on CPU/GPU/multi GPU
## - Running the model on CPU
``` Python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("lelapa/InkubaLM-0.4B",trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("lelapa/InkubaLM-0.4B",trust_remote_code=True)

text = "Today I planned to"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs.input_ids 

# Create an attention mask
attention_mask = inputs.attention_mask 

# Generate outputs using the attention mask
outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=60,pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```
## - Using full precision
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("lelapa/InkubaLM-0.4B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("lelapa/InkubaLM-0.4B", trust_remote_code=True)

model.to('cuda')
text = "Today i planned to  "
input_ids = tokenizer(text, return_tensors="pt").to('cuda').input_ids
outputs = model.generate(input_ids, max_length=1000, repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.batch_decode(outputs[:, input_ids.shape[1]:-1])[0].strip())

```
## - Using torch.bfloat16
``` python 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
checkpoint = "lelapa/InkubaLM-0.4B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True)
inputs = tokenizer.encode("Today i planned to ", return_tensors="pt").to("cuda")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))

```
## - Using quantized Versions via bitsandbytes

``` python
pip install bitsandbytes accelerate
```
``` python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True) # to use 4bit use `load_in_4bit=True` instead
checkpoint = "lelapa/InkubaLM-0.4B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=quantization_config, trust_remote_code=True)
inputs = tokenizer.encode("Today i planned to ", return_tensors="pt").to("cuda")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))

```


## Training Details

### Training Data

- For training, we used the [Inkuba-mono](https://huggingface.co/datasets/lelapa/Inkuba-Mono) dataset. 



#### Training Hyperparameters

| Hyperparameter      | Value |
| ----------- | ----------- |
| Total Parameters      | 0.422B       |
| Hidden Size   | 2048        |
| Intermediate Size (MLPs)   | 5632        |
| Number of Attention Heads   | 32        |
| Number of Hidden Layers  | 8        |
| RMSNorm ɛ  | 1e^-5        |
| Max Seq Length   | 2048        |
| Vocab Size | 61788 |

## Limitations
The InkubaLM model has been trained on multilingual datasets but does have some limitations. It is capable of understanding and generating content in five African languages: Swahili, Yoruba, Hausa, isiZulu, and isiXhosa, as well as English and French. While it can generate text on various topics, the resulting content may not always be entirely accurate, logically consistent, or free from biases found in the training data. Additionally, the model may sometimes use different languages when generating text. Nonetheless, this model is intended to be a foundational tool to aid research in African languages.

## Ethical Considerations and Risks
InkubaLM is a small LM developed for five African languages. The model is evaluated only in sentiment analysis, machine translation, AfriMMLU, and AfriXNLI tasks and has yet to cover all possible evaluation scenarios. Similar to other language models, it is impossible to predict all of InkubaLM's potential outputs in advance, and in some cases, the model may produce inaccurate, biased, or objectionable responses. Therefore, before using the model in any application, the users should conduct safety testing and tuning tailored to their intended use.

## Citation

```
@article{tonja2024inkubalm,
  title={InkubaLM: A small language model for low-resource African languages},
  author={Tonja, Atnafu Lambebo and Dossou, Bonaventure FP and Ojo, Jessica and Rajab, Jenalea and Thior, Fadel and Wairagala, Eric Peter and Anuoluwapo, Aremu and Moiloa, Pelonomi and Abbott, Jade and Marivate, Vukosi and others},
  journal={arXiv preprint arXiv:2408.17024},
  year={2024}
}
```

## Model Card Authors

[Lelapa AI](https://lelapa.ai/) - Fundamental Research Team

## Model Card Contact

[Lelapa AI](https://lelapa.ai/)