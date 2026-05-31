# MedBot — Fine-Tuned Medical Q&A Assistant

A medical chatbot built by fine-tuning **Google Gemma-2B** on 3,000 real patient-doctor 
conversations using **QLoRA (4-bit quantization + LoRA adapters)** on free Google Colab T4 GPU.

## Live Demo
[HuggingFace Space →](https://huggingface.co/spaces/Abdullah-1-23/medbot-assistant)

## Model
[Abdullah-1-23/medbot-gemma-2b-qlora →](https://huggingface.co/Abdullah-1-23/medbot-gemma-2b-qlora)

## Architecture
Base Model (Gemma-2B) → 4-bit Quantization (QLoRA) → LoRA Adapters (r=16) → Fine-Tuned MedBot

## Dataset
- Source: ChatDoctor-HealthCareMagic-100k (Hugging Face)
- Training samples: 2,850 | Eval samples: 150
- Format: Instruction-tuned patient-doctor conversation pairs

## Results
| Metric | Value |
|---|---|
| ROUGE-L (50 eval samples) | 0.1422 |
| Trainable parameters | ~4M / 2B (0.2%) |
| Training time (T4 GPU) | ~30 minutes |
| Fine-tuning method | QLoRA (4-bit NF4 + LoRA r=16) |

## Sample Output
**Q: I have had a persistent cough for two weeks with mild fever. What could it be?**

*It could be viral infection or bronchitis. Treatment will depend on severity of symptoms. 
If severe then antibiotics can be given to clear it up in 3-7 days. Take rest and eat warm 
beverages such as ginger tea to improve immunity...*

## Tech Stack
- Model: google/gemma-2b
- Fine-tuning: QLoRA (bitsandbytes + PEFT)
- Training framework: TRL SFTTrainer
- UI: Streamlit
- Deployment: Hugging Face Spaces

## Disclaimer
MedBot is for informational purposes only. Not a substitute for professional medical advice.
