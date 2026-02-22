# LLM-Based-Spam-vs-Ham-SMS-Classifier
Fine-tunes a pretrained LLM on the SMS Spam Collection dataset to classify messages as spam or ham and output spam/ham probabilities.

**Author:** Azam (Meetra) Nouri

This repository builds a **spam vs. non-spam (ham) SMS classifier** by fine-tuning a pretrained **LLM (GPT-2)** on the public **SMS Spam Collection** dataset. SMS messages are tokenized with the model’s tokenizer and used to fine-tune a lightweight classification head on top of the LLM’s pretrained language representations. Training runs on a Colab GPU and the final model outputs **spam/ham probabilities** for new messages.

## Key Features
- Fine-tunes a pretrained LLM for binary SMS spam classification (spam vs ham)
- Uses a real labeled dataset (SMS Spam Collection)
- Produces spam/ham probability scores for inference

## Dataset
- **SMS Spam Collection**
- Loaded via Hugging Face Datasets: `ucirvine/sms_spam`

## Requirements
Recommended (Colab or local):
```bash
pip install torch transformers datasets accelerate evaluate
