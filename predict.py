"predict.py - single-text CLI inference for the fine-tuned sentiment model"

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_NAMES = {0: 'negative', 1: 'positive'}


def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def predict(text, tokenizer, model, max_seq_len=256):
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_seq_len,
        return_tensors='pt'
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label_id = int(torch.argmax(probs).item())
    return {
        'text': text,
        'label_id': label_id,
        'label_name': LABEL_NAMES[label_id],
        'confidence': float(probs[label_id].item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment for a single financial sentence.")
    parser.add_argument('text', type=str, help="Sentence to classify.")
    parser.add_argument('--model-dir', type=str, default='artifacts/final_model',
                        help="Directory containing the fine-tuned model and tokenizer.")
    parser.add_argument('--max-seq-len', type=int, default=256)
    cli_args = parser.parse_args()

    tokenizer, model = load_model(cli_args.model_dir)
    result = predict(cli_args.text, tokenizer, model, max_seq_len=cli_args.max_seq_len)

    for key, value in result.items():
        if key == 'confidence':
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == '__main__':
    main()
