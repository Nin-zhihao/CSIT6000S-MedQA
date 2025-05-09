# -*- coding: utf-8 -*-
"""Fine-tune and inference script for MedQA"""
# python MedQA_fine_tune.py --fine_tune
# python MedQA_fine_tune.py --question "What is asthma?"
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from  Mytrainer import UnlikelihoodTrainer
# Constants
MODEL_NAME = "Nin8520/MedQA"
DATASET_NAME = "Nin8520/medquad-alpaca"
MAX_TARGET_LENGTH = 512

def preprocess_function(examples, tokenizer):
    """Preprocess the dataset for training."""
    inputs = tokenizer(
        ["Question: " + question for question in examples["instruction"]],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    answers = tokenizer(
        ["Answer: " + answer for answer in examples["output"]],
        add_special_tokens=False
    )["input_ids"]

    batch_input_ids = []
    batch_attention = []
    batch_labels = []

    for inp_ids, attn_mask, ans_ids in zip(inputs["input_ids"], inputs["attention_mask"], answers):
        for i in range(0, len(ans_ids), MAX_TARGET_LENGTH):
            chunk = ans_ids[i : i + MAX_TARGET_LENGTH]
            if len(chunk) < MAX_TARGET_LENGTH:
                chunk = chunk + [tokenizer.pad_token_id] * (MAX_TARGET_LENGTH - len(chunk))
            else:
                chunk = chunk[:MAX_TARGET_LENGTH]

            batch_input_ids.append(inp_ids)
            batch_attention.append(attn_mask)
            batch_labels.append(chunk)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention,
        "labels": batch_labels
    }

def fine_tune_model():
    """Fine-tune the model on the MedQA dataset."""
    # Load dataset
    dataset = load_dataset(DATASET_NAME, split="train")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.config.use_cache = False

    # Preprocess the dataset
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        weight_decay=0.01,
        save_steps=500,
        logging_dir="./logs",
        logging_steps=100,
    )

    # Initialize the Trainer
    trainer = UnlikelihoodTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        alpha=0.5,
        processing_class=tokenizer,
    )

    # Fine-tune the model
    trainer.train()

    # Save the model locally and push to the hub
    model.save_pretrained("MedQA")
    tokenizer.save_pretrained("MedQA")
    print("Model and tokenizer saved locally to 'MedQA'.")

def load_model():
    """Load the fine-tuned model for inference."""
    tokenizer = AutoTokenizer.from_pretrained("MedQA")
    model = AutoModelForSeq2SeqLM.from_pretrained("MedQA")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def generate_answer(
    question: str,
    tokenizer,
    model,
    device,
    max_input_length: int = 128,
    max_output_length: int = 512,
    num_beams: int = 5,
    length_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    repetition_penalty: float = 1.2
) -> str:
    """Generate an answer for a given question."""
    inputs = tokenizer(
        f"Question: {question}",
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_input_length
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_output_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            penalty_alpha=0.6,
            top_k=50,
            do_sample=True,
            early_stopping=True
        )
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return answer

def main():
    """Main function to handle fine-tuning and inference."""
    parser = argparse.ArgumentParser(description="MedQA Fine-tuning and Inference")
    parser.add_argument("--fine_tune", action="store_true", help="Fine-tune the model on the MedQA dataset")
    parser.add_argument("--question", type=str, help="Question to generate an answer for")
    args = parser.parse_args()

    if args.fine_tune:
        fine_tune_model()
    elif args.question:
        tokenizer, model, device = load_model()
        answer = generate_answer(args.question, tokenizer, model, device)
        print(f"Q: {args.question}")
        print(f"A: {answer}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()