import math
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    BertTokenizerFast, AutoModelForMaskedLM,
    Trainer, TrainingArguments)


def tokenize_func(row):
    return tokenizer(row['title'], padding=True, truncation=True)


model = AutoModelForMaskedLM.from_pretrained(
    "indobenchmark/indobert-lite-base-p2")
tokenizer = BertTokenizerFast.from_pretrained(
    "indobenchmark/indobert-lite-base-p2")
tokenizer.model_max_length = 128

# Split the files in advance if you want to train with a validation set
mlm_datasets = load_dataset(
    'csv',
    data_files={'train': '../data/raw/train_mlm.csv',
                'val': '../data/raw/val_mlm.csv'}
)
tokenized_datasets = mlm_datasets.map(
    tokenize_func,
    batched=True,
    num_proc=8,
    remove_columns=['title'])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15)

training_args = TrainingArguments(
    "../model_v2",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    logging_steps=100,
    warmup_steps=300,
    num_train_epochs=60,
    logging_dir='../log',
    load_best_model_at_end=True,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    data_collator=data_collator
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
