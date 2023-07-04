import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoFeatureExtractor, TrainingArguments, Trainer, AutoModelForAudioClassification

accuracy = evaluate.load("accuracy")
dataset_name = 'marsyas/gtzan'
dataset = load_dataset(dataset_name)['train']
dataset = dataset.train_test_split(test_size=0.2, seed=42)
dataset = dataset['test']

model_name = 'adamkatav/wav2vec2_100k_gtzan_30s_model'
from transformers import AutoFeatureExtractor
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000*30, truncation=True)
    return inputs

encoded_minds = dataset.map(preprocess_function, remove_columns="audio", batched=True)

label_col = 'genre'
encoded_minds = encoded_minds.rename_column(label_col, "label")

model = AutoModelForAudioClassification.from_pretrained(model_name)
training_args = TrainingArguments(
    output_dir="wav2vec2_100k_gtzan_30s_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=1,
    num_train_epochs=1000,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=feature_extractor,
    eval_dataset=encoded_minds,
)

result = trainer.evaluate(eval_dataset=encoded_minds)