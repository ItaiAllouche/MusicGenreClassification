from datasets import load_dataset
from transformers import AutoFeatureExtractor
import evaluate
import numpy as np
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer


dataset_name = 'marsyas/gtzan'
dataset = load_dataset(dataset_name)['train']
dataset = dataset.train_test_split(test_size=0.2, seed=42)
dataset_test = dataset['test']
dataset = dataset['train']
dataset = dataset.train_test_split(test_size=0.25, seed=42)

model_name = 'facebook/wav2vec2-base-100k-voxpopuli'
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000*30, truncation=True)
    return inputs

labels = dataset["train"].features['genre'].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

encoded_minds = dataset.map(preprocess_function, remove_columns="audio", batched=True)
label_col = 'genre'
encoded_minds = encoded_minds.rename_column(label_col, "label")

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    model_name, num_labels=num_labels, label2id=label2id, id2label=id2label
)

training_args = TrainingArguments(
    output_dir="wav2vec2_100k_gtzan_30s_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    num_train_epochs=100,
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
    train_dataset=encoded_minds["train"],
    eval_dataset=encoded_minds["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)
trainer.train()
dataset_test = dataset_test.map(preprocess_function, remove_columns="audio", batched=True)
label_col = 'genre'
dataset_test = dataset_test.rename_column(label_col, "label")
result = trainer.evaluate(eval_dataset=dataset_test)
print(result)
