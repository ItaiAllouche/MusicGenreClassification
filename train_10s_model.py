from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import evaluate
import numpy as np


dataset_name = 'marsyas/gtzan'
dataset_train1 = load_dataset(dataset_name)['train'].train_test_split(test_size=0.2, seed=42)['train']
dataset_train2 = load_dataset(dataset_name)['train'].train_test_split(test_size=0.2, seed=42)['train']
dataset_train3 = load_dataset(dataset_name)['train'].train_test_split(test_size=0.2, seed=42)['train']
dataset_test1 = load_dataset(dataset_name)['train'].train_test_split(test_size=0.2, seed=42)['test']
dataset_test2 = load_dataset(dataset_name)['train'].train_test_split(test_size=0.2, seed=42)['test']
dataset_test3 = load_dataset(dataset_name)['train'].train_test_split(test_size=0.2, seed=42)['test']
dataset_train1 = dataset_train1.train_test_split(test_size=0.2, seed=42)
dataset_train2 = dataset_train2.train_test_split(test_size=0.2, seed=42)
dataset_train3 = dataset_train3.train_test_split(test_size=0.2, seed=42)

def slice_first_10s(example):
    sampling_rate = example['audio']['sampling_rate']
    example['audio']['array'] = example['audio']['array'][:10*sampling_rate]
    return example

def slice_middle_10s(example):
    sampling_rate = example['audio']['sampling_rate']
    example['audio']['array'] = example['audio']['array'][10*sampling_rate:20*sampling_rate]
    return example

def slice_last_10s(example):
    sampling_rate = example['audio']['sampling_rate']
    example['audio']['array'] = example['audio']['array'][20*sampling_rate:]
    return example

dataset_train1.map(slice_first_10s)
dataset_train2.map(slice_middle_10s)
dataset_train3.map(slice_last_10s)

dataset_test1.map(slice_first_10s)
dataset_test2.map(slice_middle_10s)
dataset_test3.map(slice_last_10s)

dataset = DatasetDict({ 'train':concatenate_datasets([dataset_train1['train'], dataset_train2['train'], dataset_train3['train']]), 'test':concatenate_datasets([dataset_train1['test'], dataset_train2['test'], dataset_train3['test']])})
dataset_test = concatenate_datasets([dataset_test1, dataset_test2])

model_name = 'facebook/wav2vec2-base-100k-voxpopuli'
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000*10, truncation=True)
    return inputs


dataset = dataset.map(preprocess_function, remove_columns="audio", batched=True)
label_col = 'genre'
dataset = dataset.rename_column(label_col, "label")

labels = dataset["train"].features['label'].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    model_name, num_labels=num_labels, label2id=label2id, id2label=id2label
)

output_model_name = 'wav2vec2_100k_10s_augmentation_gtzan_model'
training_args = TrainingArguments(
    output_dir=output_model_name,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=6,
    num_train_epochs=15,
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
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()
dataset_test = dataset_test.map(preprocess_function, remove_columns="audio", batched=True)
label_col = 'genre'
dataset_test = dataset_test.rename_column(label_col, "label")
cfm_metric = evaluate.load("BucketHeadP65/confusion_matrix")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return cfm_metric.compute(predictions=predictions, references=eval_pred.label_ids)

trainer.compute_metrics = compute_metrics
confusion = trainer.evaluate(eval_dataset=dataset_test)
print(confusion)
