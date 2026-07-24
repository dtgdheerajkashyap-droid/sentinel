from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection"
)

print(classifier.model.config.id2label)