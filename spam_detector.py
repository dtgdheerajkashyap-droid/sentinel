from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection"
)

text = """
Congratulations!
You have won a free iPhone.
Click here immediately to claim your prize.
"""

result = classifier(text)

print(result)