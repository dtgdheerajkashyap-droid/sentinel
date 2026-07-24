from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from transformers import pipeline
import base64

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

print("Loading BERT model...")

classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection"
)

print("BERT loaded successfully.\n")

flow = InstalledAppFlow.from_client_secrets_file(
    'config/credentials.json',
    SCOPES
)

creds = flow.run_local_server(port=0)

service = build('gmail', 'v1', credentials=creds)

results = service.users().messages().list(
    userId='me',
    maxResults=5
).execute()

messages = results.get('messages', [])

for message in messages:

    msg = service.users().messages().get(
        userId='me',
        id=message['id'],
        format='full'
    ).execute()

    headers = msg['payload'].get('headers', [])

    sender = ""
    subject = ""

    for header in headers:
        if header['name'] == 'From':
            sender = header['value']

        if header['name'] == 'Subject':
            subject = header['value']

    body = ""

    try:
        data = msg['payload']['body']['data']
        body = base64.urlsafe_b64decode(data).decode('utf-8')
    except:
        continue

    prediction = classifier(body[:512])

    print("\n" + "=" * 80)
    print("Sender :", sender)
    print("Subject:", subject)
    print()

    label = prediction[0]["label"]
    score = prediction[0]["score"]

    print("Prediction :", label)
    print("Confidence :", round(score * 100, 2), "%")
    print("=" * 80)