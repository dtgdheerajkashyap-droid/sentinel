from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

flow = InstalledAppFlow.from_client_secrets_file(
    'config/credentials.json',
    SCOPES
)

creds = flow.run_local_server(port=0)

service = build('gmail', 'v1', credentials=creds)

results = service.users().messages().list(
    userId='me',
    maxResults=3
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

    print("\n" + "="*70)
    print("Sender :", sender)
    print("Subject:", subject)

    try:
        data = msg['payload']['body']['data']
        body = base64.urlsafe_b64decode(data).decode('utf-8')
        print("\nBody:\n")
        print(body[:1000])
    except:
        print("\nBody could not be extracted.")