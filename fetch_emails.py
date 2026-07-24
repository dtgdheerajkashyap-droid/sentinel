from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

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

print(f"\nFound {len(messages)} emails\n")

for message in messages:
    msg = service.users().messages().get(
        userId='me',
        id=message['id']
    ).execute()

    headers = msg['payload'].get('headers', [])

    subject = ""
    sender = ""

    for header in headers:
        if header['name'] == 'Subject':
            subject = header['value']

        if header['name'] == 'From':
            sender = header['value']

    print("=" * 50)
    print("Sender :", sender)
    print("Subject:", subject)
    print("=" * 50)