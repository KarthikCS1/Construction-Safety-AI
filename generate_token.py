from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
import pickle

SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def main():
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=8080)
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)
    print("✅ token.pickle generated")

if __name__ == '__main__':
    main()