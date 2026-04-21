from azure.communication.email import EmailClient
from dotenv import load_dotenv
import os

load_dotenv()

connection_string = os.getenv("AZURE_COMMUNICATION_CONNECTION_STRING")
sender = os.getenv("AZURE_SENDER_EMAIL")

client = EmailClient.from_connection_string(connection_string)

mensaje = {
    "senderAddress": sender,
    "recipients": {"to": [{"address": "alexisgarciaherrera320@gmail.com"}]},
    "content": {
        "subject": "Prueba LostFinder AI",
        "html": "<h1>Si recibes esto, el email funciona ✅</h1>"
    }
}

try:
    poller = client.begin_send(mensaje)
    resultado = poller.result()
    print("Email enviado correctamente:", resultado)
except Exception as e:
    print("Error:", e)