import smtplib
import ssl
import os
from email.message import EmailMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

if not SENDER_EMAIL or not EMAIL_PASSWORD:
    print("Error: SENDER_EMAIL or EMAIL_PASSWORD not found in .env file.")
    exit()

context = ssl.create_default_context()

print("Email Sender Script Started. Type 'exit' at any prompt to quit.")

while True:
    try:
        receiver_email = input("\nEnter receiver's email address: ").strip()
        if receiver_email.lower() == 'exit':
            print("Exiting script...")
            break

        message_body = input("Enter message to send: ")
        if message_body.lower() == 'exit':
            print("Exiting script...")
            break

        em = EmailMessage()
        em['From'] = SENDER_EMAIL
        em['To'] = receiver_email
        em['Subject'] = "LOW STOCK INVENTORY ALERT"
        em.set_content(message_body)

        print("Sending email...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(SENDER_EMAIL, EMAIL_PASSWORD)
            smtp.sendmail(SENDER_EMAIL, receiver_email, em.as_string())
            print(f"Email successfully sent to {receiver_email}!")

    except Exception as e:
        print(f"An error occurred: {e}")
