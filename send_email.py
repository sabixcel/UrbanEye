import smtplib
import os
from email.message import EmailMessage

def send_email_with_image(subject, body, image_path):
    sender_email = "onut.sabina27@yahoo.com"
    sender_password = ""
    receiver_email = "onut.sabina27@gmail.com"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg.set_content(body)

    ### attach the image
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename=os.path.basename(image_path))

    ### send email using gmail SMTP
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            #server.send_message(msg)
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False
