import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email_to_student(student_email, answer):
    # Email configuration
    sender_email = "nktyagi423@gmail.com"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_username = "nktyagi423@gmail.com"
    smtp_password = "qgyd xtlb wcfh qdyv"

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = student_email
    message["Subject"] = "Answer to Your Query"
    # Add answer to email body
    email_body = f"\n{answer}\n"
    message.attach(MIMEText(email_body, "plain"))

    # Send email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(sender_email, student_email, message.as_string())


