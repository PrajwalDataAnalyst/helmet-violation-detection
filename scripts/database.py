import mysql.connector
from datetime import datetime

def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='prajwal@12345',
        database='helmet_fines'
    )

def update_fine(plate_number, fine_amount=100):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT fine_amount FROM fines WHERE plate_number=%s", (plate_number,))
    result = cursor.fetchone()

    if result:
        new_fine = result[0] + fine_amount
        cursor.execute(
            "UPDATE fines SET fine_amount=%s, last_fined=%s WHERE plate_number=%s",
            (new_fine, datetime.now(), plate_number)
        )
        message = f"✅ Updated fine for bike {plate_number}: ₹{new_fine}"
    else:
        cursor.execute(
            "INSERT INTO fines (plate_number, fine_amount, last_fined) VALUES (%s, %s, %s)",
            (plate_number, fine_amount, datetime.now())
        )
        message = f"💸 Fine registered for bike {plate_number}: ₹{fine_amount}"

    conn.commit()
    cursor.close()
    conn.close()
    return message
