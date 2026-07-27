import sqlite3
import os
from .interfaces import IAuthorizationService

class SQLiteAuthService(IAuthorizationService):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

    def _init_db(self):
        # Create database and table if not exists
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS authorized_plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT UNIQUE NOT NULL,
                owner_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def is_authorized(self, plate_text: str) -> bool:
        if not plate_text:
            return False
            
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM authorized_plates WHERE plate_text = ?", (plate_text.upper(),))
        result = cursor.fetchone()
        return result is not None

    def add_authorized_plate(self, plate_text: str, owner_name: str = ""):
        cursor = self.conn.cursor()
        try:
            cursor.execute("INSERT INTO authorized_plates (plate_text, owner_name) VALUES (?, ?)", 
                           (plate_text.upper(), owner_name))
            self.conn.commit()
            print(f"Added authorized plate: {plate_text.upper()} ({owner_name})")
        except sqlite3.IntegrityError:
            print(f"Plate {plate_text.upper()} already authorized.")
