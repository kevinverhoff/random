import tkinter as tk
from tkinter import simpledialog, messagebox
import random

LUCKY_GREETINGS = [
    "Bonjour",       # French
    "Guten Tag",     # German
    "Hola",          # Spanish
    "Ciao",          # Italian
    "Konnichiwa",    # Japanese
    "Annyeonghaseyo",# Korean
    "Ni Hao",        # Chinese
    "Howdy",         # English (Southern US)
    "Good day to you",  # English (formal)
    "Aloha",         # Hawaiian
    "G'day",         # Australian English
    "What's up",     # English (casual)
    "Salut",         # French (informal)
    "Merhaba",       # Turkish
    "Namaste",       # Hindi/Sanskrit
]

root = tk.Tk()
root.withdraw()

name = simpledialog.askstring("Hello World", "What is your name?")

if not name:
    messagebox.showinfo("Goodbye", "No name entered. Goodbye!")
else:
    greeting_type = simpledialog.askstring(
        "Greeting Type",
        "What kind of greeting would you like?\nEnter: Normal or Lucky"
    )

    if not greeting_type:
        messagebox.showinfo("Goodbye", "No greeting type selected. Goodbye!")
    elif greeting_type.strip().lower() == "normal":
        messagebox.showinfo("Hello!", f"Hello, {name}!")
    elif greeting_type.strip().lower() == "lucky":
        greeting = random.choice(LUCKY_GREETINGS)
        messagebox.showinfo("Lucky Greeting!", f"{greeting}, {name}!")
    else:
        messagebox.showwarning("Unknown Choice", f"'{greeting_type}' is not a valid option. Please enter Normal or Lucky.")

root.destroy()
