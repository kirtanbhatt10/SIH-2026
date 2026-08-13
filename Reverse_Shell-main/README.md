Reverse Shell – Educational Cybersecurity Project
📌 Project Description

This repository contains a Reverse Shell implementation created strictly for educational and academic purposes.
The goal of this project is to help students and beginners understand how remote connections, client–server communication, and command execution mechanisms work at a fundamental level in cybersecurity.

⚠️ This project is NOT intended for illegal use.

🎯 Objective

The main objectives of this project are:

To understand how reverse connections are established

To study attacker techniques from a defensive learning perspective

To learn socket programming and system interaction

To spread awareness about why such techniques are dangerous when misused

🧠 Learning Outcomes

By studying this project, you will learn:

Client–server architecture

TCP socket communication

Remote command execution concepts

Module dependency handling in Python

Executable creation using PyInstaller

Importance of ethical hacking practices

🛠 Technologies Used

Python (Windows environment)

Socket Programming

PyInstaller

Wine (Linux → Windows executable build)

MSS (screen capture)

Pynput (keyboard input handling)

⚙ Build Command (Used in This Project)
wine ~/.wine/drive_c/Python27/python.exe -m PyInstaller reverse.py \
  --onefile --noconsole \
  --icon Dragon-Wallpaper-Chinese.ico \
  --hidden-import mss --hidden-import mss.windows \
  --hidden-import pynput --hidden-import pynput.keyboard \
  --hidden-import pynput._util --hidden-import pynput._util.win32 \
  --hidden-import keylogger \
  --add-data "Dragon-Wallpaper-Chinese.jpg;."

🔍 Purpose of This Command

Generates a single Windows executable

Includes all required hidden imports

Adds custom icon and resource files

Built using Wine for cross-platform compilation

📽 Demonstration

A step-by-step video demonstration is provided showing:

Project structure

How the reverse connection works

Execution flow

Security implications and risks

Defensive learning points

All demonstrations are performed in controlled lab environments only.

🔐 Usage Policy
✅ Allowed

Academic learning

College projects

Personal lab testing

Cybersecurity research

Awareness and defense study

❌ Not Allowed

Unauthorized access

Attacking real systems

Malicious distribution

Any illegal or unethical activity

⚠️ Legal & Ethical Disclaimer

This project is published only for educational purposes.
The author takes no responsibility for misuse of this code.
Always follow local laws, institutional rules, and ethical hacking guidelines.

📚 Educational Note

Understanding offensive security techniques is essential for building strong defensive systems.
This project exists to educate students and security enthusiasts, not to exploit systems.

👨‍🎓 Author

Tejas Padia
Cybersecurity Student
📍 India
🔗 LinkedIn: https://www.linkedin.com/in/tejas-padia-b56b1933b/
