import sqlite3
import time

import argon2
import pyotp
from argon2 import PasswordHasher
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

app = Flask(__name__)
ph = PasswordHasher()


# fonction pour tester le mot de passe
def check_password(password):
    has_num = False
    has_upper = False
    has_special = False
    for s in password:
        if s.isdigit():
            has_num = True
        if s.isupper():
            has_upper = True
        if not s.isdigit() and not s.isalpha():
            has_special = True
    return has_num and has_upper and has_special


# Fonction pour se connecter à la base de données
def get_db_connection():
    conn = sqlite3.connect("local.db")
    conn.row_factory = sqlite3.Row
    return conn


# Créer la table si elle n'existe pas
def init_db():
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            secret_totp TEXT NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()


# Route pour s'inscrire
@app.route("/register", methods=["POST"])
def register():
    username = request.json["username"]
    password = request.json["password"]
    secret_totp = request.json["totp_secret"]
    code = request.json["totp"]

    totp = pyotp.TOTP(secret_totp)

    if len(password) < 12:
        return jsonify({"message": "Password too short (min length: 12)"}), 400
    elif not check_password(password):
        return (
            jsonify(
                {
                    "message": "Password not valid !\n\
                    You need at least:\n\
                    - 1 Uppercase letter\n\
                    - 1 Number\n\
                    - 1 Special Caracter(#$%^*()@!...) "
                }
            ),
            400,
        )
    elif not totp.verify(code):
        return jsonify({"message": "Le code ne correspond pas !"}), 400
    password = ph.hash(password)
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password, secret_totp) VALUES (?, ?, ?)",
            (username, password, secret_totp),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"message": "Username already exists!"}), 400
    finally:
        conn.close()

    return jsonify({"message": "User registered successfully!"}), 201


# Route pour se connecter
@app.route("/login", methods=["POST"])
def login():
    username = request.json["username"]
    password = request.json["password"]
    code = request.json["totp"]

    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    if not user:
        return jsonify({"message": "Invalid credentials!"}), 401
    try:
        ph.verify(user["password"], password)
    except:
        return jsonify({"message": "Invalid credentials!"}), 401
    finally:
        totp = pyotp.TOTP(user["secret_totp"])
        if not totp.verify(code):
            return jsonify({"message": "Invalid credentials!"}), 401
        return jsonify({"message": "OK. Welcome user " + user["username"]}), 200


# Route pour la page d'inscription
@app.route("/web/register")
def register_page():
    totp_secret = pyotp.random_base32()
    return render_template("register_otp.html", totp_secret=totp_secret)


# Route pour la page de connexion
@app.route("/web/login")
def login_page():
    totp_secret = pyotp.random_base32()
    return render_template("login_otp.html")


# Route pour la racine
@app.route("/")
def home():
    return redirect(url_for("login_page"))


if __name__ == "__main__":
    init_db()  # Initialiser la base de données
    app.run(debug=True)
