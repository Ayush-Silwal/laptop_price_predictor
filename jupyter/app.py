from flask import Flask, render_template, request, redirect, session, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from db_connection import create_connection, close_connection
from customModel import DecisionTree,RandomForest
import re
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the model and data
pipe = pickle.load(open("pipe.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html', 
                           companies=df["Company"].unique(), 
                           types=df["TypeName"].unique(),
                           rams=df["Ram"].unique(),
                           cpus=df["Cpu brand"].unique(),
                           gpus=df["Gpu brand"].unique(),
                           oss=df["os"].unique())

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    type_ = request.form.get('type')
    ram = int(request.form.get('ram'))
    weight = float(request.form.get('weight'))
    touchscreen = 1 if request.form.get('touchscreen') == 'Yes' else 0
    ips = 1 if request.form.get('ips') == 'Yes' else 0
    screen_size = float(request.form.get('screen_size'))
    resolution = request.form.get('resolution')
    cpu = request.form.get('cpu')
    hdd = int(request.form.get('HDD'))
    ssd = int(request.form.get('SSD'))
    gpu = request.form.get('gpu')
    os = request.form.get('os')

    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    query = np.array([company, type_, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)

    predicted_price = int(np.exp(pipe.predict(query)[0]))

    return render_template('index.html', 
                           companies=df["Company"].unique(), 
                           types=df["TypeName"].unique(),
                           rams=df["Ram"].unique(),
                           cpus=df["Cpu brand"].unique(),
                           gpus=df["Gpu brand"].unique(),
                           oss=df["os"].unique(),
                           predicted_price=predicted_price)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        errors = {}

        # Server-side validation
        if not username:
            errors['username'] = "Please enter your username"
        if not password:
            errors['password'] = "Please enter your password"

        # If there are errors, re-render the form with error messages
        if errors:
            return render_template('login.html', errors=errors)

        # Authenticate user
        connection = create_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                query = "SELECT * FROM users WHERE username = %s"
                cursor.execute(query, (username,))
                user = cursor.fetchone()
                cursor.close()

                if user and check_password_hash(user['password'], password):
                    # Successful login, set session, etc.
                    session['user_id'] = user['uid']
                    return redirect(url_for('dashboard'))
                else:
                    errors['general'] = "Invalid username or password"
                    return render_template('login.html', errors=errors)

            except Exception as e:
                flash(f"An error occurred: {e}", 'error')
                return render_template('login.html', errors=errors)
            finally:
                close_connection(connection)
    
    return render_template('login.html')
    
@app.route('/admin login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        admin_username = request.form['username']
        admin_password = request.form['password']
        errors = {}

        if not admin_username:
            errors['username'] = "Please enter the admin username"
        if not admin_password:
            errors['password'] = "Please enter the admin password"

        if errors:
            return render_template('admin login.html', errors=errors)

        ADMIN_CREDENTIALS = {
            'username': 'ayush',  #admin username
            'password': 'ayush123'  #admin password
        }

        if admin_username == ADMIN_CREDENTIALS['username'] and admin_password == ADMIN_CREDENTIALS['password']:
            session['admin_logged_in'] = True
            return redirect(url_for('admindashboard'))
        else:
            errors['general'] = "Invalid username or password"
            return render_template('admin login.html', errors=errors)

    return render_template('admin login.html')
        

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        cpassword = request.form.get('confirmPassword')

        if len(username) < 8 or not re.match(r'^[A-Za-z][A-Za-z0-9]{7,19}$', username):
            flash('Username must be valid and at least 8 characters long', 'error')
            return redirect(url_for('signup'))

        if not re.match(r'^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$', email):
            flash('Invalid email address', 'error')
            return redirect(url_for('signup'))

        if len(password) < 8 or not re.match(r'^[a-zA-Z0-9]{8,20}$', password):
            flash('Password must be between 8 and 20 characters long', 'error')
            return redirect(url_for('signup'))

        if password != cpassword:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        connection = create_connection()
        if connection:
            try:
                cursor = connection.cursor()
                insert_query = """
                INSERT INTO users (username, email, password) 
                VALUES (%s, %s, %s)
                """
                cursor.execute(insert_query, (username, email, hashed_password))
                connection.commit()
                cursor.close()
            except Exception as e:
                flash(f"An error occurred: {e}", 'error')
                return redirect(url_for('signup'))
            finally:
                close_connection(connection)

        flash('You have successfully registered', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        company = request.form.get('company')
        type_ = request.form.get('type')
        ram = int(request.form.get('ram'))
        weight = float(request.form.get('weight'))
        touchscreen = 1 if request.form.get('touchscreen') == 'Yes' else 0
        ips = 1 if request.form.get('ips') == 'Yes' else 0
        screen_size = float(request.form.get('screen_size'))
        resolution = request.form.get('resolution')
        cpu = request.form.get('cpu')
        hdd = int(request.form.get('HDD'))
        ssd = int(request.form.get('SSD'))
        gpu = request.form.get('gpu')
        os = request.form.get('os')

        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

        query = np.array([company, type_, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
        query = query.reshape(1, 12)

        predicted_price = int(np.exp(pipe.predict(query)[0]))

        connection = create_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("INSERT INTO predictions (uid, company, type, ram, weight, touchscreen, ips, screen_size, resolution, cpu, hdd, ssd, gpu, os, predicted_price) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                                   (session['user_id'], company, type_, ram, weight, touchscreen, ips, screen_size, resolution, cpu, hdd, ssd, gpu, os, predicted_price))
                connection.commit()
            except Exception as e:
                flash(f"An error occurred: {e}", 'error')
            finally:
                close_connection(connection)

        return render_template('dashboard.html', 
                               companies=df["Company"].unique(), 
                               types=df["TypeName"].unique(),
                               rams=df["Ram"].unique(),
                               cpus=df["Cpu brand"].unique(),
                               gpus=df["Gpu brand"].unique(),
                               oss=df["os"].unique(),
                               predicted_price=predicted_price)

    return render_template('dashboard.html', 
                           companies=df["Company"].unique(), 
                           types=df["TypeName"].unique(),
                           rams=df["Ram"].unique(),
                           cpus=df["Cpu brand"].unique(),
                           gpus=df["Gpu brand"].unique(),
                           oss=df["os"].unique())

@app.route('/predictionhistory')
def prediction_history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    connection = create_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM predictions WHERE uid = %s ORDER BY created_at DESC", (session['user_id'],))
                history = cursor.fetchall()
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
            history = []
        finally:
            close_connection(connection)

    return render_template('predictionhistory.html', history=history)

@app.route('/admindashboard')
def admindashboard():
    if 'admin_logged_in' not in session:
        return redirect(url_for('admin_login'))

    connection = create_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM users")
                users = cursor.fetchall()
                cursor.execute("SELECT * FROM predictions")
                predictions = cursor.fetchall()
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
            users = []
            predictions = []
        finally:
            close_connection(connection)

    return render_template('admindashboard.html', users=users, predictions=predictions)

@app.route('/userlist')
def user_list():
    if 'admin_logged_in' not in session:
        return redirect(url_for('admin_login'))

    connection = create_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM users")
                users = cursor.fetchall()
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
            users = []
        finally:
            close_connection(connection)

    return render_template('user_list.html', users=users)

@app.route('/userprediction')
def user_prediction():
    if 'admin_logged_in' not in session:
        return redirect(url_for('admin_login'))

    connection = create_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT ALL * FROM predictions " )
                predictions = cursor.fetchall()
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
            predictions = []
        finally:
            close_connection(connection)

    return render_template('user_prediction.html', predictions=predictions)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))



if __name__ == '__main__':
    app.run(debug=True)
