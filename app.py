import os
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash, jsonify, get_flashed_messages
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from werkzeug.utils import secure_filename
from flask_mysqldb import MySQL
from dotenv import load_dotenv
import pandas as pd
import uuid
import MySQLdb
import subprocess
import threading
from werkzeug.security import generate_password_hash, check_password_hash
import re
from datetime import datetime
import tempfile
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'temp-secret-key')
# MySQL Config
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', '')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB', 'voterdata')

# File upload config
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB Limit
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Init MySQL
mysql = MySQL(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Simple User model for login
class User(UserMixin):
    def __init__(self, id, username, is_admin=False):
        self.id = id
        self.username = username
        self.is_admin = is_admin

@login_manager.user_loader
def load_user(user_id):
    try:
        db = MySQLdb.connect(host='localhost', user='root', passwd='', db=app.config['MYSQL_DB'])
        cur = db.cursor()
        cur.execute('SELECT id, username, is_admin FROM users WHERE id=%s', (user_id,))
        row = cur.fetchone()
        cur.close()
        db.close()
        if row:
            return User(str(row[0]), row[1], is_admin=bool(row[2]))
        return None
    except Exception as e:
        print(f"load_user error: {e}")
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    job = None
    table_html = None
    jobs_list = []
    running_jobs = []
    selected_job_id = requestArgsJobId = request.args.get('job_id', type=int)
    try:
        db = MySQLdb.connect(host='localhost', user='root', passwd='', db=app.config['MYSQL_DB'])
        cur = db.cursor()
        # Running/pending jobs indicator
        cur.execute(
            'SELECT ej.id, u.filename, ej.status FROM extraction_jobs ej '
            'JOIN uploads u ON ej.upload_id=u.id '
            'JOIN users us ON u.user_id=us.id '
            'WHERE us.id=%s AND ej.status IN ("pending","running") ORDER BY ej.id DESC', (current_user.id,)
        )
        for r in cur.fetchall():
            running_jobs.append({'id': r[0], 'filename': r[1], 'status': r[2]})
        # Fetch all finished jobs with existing files for selector
        cur.execute(
            'SELECT ej.id, u.filename, ej.result_file '
            'FROM extraction_jobs ej '
            'JOIN uploads u ON ej.upload_id = u.id '
            'JOIN users us ON u.user_id = us.id '
            'WHERE us.id=%s AND ej.status="finished" AND ej.result_file IS NOT NULL '
            'ORDER BY ej.id DESC', (current_user.id,)
        )
        rows = cur.fetchall()
        for r in rows:
            rf = r[2]
            if rf and os.path.exists(rf):
                jobs_list.append({ 'id': r[0], 'filename': r[1], 'result_file': rf })
        # choose selected job or latest
        if not selected_job_id and jobs_list:
            selected_job_id = jobs_list[0]['id']
        if selected_job_id:
            cur.execute(
                'SELECT ej.id, ej.status, ej.result_file '
                'FROM extraction_jobs ej '
                'JOIN uploads u ON ej.upload_id = u.id '
                'JOIN users us ON u.user_id = us.id '
                'WHERE us.id=%s AND ej.id=%s LIMIT 1', (current_user.id, selected_job_id)
            )
            row = cur.fetchone()
        else:
            row = None
        if row:
            job = { 'id': row[0], 'status': row[1], 'result_file': row[2] }
            if job['status'] == 'finished' and job['result_file'] and os.path.exists(job['result_file']):
                try:
                    if job['result_file'].lower().endswith('.xlsx'):
                        df = pd.read_excel(job['result_file'])
                    else:
                        df = pd.read_csv(job['result_file'])
                    table_html = df.to_html(classes='table table-sm table-striped', index=False, border=0)
                except Exception as e:
                    print(f"Render table error: {e}")
        cur.close(); db.close()
    except Exception as e:
        print(f"Index DB error: {e}")
    return render_template('index.html', user=current_user, job=job, table_html=table_html, jobs_list=jobs_list, selected_job_id=selected_job_id, running_jobs=running_jobs)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')
        address = request.form.get('address')
        gender = request.form.get('gender')
        age = request.form.get('age')
        profession = request.form.get('profession')
        errors = []
        # Validation
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            errors.append("Invalid email format.")
        if not password or len(password) < 6:
            errors.append("Password must be at least 6 characters.")
        if not name:
            errors.append("Name required.")
        if not address or not gender or not age or not profession:
            errors.append("All fields required.")
        if errors:
            for e in errors:
                flash(e, 'danger')
            return render_template('register.html')
        db = MySQLdb.connect(host='localhost', user='root', passwd='', db=app.config['MYSQL_DB'])
        cur = db.cursor()
        cur.execute('SELECT id FROM users WHERE email=%s', (email,))
        if cur.fetchone():
            flash('Email already registered.', 'danger')
            cur.close()
            db.close()
            return render_template('register.html')
        pw_hash = generate_password_hash(password)
        cur.execute('INSERT INTO users (email, username, password_hash, is_admin, name, address, gender, age, profession) VALUES (%s, %s, %s, 0, %s, %s, %s, %s, %s)',
                    (email, email, pw_hash, name, address, gender, age, profession))
        db.commit()
        cur.close()
        db.close()
        flash('Registration complete! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

# Update login route for real DB users
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('username')
        pw = request.form.get('password')
        db = MySQLdb.connect(host='localhost', user='root', passwd='', db=app.config['MYSQL_DB'])
        cur = db.cursor()
        cur.execute('SELECT id, username, password_hash, is_admin FROM users WHERE email=%s', (email,))
        user = cur.fetchone()
        cur.close()
        db.close()
        # Debug block
        print("DB user:", user)
        if user:
            print("Password Hash:", user[2])
            print("Password Entered:", pw)
            print("Check:", check_password_hash(user[2], pw))
        if user and check_password_hash(user[2], pw):
            login_user(User(str(user[0]), user[1], is_admin=bool(user[3])))
            return redirect(url_for('index'))
        flash('Invalid login', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

USAGE_LIMIT = 5

@app.context_processor
def inject_globals():
    usage_left = None
    has_api_key = False
    if hasattr(current_user, 'id') and current_user.is_authenticated:
        try:
            db = get_db(); cur = db.cursor()
            cur.execute('SELECT gemini_api_key, extraction_uses, extraction_uses_date FROM users WHERE id=%s', (current_user.id,))
            row = cur.fetchone()
            if row:
                api_key, uses, uses_date = row[0], int(row[1] or 0), row[2]
                # reset per day
                from datetime import date
                today = date.today()
                if uses_date is None or uses_date != today:
                    try:
                        cur.execute('UPDATE users SET extraction_uses=0, extraction_uses_date=%s WHERE id=%s', (today, current_user.id))
                        db.commit(); uses = 0
                    except Exception:
                        pass
                has_api_key = bool(api_key) and len(str(api_key).strip()) > 0
                usage_left = max(0, USAGE_LIMIT - uses)
            cur.close(); db.close()
        except Exception:
            pass
    return dict(get_flashed_messages=get_flashed_messages, usage_left=usage_left, has_api_key=has_api_key)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    msg = None
    api_key = ''
    uses = 0
    try:
        db = get_db(); cur = db.cursor()
        if request.method == 'POST':
            if 'delete_key' in request.form:
                cur.execute('UPDATE users SET gemini_api_key=NULL WHERE id=%s', (current_user.id,))
                db.commit(); flash('API key removed.', 'success')
            else:
                new_key = request.form.get('api_key', '').strip()
                cur.execute('UPDATE users SET gemini_api_key=%s WHERE id=%s', (new_key, current_user.id))
                db.commit(); flash('API key updated.', 'success')
        cur.execute('SELECT gemini_api_key, extraction_uses FROM users WHERE id=%s', (current_user.id,))
        row = cur.fetchone(); cur.close(); db.close()
        if row:
            api_key = row[0] or ''
            uses = int(row[1] or 0)
    except Exception as e:
        flash(f'Settings load error: {e}', 'danger')
    masked = ('*' * len(api_key)) if api_key else ''
    return render_template('settings.html', user=current_user, api_key_masked=masked, has_key=bool(api_key), uses=uses, limit=USAGE_LIMIT)

def launch_extraction_job(job_id, pdf_path, location_prefix='GEN', user_api_key=None):
    def run_extraction():
        db = MySQLdb.connect(host='localhost', user='root', passwd='', db=app.config['MYSQL_DB'])
        cur = db.cursor()
        cur.execute("UPDATE extraction_jobs SET status='running' WHERE id=%s", (job_id,))
        db.commit()
        try:
            env = os.environ.copy()
            if user_api_key:
                env['GEMINI_API_KEY'] = user_api_key
            proc = subprocess.Popen(['python', 'main.py', location_prefix, pdf_path], env=env)
            # store pid
            try:
                cur.execute('UPDATE extraction_jobs SET pid=%s WHERE id=%s', (proc.pid, job_id))
                db.commit()
            except Exception:
                pass
            proc.wait()
            if proc.returncode == 0:
                folder_path = './output'
                pdf_basename = os.path.basename(pdf_path)
                file_stem = os.path.splitext(pdf_basename)[0]
                excel_path = os.path.join(folder_path, f"{location_prefix}_{file_stem}_processed.xlsx")
                cur.execute("UPDATE extraction_jobs SET status='finished', result_file=%s, pid=NULL WHERE id=%s", (excel_path, job_id))
            else:
                cur.execute("UPDATE extraction_jobs SET status='error', error_message=%s, pid=NULL WHERE id=%s", (f'Process exited {proc.returncode}', job_id))
        except Exception as e:
            cur.execute("UPDATE extraction_jobs SET status='error', error_message=%s, pid=NULL WHERE id=%s", (str(e), job_id))
        db.commit(); cur.close(); db.close()
    threading.Thread(target=run_extraction).start()

@app.route('/job/cancel/<int:job_id>', methods=['POST'])
@login_required
def cancel_job(job_id):
    try:
        db = get_db(); cur = db.cursor()
        cur.execute('SELECT ej.pid FROM extraction_jobs ej JOIN uploads u ON ej.upload_id=u.id WHERE ej.id=%s AND u.user_id=%s', (job_id, current_user.id))
        row = cur.fetchone()
        pid = row[0] if row else None
        if not pid:
            flash('Job not cancellable or already finished.', 'warning'); cur.close(); db.close(); return redirect(url_for('index'))
        try:
            import platform
            if platform.system().lower().startswith('win'):
                subprocess.run(['taskkill','/PID',str(pid),'/F','/T'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.kill(int(pid), 9)
        except Exception as e:
            print(f"cancel kill error: {e}")
        cur.execute("UPDATE extraction_jobs SET status='cancelled', pid=NULL WHERE id=%s", (job_id,))
        db.commit(); cur.close(); db.close()
        flash('Job cancelled.', 'info')
    except Exception as e:
        flash(f'Cancel failed: {e}', 'danger')
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'pdfs' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    files = request.files.getlist('pdfs')
    if len(files) > 5:
        flash('Upload max 5 PDFs at once')
        return redirect(url_for('index'))
    db = get_db(); cur = db.cursor()
    cur.execute('SELECT gemini_api_key, extraction_uses, extraction_uses_date FROM users WHERE id=%s', (current_user.id,))
    row = cur.fetchone()
    user_api_key = (row[0] or '').strip() if row else ''
    uses = int(row[1] or 0) if row else 0
    from datetime import date
    today = date.today()
    if row and (row[2] is None or row[2] != today):
        # reset daily counter
        cur.execute('UPDATE users SET extraction_uses=0, extraction_uses_date=%s WHERE id=%s', (today, current_user.id))
        db.commit(); uses = 0
    if not user_api_key and uses >= USAGE_LIMIT:
        cur.close(); db.close()
        flash('Daily limit reached (5). Add your own Gemini API key in Settings to continue, or try tomorrow.', 'warning')
        return redirect(url_for('index'))
    saved_paths = []
    job_ids = []
    for file in files:
        if file and allowed_file(file.filename):
            if not user_api_key and uses >= USAGE_LIMIT:
                break
            filename = secure_filename(str(uuid.uuid4()) + '_' + file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            saved_paths.append(save_path)
            cur.execute('INSERT INTO uploads (user_id, filename) VALUES (%s, %s)', (current_user.id, filename))
            upload_id = cur.lastrowid
            cur.execute('INSERT INTO extraction_jobs (upload_id, status) VALUES (%s, %s)', (upload_id, 'pending'))
            job_id = cur.lastrowid; job_ids.append(job_id); db.commit()
            if not user_api_key:
                cur.execute('UPDATE users SET extraction_uses = extraction_uses + 1, extraction_uses_date=%s WHERE id=%s', (today, current_user.id))
                db.commit(); uses += 1
            launch_extraction_job(job_id, save_path, location_prefix=str(current_user.id), user_api_key=(user_api_key or None))
    cur.close(); db.close()
    if not saved_paths:
        return redirect(url_for('index'))
    if not user_api_key and uses >= USAGE_LIMIT:
        flash('Daily usage limit reached during this upload. Remaining files were not started.', 'warning')
    else:
        flash(f'Extraction started for {len(saved_paths)} pdf(s)!')
    return redirect(url_for('index'))

# Utility for marking job as finished (CLI)
def mark_job_finished(job_id, status):
    db = MySQLdb.connect(host='localhost', user='root', passwd='', db=app.config['MYSQL_DB'])
    cur = db.cursor()
    cur.execute('UPDATE extraction_jobs SET status=%s WHERE id=%s', (status, job_id))
    db.commit()
    cur.close()
    db.close()

# DB Auto-migrate on startup
migration_done = False

def migrate_db_once():
    global migration_done
    if migration_done:
        return
    try:
        db = MySQLdb.connect(host='localhost', user='root', passwd='', db=app.config['MYSQL_DB'])
        cursor = db.cursor()
        with open('db_init.sql', 'r') as f:
            sql = f.read()
        for stmt in sql.split(';'):
            stmt = stmt.strip()
            if stmt:
                cursor.execute(stmt)
        # Ensure new columns exist in users
        try:
            cursor.execute("SHOW COLUMNS FROM users LIKE 'gemini_api_key'")
            if cursor.fetchone() is None:
                cursor.execute("ALTER TABLE users ADD COLUMN gemini_api_key TEXT NULL")
            cursor.execute("SHOW COLUMNS FROM users LIKE 'extraction_uses'")
            if cursor.fetchone() is None:
                cursor.execute("ALTER TABLE users ADD COLUMN extraction_uses INT DEFAULT 0")
            cursor.execute("SHOW COLUMNS FROM users LIKE 'extraction_uses_date'")
            if cursor.fetchone() is None:
                cursor.execute("ALTER TABLE users ADD COLUMN extraction_uses_date DATE NULL")
        except Exception as e:
            print(f"Column ensure error: {e}")
        # Ensure pid column exists in extraction_jobs
        try:
            cursor.execute("SHOW COLUMNS FROM extraction_jobs LIKE 'pid'")
            if cursor.fetchone() is None:
                cursor.execute("ALTER TABLE extraction_jobs ADD COLUMN pid INT NULL")
        except Exception as e:
            print(f"jobs pid column ensure error: {e}")
        db.commit()
        cursor.close()
        db.close()
        migration_done = True
    except Exception as e:
        print(f"DB migration error: {e}")
        migration_done = True  # Prevents endless loop if repeated failure

@app.before_request
def ensure_db_migrated():
    migrate_db_once()

# Utility: Show flash errors on all templates, progress bar status
# Error/flashing wrapper
import traceback

def handle_error(msg, exception=None):
    if exception:
        print(traceback.format_exc())
    flash(msg, 'danger')

# Extraction job status endpoint (for progress bar AJAX)
@app.route('/job-status/<int:job_id>')
def job_status(job_id):
    # (Stub: Replace with real DB status lookup)
    # Example: return current percent, status, error msg
    db = MySQLdb.connect(host='localhost', user='root', passwd='', db=app.config['MYSQL_DB'])
    cursor = db.cursor()
    cursor.execute('SELECT status, error_message FROM extraction_jobs WHERE id=%s', (job_id,))
    row = cursor.fetchone()
    percent = 0 if not row or row[0]!="finished" else 100
    status = row[0] if row else 'unknown'
    error = row[1] if row else ''
    cursor.close()
    db.close()
    return jsonify({"percent": percent, "status": status, "error": error})

def get_db():
    return MySQLdb.connect(host='localhost', user='root', passwd='', db=app.config['MYSQL_DB'])

@app.route('/history')
@login_required
def history():
    jobs = []
    dedupes = []
    try:
        db = get_db()
        cur = db.cursor()
        # Extraction jobs
        cur.execute(
            'SELECT ej.id, u.filename, ej.status, ej.result_file, ej.started_at, ej.finished_at '
            'FROM extraction_jobs ej '
            'JOIN uploads u ON ej.upload_id = u.id '
            'WHERE u.user_id=%s ORDER BY ej.id DESC', (current_user.id,)
        )
        for row in cur.fetchall():
            jobs.append({
                'id': row[0], 'filename': row[1], 'status': row[2],
                'result_file': row[3], 'started_at': row[4], 'finished_at': row[5]
            })
        # Saved dedupe results (download history)
        cur.execute(
            'SELECT dr.id, dr.result_file, dr.fields_compared, dr.created_at, ej.id '
            'FROM dedupe_results dr '
            'JOIN extraction_jobs ej ON dr.job_id = ej.id '
            'JOIN uploads u ON ej.upload_id = u.id '
            'WHERE u.user_id=%s ORDER BY dr.id DESC', (current_user.id,)
        )
        for row in cur.fetchall():
            dedupes.append({
                'id': row[0],
                'result_file': row[1],
                'fields': row[2],
                'created_at': row[3],
                'job_id': row[4]
            })
        cur.close(); db.close()
    except Exception as e:
        print(f"History error: {e}")
        flash('Failed to load history', 'danger')
    return render_template('history.html', user=current_user, jobs=jobs, dedupes=dedupes)

@app.route('/download/<int:job_id>')
@login_required
def download(job_id):
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute(
            'SELECT ej.result_file FROM extraction_jobs ej '\
            'JOIN uploads u ON ej.upload_id = u.id '\
            'WHERE ej.id=%s AND u.user_id=%s', (job_id, current_user.id)
        )
        row = cur.fetchone()
        cur.close(); db.close()
        if not row or not row[0]:
            flash('File not available', 'warning')
            return redirect(url_for('history'))
        path = row[0]
        if not os.path.exists(path):
            flash('File missing on server', 'danger')
            return redirect(url_for('history'))
        return send_file(path, as_attachment=True)
    except Exception as e:
        print(f"Download error: {e}")
        flash('Download failed', 'danger')
        return redirect(url_for('history'))

def slugify_fields(fields):
    joined = '-'.join(fields).lower().replace('/', '-').replace(' ', '-')
    return re.sub(r'[^a-z0-9\-]', '', joined)[:60]

@app.route('/dedupe', methods=['POST'])
@login_required
def dedupe():
    fields = request.form.getlist('fields')
    if not fields or len(fields) < 1:
        flash('Select at least 1 field for duplicate checking.', 'warning')
        return redirect(url_for('index'))
    if len(fields) > 4:
        flash('Select at most 4 fields.', 'warning')
        return redirect(url_for('index'))
    # Cleanup previous temp if present
    prev_tmp = session.get('dedupe_tmp')
    prev_run = session.get('dedupe_run_id')
    if prev_tmp and os.path.exists(prev_tmp):
        try:
            os.remove(prev_tmp)
        except Exception:
            pass
    if prev_run:
        try:
            db = get_db(); cur = db.cursor()
            cur.execute('UPDATE dedupe_runs SET status="discarded", updated_at=NOW() WHERE id=%s AND user_id=%s', (prev_run, current_user.id))
            db.commit(); cur.close(); db.close()
        except Exception as e:
            print(f"dedupe prev_run update error: {e}")
    # Load latest finished job
    db = get_db(); cur = db.cursor()
    cur.execute(
        'SELECT ej.id, ej.result_file FROM extraction_jobs ej '
        'JOIN uploads u ON ej.upload_id = u.id '
        'WHERE u.user_id=%s AND ej.status="finished" ORDER BY ej.id DESC LIMIT 1', (current_user.id,)
    )
    row = cur.fetchone()
    cur.close(); db.close()
    if not row:
        flash('No finished extraction to dedupe.', 'warning')
        return redirect(url_for('index'))
    job_id = row[0]
    result_path = row[1]
    if not result_path or not os.path.exists(result_path):
        flash('Extracted file missing for dedupe.', 'danger')
        return redirect(url_for('index'))
    # Load dataframe
    try:
        if result_path.lower().endswith('.xlsx'):
            df = pd.read_excel(result_path)
        else:
            df = pd.read_csv(result_path)
    except Exception as e:
        flash(f'Failed to read extracted data: {e}', 'danger')
        return redirect(url_for('index'))
    # Compute duplicates
    try:
        subset = [c for c in fields if c in df.columns]
        if len(subset) < len(fields):
            missing = list(set(fields) - set(subset))
            flash(f'Missing columns in data: {", ".join(missing)}', 'danger')
            return redirect(url_for('index'))
        dup_df = df[df.duplicated(subset=subset, keep=False)].copy()
        message = None
        if dup_df.empty:
            message = 'No duplicates found for selected fields.'
            session['dedupe_tmp'] = None
            session['dedupe_fields'] = ', '.join(fields)
            session['dedupe_job'] = job_id
            session['dedupe_run_id'] = None
            return render_template('dedupe.html', user=current_user, fields=fields, table_html=None, tmp_path=None, message=message)
        dup_df.sort_values(by=subset + (["Page_Number"] if "Page_Number" in dup_df.columns else []), inplace=True)
        dup_df['Duplicate_Group_ID'] = dup_df.groupby(subset).ngroup() + 1
        # Unique filename with user, job, and fields
        os.makedirs('output', exist_ok=True)
        field_slug = slugify_fields(fields)
        tmp_name = f"output/dedupe_u{current_user.id}_j{job_id}_{field_slug}_{int(time.time())}.xlsx"
        dup_df.to_excel(tmp_name, index=False, engine='openpyxl')
        # Log run
        run_id = None
        try:
            db = get_db(); cur = db.cursor()
            cur.execute('INSERT INTO dedupe_runs (user_id, job_id, fields_compared, result_file, status) VALUES (%s,%s,%s,%s,%s)',
                        (current_user.id, job_id, ', '.join(fields), tmp_name, 'temp'))
            db.commit(); run_id = cur.lastrowid; cur.close(); db.close()
        except Exception as e:
            print(f"dedupe run log error: {e}")
        # Stash in session for optional save
        session['dedupe_tmp'] = tmp_name
        session['dedupe_fields'] = ', '.join(fields)
        session['dedupe_job'] = job_id
        session['dedupe_run_id'] = run_id
        table_html = dup_df.to_html(classes='table table-sm table-striped', index=False, border=0)
        return render_template('dedupe.html', user=current_user, fields=fields, table_html=table_html, tmp_path=tmp_name, message=None)
    except Exception as e:
        flash(f'Failed to compute duplicates: {e}', 'danger')
        return redirect(url_for('index'))

@app.route('/dedupe/save', methods=['POST'])
@login_required
def dedupe_save():
    tmp = session.get('dedupe_tmp')
    job_id = session.get('dedupe_job')
    fields = session.get('dedupe_fields', '')
    run_id = session.get('dedupe_run_id')
    if not tmp or not os.path.exists(tmp):
        flash('No dedupe result to save.', 'warning')
        return redirect(url_for('index'))
    try:
        if job_id:
            db = get_db(); cur = db.cursor()
            cur.execute('INSERT INTO dedupe_results (job_id, fields_compared, result_file) VALUES (%s, %s, %s)', (job_id, fields, tmp))
            if run_id:
                cur.execute('UPDATE dedupe_runs SET status="saved", updated_at=NOW() WHERE id=%s AND user_id=%s', (run_id, current_user.id))
            db.commit(); cur.close(); db.close()
    except Exception as e:
        print(f"dedupe_save persist error: {e}")
    session.pop('dedupe_tmp', None)
    session.pop('dedupe_job', None)
    session.pop('dedupe_fields', None)
    session.pop('dedupe_run_id', None)
    return send_file(tmp, as_attachment=True)

@app.route('/dedupe/discard', methods=['POST'])
@login_required
def dedupe_discard():
    tmp = session.get('dedupe_tmp')
    run_id = session.get('dedupe_run_id')
    if tmp and os.path.exists(tmp):
        try:
            os.remove(tmp)
        except Exception:
            pass
    if run_id:
        try:
            db = get_db(); cur = db.cursor()
            cur.execute('UPDATE dedupe_runs SET status="discarded", updated_at=NOW() WHERE id=%s AND user_id=%s', (run_id, current_user.id))
            db.commit(); cur.close(); db.close()
        except Exception as e:
            print(f"dedupe_discard log error: {e}")
    session.pop('dedupe_tmp', None)
    session.pop('dedupe_job', None)
    session.pop('dedupe_fields', None)
    session.pop('dedupe_run_id', None)
    flash('Temporary dedupe file discarded.', 'info')
    return redirect(url_for('index'))

@app.route('/delete-job-file/<int:job_id>', methods=['POST'])
@login_required
def delete_job_file(job_id):
    try:
        db = get_db(); cur = db.cursor()
        cur.execute('SELECT ej.result_file FROM extraction_jobs ej JOIN uploads u ON ej.upload_id=u.id WHERE ej.id=%s AND u.user_id=%s', (job_id, current_user.id))
        row = cur.fetchone()
        if not row or not row[0]:
            flash('No file to delete.', 'warning'); cur.close(); db.close(); return redirect(url_for('history'))
        path = row[0]
        try:
            if os.path.exists(path): os.remove(path)
        except Exception:
            pass
        cur.execute('UPDATE extraction_jobs SET result_file=NULL WHERE id=%s', (job_id,))
        db.commit(); cur.close(); db.close()
        flash('File deleted from server.', 'success')
    except Exception as e:
        flash(f'Delete failed: {e}', 'danger')
    return redirect(url_for('history'))

@app.route('/dedupe/delete/<int:dedupe_id>', methods=['POST'])
@login_required
def delete_dedupe_file(dedupe_id):
    try:
        db = get_db(); cur = db.cursor()
        cur.execute('SELECT dr.result_file FROM dedupe_results dr JOIN extraction_jobs ej ON dr.job_id=ej.id JOIN uploads u ON ej.upload_id=u.id WHERE dr.id=%s AND u.user_id=%s', (dedupe_id, current_user.id))
        row = cur.fetchone()
        if not row:
            flash('Not found.', 'warning'); cur.close(); db.close(); return redirect(url_for('history'))
        path = row[0]
        try:
            if path and os.path.exists(path): os.remove(path)
        except Exception:
            pass
        cur.execute('DELETE FROM dedupe_results WHERE id=%s', (dedupe_id,))
        db.commit(); cur.close(); db.close()
        flash('Duplicate file removed.', 'success')
    except Exception as e:
        flash(f'Delete failed: {e}', 'danger')
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True)
