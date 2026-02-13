from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, UploadFile, File, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from sqlmodel import Session, select
import subprocess
import os
import shutil
from datetime import datetime
from dotenv import load_dotenv
from models import create_db_and_tables, engine, User, Announcement, QuizResult

# Load environment variables
load_dotenv()

app = FastAPI()

# Database initialization
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# Security & Middleware
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# OAuth Setup
oauth = OAuth()
oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

# Helper: Get Current User
def get_current_user(request: Request):
    user = request.session.get('user')
    if not user:
        return None
    return user

async def require_auth(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

async def require_admin(request: Request):
    user = await require_auth(request)
    with Session(engine) as session:
        db_user = session.exec(select(User).where(User.email == user['email'])).first()
        if not db_user or db_user.role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
    return db_user

# Mount static directories for reports
# We will have output_mistral and output_groq
Path("output_mistral").mkdir(exist_ok=True)
Path("output_groq").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True) # For backward compatibility

app.mount("/output_mistral", StaticFiles(directory="output_mistral"), name="output_mistral")
app.mount("/output_groq", StaticFiles(directory="output_groq"), name="output_groq")
app.mount("/output", StaticFiles(directory="output"), name="output")

# Setup templates
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    user = get_current_user(request)
    if not user:
        return templates.TemplateResponse("login.html", {"request": request})
    
    with Session(engine) as session:
        # Get announcements
        announcements = session.exec(select(Announcement).order_by(Announcement.created_at.desc())).all()
        # Check if current user is admin
        db_user = session.exec(select(User).where(User.email == user['email'])).first()
        is_admin = db_user.role == "admin" if db_user else False

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "user": user, 
        "is_admin": is_admin,
        "announcements": announcements
    })

# Auth Routes
@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for('auth_callback')
    return await oauth.google.authorize_redirect(request, str(redirect_uri))

@app.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get('userinfo')
    if user_info:
        request.session['user'] = user_info
        
        # Save or update user in DB
        with Session(engine) as session:
            db_user = session.exec(select(User).where(User.email == user_info['email'])).first()
            if not db_user:
                # Check for ADMIN_EMAIL from env
                admin_email = os.getenv("ADMIN_EMAIL")
                user_count = session.exec(select(User)).all()
                
                if admin_email and user_info['email'] == admin_email:
                    role = "admin"
                elif not admin_email and len(user_count) == 0:
                    role = "admin"
                else:
                    role = "user"
                    
                db_user = User(
                    email=user_info['email'],
                    full_name=user_info.get('name'),
                    picture=user_info.get('picture'),
                    role=role
                )
                session.add(db_user)
            else:
                db_user.last_login = datetime.utcnow()
                session.add(db_user)
            session.commit()
            
    return RedirectResponse(url='/')

@app.get("/logout")
async def logout(request: Request):
    request.session.pop('user', None)
    return RedirectResponse(url='/')

@app.get("/api/videos")
async def get_videos(user: dict = Depends(require_auth)):
    video_dir = Path("Video")
    if not video_dir.exists():
        return []
    exts = {".mp4", ".mkv", ".mov", ".avi"}
    videos = [f.name for f in video_dir.iterdir() if f.suffix.lower() in exts]
    return videos

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...), admin: User = Depends(require_admin)):
    video_dir = Path("Video")
    video_dir.mkdir(exist_ok=True)
    
    # Save file
    file_path = video_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"filename": file.filename, "message": "上傳成功"}

@app.get("/api/reports")
async def get_reports(user: dict = Depends(require_auth)):
    reports = []
    # Helper to scan dir
    def scan(dir_name, model_display_name):
        d = Path(dir_name)
        if not d.exists(): return
        for f in d.glob("*_report_v2.html"):
            # File format: [stem]_report_v2.html
            video_stem = f.name.replace("_report_v2.html", "")
            
            # Find corresponding quiz file
            # Quiz format: [stem]_quiz.html
            quiz_file = d / f"{video_stem}_quiz.html"
            
            reports.append({
                "video_name": video_stem,
                "model": model_display_name,
                "model_key": dir_name, # "output_mistral", "output_groq", etc.
                "filename": f.name,
                "report_url": f"/{dir_name}/{f.name}",
                "quiz_url": f"/{dir_name}/{quiz_file.name}" if quiz_file.exists() else None,
                "timestamp": f.stat().st_mtime,
                "status": "completed"
            })
        
        # Scan for processing files
        for f in d.glob("*.processing"):
            video_stem = f.stem # filename is [stem].processing
            reports.append({
                "video_name": video_stem,
                "model": model_display_name,
                "model_key": dir_name,
                "filename": f.name,
                "report_url": "#",
                "quiz_url": None,
                "timestamp": f.stat().st_mtime,
                "status": "processing"
            })
    
    scan("output", "Mistral (Default)")
    scan("output_mistral", "Mistral")
    scan("output_groq", "Groq")
    
    # Sort by timestamp desc
    reports.sort(key=lambda x: x["timestamp"], reverse=True)
    return reports

# Admin: Delete Report
@app.delete("/api/reports/{model_key}/{video_stem}")
async def delete_report(model_key: str, video_stem: str, admin: User = Depends(require_admin)):
    if model_key not in ["output", "output_mistral", "output_groq"]:
        raise HTTPException(status_code=400, detail="Invalid model key")
    
    d = Path(model_key)
    # Files to delete
    patterns = [
        f"{video_stem}_report_v2.html",
        f"{video_stem}_quiz.html",
        f"{video_stem}_quiz.json",
        f"{video_stem}_transcription.json"
    ]
    
    for p in patterns:
        f = d / p
        if f.exists():
            f.unlink()
            
    # Screenshots
    screenshot_dir = d / "screenshots" / video_stem.replace(' ', '_')
    if screenshot_dir.exists() and screenshot_dir.is_dir():
        shutil.rmtree(screenshot_dir)
        
    return {"message": "教材已刪除"}

# Announcements API
@app.post("/api/announcements")
async def create_announcement(data: dict, admin: User = Depends(require_admin)):
    content = data.get("content")
    if not content:
        raise HTTPException(status_code=400, detail="Content required")
    
    with Session(engine) as session:
        ann = Announcement(content=content, created_by=admin.email)
        session.add(ann)
        session.commit()
    return {"message": "公告已發布"}

@app.get("/api/announcements")
async def get_announcements(user: dict = Depends(require_auth)):
    with Session(engine) as session:
        anns = session.exec(select(Announcement).order_by(Announcement.created_at.desc())).all()
        return anns

@app.delete("/api/announcements/{id}")
async def delete_announcement(id: int, admin: User = Depends(require_admin)):
    with Session(engine) as session:
        ann = session.get(Announcement, id)
        if ann:
            session.delete(ann)
            session.commit()
    return {"message": "公告已刪除"}

# Quiz Results API
@app.post("/api/quiz/submit")
async def submit_quiz(data: dict, user: dict = Depends(require_auth)):
    with Session(engine) as session:
        result = QuizResult(
            user_email=user['email'],
            video_name=data.get("video_name"),
            score=data.get("score"),
            total_questions=data.get("total_questions")
        )
        session.add(result)
        session.commit()
    print(f"DEBUG: 成績已成功紀錄 - User: {user['email']}, Video: {data.get('video_name')}, Score: {data.get('score')}")
    return {"message": "成績已紀錄"}

@app.get("/api/quiz-results")
async def get_quiz_results(admin: User = Depends(require_admin)):
    with Session(engine) as session:
        results = session.exec(select(QuizResult).order_by(QuizResult.created_at.desc())).all()
        return results

def run_script(script_name, video_path, output_dir_str):
    output_dir = Path(output_dir_str)
    processing_file = output_dir / f"{video_path.stem}.processing"
    
    # Mark as processing
    with open(processing_file, "w") as f:
        f.write("processing")

    try:
        # Using 'uv run python ...'
        cmd = ["uv", "run", "python", script_name, "--video", str(video_path), "--output", output_dir_str]
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd)
    finally:
        # Cleanup
        if processing_file.exists():
            processing_file.unlink()
@app.post("/api/process")
async def process_video(request: Request, background_tasks: BackgroundTasks, admin: User = Depends(require_admin)):
    data = await request.json()
    video_name = data.get("video_name")
    model = data.get("model") # 'mistral' or 'groq'
    
    if not video_name or not model:
        raise HTTPException(status_code=400, detail="Missing video_name or model")
    
    video_path = Path("Video") / video_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
        
    if model == "mistral":
        script = "transcribe_video.py"
        target_dir = "output_mistral"
    elif model == "groq":
        script = "transcribe_video_groq.py"
        target_dir = "output_groq"
    else:
        raise HTTPException(status_code=400, detail="Invalid model")
    
    # Check permission placeholder
    # user = request.state.user... (Future implementation)
    
    background_tasks.add_task(run_script, script, video_path, target_dir)
    return {"status": "processing", "message": f"此排程已送出：{video_name} (Model: {model})，請稍待片刻重新整理列表。"}
