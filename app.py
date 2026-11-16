from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime
import hashlib
import os

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# --- CẤU HÌNH DATABASE ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./prediction_history.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)

class PredictionRecord(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    algorithm = Column(String, default="RBF")
    sepal_length = Column(Float)
    sepal_width = Column(Float)
    petal_length = Column(Float)
    petal_width = Column(Float)
    prediction = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- KHỞI TẠO APP VÀ GẮN ẢNH (Đường dẫn tuyệt đối) ---
app = FastAPI(title="Dự án 1 - Nguyễn Khoa Thông Minh")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Lệnh này ép máy chủ mở đúng thư mục static trong dự án
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- NẠP MÔ HÌNH SVM ---
try:
    model = joblib.load('svm_model.pkl')
except:
    model = None
    
iris_target_names = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']

# --- MODELS ĐẦU VÀO ---
class UserAuth(BaseModel):
    username: str
    password: str

class IrisFeatures(BaseModel):
    user_id: int
    algorithm: str = "RBF"
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

# --- API ENDPOINTS ---
@app.get("/")
def home():
    if not os.path.exists(os.path.join(BASE_DIR, "index.html")):
        return {"error": "Không tìm thấy file index.html"}
    with open(os.path.join(BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/register")
def register(user: UserAuth, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Tài khoản đã tồn tại")
    new_user = User(username=user.username, password_hash=hash_password(user.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"status": "success", "message": "Đăng ký thành công", "user_id": new_user.id}

@app.post("/login")
def login(user: UserAuth, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or db_user.password_hash != hash_password(user.password):
        raise HTTPException(status_code=401, detail="Sai tài khoản hoặc mật khẩu")
    return {"status": "success", "message": "Đăng nhập thành công", "user_id": db_user.id}

@app.post("/predict")
def predict(features: IrisFeatures, db: Session = Depends(get_db)):
    if not model:
        raise HTTPException(status_code=500, detail="Chưa có file mô hình svm_model.pkl")
    user = db.query(User).filter(User.id == features.user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="Vui lòng đăng nhập để dự đoán")
    
    input_data = np.array([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])
    pred = model.predict(input_data)[0]
    predicted_class = iris_target_names[pred] if isinstance(pred, (int, np.integer)) else str(pred)
    
    new_record = PredictionRecord(
        user_id=features.user_id,
        algorithm=features.algorithm,
        sepal_length=features.sepal_length,
        sepal_width=features.sepal_width,
        petal_length=features.petal_length,
        petal_width=features.petal_width,
        prediction=predicted_class
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    return {"status": "success", "user": user.username, "kernel_used": features.algorithm, "prediction": predicted_class, "record_id": new_record.id}

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    records = db.query(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(10).all()
    return {"history": records}