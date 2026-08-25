import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'smart-farming-super-secret-2026'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///smartfarm.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Gemini API
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') or 'AIzaSyAQ8H_llYsp4qsUQi54XMGq0ugUHZS-Lzk'
    
    # OpenWeather
    OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY') or 'd57850f573f1a3aca0d8650f43fa844f'
    
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
