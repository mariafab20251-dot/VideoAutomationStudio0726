"""
Configuration file for Video Quote Automation
Defines default paths for the application
"""

from pathlib import Path


class Config:
    """Configuration object with path attributes"""
    # System fonts folder (Windows default)
    SYSTEM_FONTS_FOLDER = Path(r"C:\Windows\Fonts")

    # Default video folder
    VIDEO_FOLDER = Path(r"E:\MyAutomations\ScriptAutomations\VideoFolder")

    # Default quotes file
    QUOTES_FILE = Path(r"E:\MyAutomations\ScriptAutomations\VideoFolder\Quotes.txt")

    # Default output folder
    OUTPUT_FOLDER = Path(r"E:\MyAutomations\ScriptAutomations\VideoFolder\FinalVideos")


# Create config instance
config = Config()
