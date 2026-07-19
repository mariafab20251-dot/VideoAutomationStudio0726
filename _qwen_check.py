try:
 import qwen_tts
 print("qwen_tts OK:", getattr(qwen_tts, "__version__", "installed"))
 print("torch:", __import__("torch").__version__)
except ImportError as e:
 print("NOT INSTALLED:", e)