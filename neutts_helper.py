"""
NeuTTS Integration Helper Module
Provides voice cloning and TTS generation using NeuTTS Gradio API
Updated to work with Gradio-based NeuTTS Voice Cloning app
"""

import requests
import json
import time
import base64
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import threading
import queue


class NeuTTSHelper:
    """Helper class for NeuTTS voice cloning and speech generation (Gradio compatible)"""

    def __init__(self, server_url: str = "http://localhost:7860"):
        """
        Initialize NeuTTS helper

        Args:
            server_url: Base URL of NeuTTS Gradio server (default: http://localhost:7860)
        """
        self.server_url = server_url.rstrip('/')
        self.voices_library = {}  # Store cloned voices
        self.status = "disconnected"

    def check_server_status(self) -> Tuple[bool, str]:
        """
        Check if NeuTTS Gradio server is running

        Returns:
            Tuple of (is_running: bool, status_message: str)
        """
        try:
            # Try Gradio's info endpoint
            response = requests.get(f"{self.server_url}/info", timeout=5)
            if response.status_code == 200:
                self.status = "connected"
                return True, "✓ NeuTTS Server Connected"

            # Fallback: try the main page
            response = requests.get(f"{self.server_url}/", timeout=5)
            if response.status_code == 200:
                self.status = "connected"
                return True, "✓ NeuTTS Server Connected"
            else:
                self.status = "error"
                return False, f"⚠ Server Error: {response.status_code}"

        except requests.ConnectionError:
            self.status = "disconnected"
            return False, "✗ Server Not Running - Start NeuTTS server"
        except requests.Timeout:
            self.status = "timeout"
            return False, "⚠ Server Timeout"
        except Exception as e:
            self.status = "error"
            return False, f"✗ Error: {str(e)}"

    def _get_gradio_api_info(self) -> Optional[dict]:
        """Get Gradio API endpoint information"""
        try:
            response = requests.get(f"{self.server_url}/info", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None

    def clone_voice(self,
                   voice_name: str,
                   audio_file_path: str,
                   reference_text: str,
                   language: str = "en") -> Tuple[bool, str]:
        """
        Clone a voice from audio sample using Gradio API

        Args:
            voice_name: Name to identify this cloned voice
            audio_file_path: Path to audio sample file (WAV/MP3)
            reference_text: Text that matches the audio sample
            language: Language code (en, es, fr, etc.)

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Check if file exists
            audio_path = Path(audio_file_path)
            if not audio_path.exists():
                return False, f"Audio file not found: {audio_file_path}"

            # Read audio file and encode as base64 for Gradio
            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()

            audio_base64 = base64.b64encode(audio_data).decode('utf-8')

            # Determine file type
            file_ext = audio_path.suffix.lower()
            if file_ext == '.wav':
                mime_type = 'audio/wav'
            elif file_ext == '.mp3':
                mime_type = 'audio/mpeg'
            else:
                mime_type = 'audio/wav'

            # Prepare Gradio API request
            # Format for file upload in Gradio
            audio_payload = {
                "data": f"data:{mime_type};base64,{audio_base64}",
                "name": audio_path.name
            }

            # Try different API endpoints that Gradio might use
            endpoints_to_try = [
                "/api/predict",
                "/run/predict",
                "/api/clone_voice",
                "/run/clone_voice"
            ]

            for endpoint in endpoints_to_try:
                try:
                    # Gradio API format
                    payload = {
                        "data": [
                            voice_name,           # Voice name
                            reference_text,       # Reference text
                            audio_payload         # Audio file
                        ]
                    }

                    response = requests.post(
                        f"{self.server_url}{endpoint}",
                        json=payload,
                        timeout=120
                    )

                    if response.status_code == 200:
                        result = response.json()

                        # Store in library
                        self.voices_library[voice_name] = {
                            'voice_id': voice_name,
                            'language': language,
                            'reference_text': reference_text,
                            'audio_file': audio_file_path,
                            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
                        }

                        return True, f"✓ Voice '{voice_name}' cloned successfully!"

                except requests.exceptions.RequestException:
                    continue

            # If direct API doesn't work, just save to library for manual use
            # The user has already cloned in the Gradio UI
            self.voices_library[voice_name] = {
                'voice_id': voice_name,
                'language': language,
                'reference_text': reference_text,
                'audio_file': audio_file_path,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            return True, f"✓ Voice '{voice_name}' saved to library"

        except Exception as e:
            return False, f"✗ Exception: {str(e)}"

    def generate_speech(self,
                       text: str,
                       voice_name: str,
                       output_path: str,
                       speed: float = 1.0,
                       pitch: float = 1.0) -> Tuple[bool, str]:
        """
        Generate speech from text using cloned voice via Gradio API
        Automatically handles multi-sentence text by processing each sentence separately

        Args:
            text: Text to convert to speech
            voice_name: Name of cloned voice to use
            output_path: Where to save the audio file
            speed: Speech speed multiplier (0.5 - 2.0)
            pitch: Pitch adjustment (0.5 - 2.0)

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Debug: Log the input text
            print(f"\n[NeuTTS DEBUG] Input text length: {len(text)} chars")
            print(f"[NeuTTS DEBUG] Input text: {text[:100]}..." if len(text) > 100 else f"[NeuTTS DEBUG] Input text: {text}")
            print(f"[NeuTTS DEBUG] Voice: {voice_name}, Speed: {speed}, Pitch: {pitch}")

            # Check if voice exists in library
            if voice_name not in self.voices_library:
                error_msg = f"Voice '{voice_name}' not found in library"
                print(f"[NeuTTS DEBUG] ERROR: {error_msg}")
                print(f"[NeuTTS DEBUG] Available voices: {list(self.voices_library.keys())}")
                return False, error_msg

            # Split text into sentences to handle NeuTTS server limitation
            # NeuTTS server appears to only process first sentence, so we split and concatenate
            sentences = self._split_into_sentences(text)
            print(f"[NeuTTS DEBUG] Split into {len(sentences)} sentence(s)")

            # If only one sentence, process normally
            if len(sentences) <= 1:
                return self._generate_single_speech(text, voice_name, output_path, speed, pitch)

            # Multiple sentences - process each and concatenate
            import os
            import tempfile
            import subprocess

            temp_dir = tempfile.mkdtemp()
            temp_files = []

            try:
                # Generate audio for each sentence
                for i, sentence in enumerate(sentences):
                    if not sentence.strip():
                        continue

                    temp_file = os.path.join(temp_dir, f"sentence_{i:03d}.wav")
                    print(f"[NeuTTS DEBUG] Processing sentence {i+1}/{len(sentences)}: {sentence[:50]}...")

                    success, msg = self._generate_single_speech(
                        sentence.strip(),
                        voice_name,
                        temp_file,
                        speed,
                        pitch
                    )

                    if not success:
                        # Cleanup and return error
                        for f in temp_files:
                            if os.path.exists(f):
                                os.remove(f)
                        os.rmdir(temp_dir)
                        return False, f"Failed on sentence {i+1}: {msg}"

                    temp_files.append(temp_file)

                # Concatenate all audio files using ffmpeg
                if len(temp_files) == 0:
                    return False, "No audio generated"

                if len(temp_files) == 1:
                    # Only one file, just copy it
                    import shutil
                    shutil.copy(temp_files[0], output_path)
                else:
                    # Create concat file list for ffmpeg
                    concat_file = os.path.join(temp_dir, "concat_list.txt")
                    with open(concat_file, 'w') as f:
                        for temp_file in temp_files:
                            f.write(f"file '{temp_file}'\n")

                    # Concatenate using ffmpeg
                    print(f"[NeuTTS DEBUG] Concatenating {len(temp_files)} audio files")
                    result = subprocess.run([
                        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                        '-i', concat_file, '-c', 'copy', output_path
                    ], capture_output=True, text=True)

                    if result.returncode != 0:
                        print(f"[NeuTTS DEBUG] FFmpeg concat error: {result.stderr}")
                        return False, f"Failed to concatenate audio: {result.stderr}"

                # Cleanup temp files
                for f in temp_files:
                    if os.path.exists(f):
                        os.remove(f)
                if os.path.exists(concat_file):
                    os.remove(concat_file)
                os.rmdir(temp_dir)

                file_size = os.path.getsize(output_path)
                print(f"[NeuTTS DEBUG] Final concatenated audio: {output_path}, Size: {file_size} bytes")
                return True, f"✓ Speech generated from {len(sentences)} sentences: {output_path}"

            except Exception as e:
                # Cleanup on error
                for f in temp_files:
                    if os.path.exists(f):
                        os.remove(f)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
                raise e

        except Exception as e:
            return False, f"✗ Exception: {str(e)}"

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for processing
        Handles common sentence endings: . ! ?
        """
        import re
        # Split on sentence boundaries but keep the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _generate_single_speech(self,
                                text: str,
                                voice_name: str,
                                output_path: str,
                                speed: float = 1.0,
                                pitch: float = 1.0) -> Tuple[bool, str]:
        """
        Generate speech for a single sentence/chunk using Gradio API
        Internal method called by generate_speech

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            import os
            import shutil

            # Try using gradio_client first (most reliable)
            try:
                from gradio_client import Client
                client = Client(self.server_url)
                print(f"[NeuTTS DEBUG] Using gradio_client to call /generate_speech")
                result = client.predict(
                    text=text,
                    voice_name=voice_name,
                    speed=speed,
                    api_name="/generate_speech"
                )
                print(f"[NeuTTS DEBUG] Result type: {type(result)}, Result: {str(result)[:200]}")

                # Handle different result types
                audio_file = None

                # Result could be string, tuple, or list
                if isinstance(result, str):
                    audio_file = result
                elif isinstance(result, (tuple, list)) and len(result) > 0:
                    # Find the first string that looks like a file path
                    for item in result:
                        if isinstance(item, str) and (item.endswith('.wav') or item.endswith('.mp3') or '\\' in item or '/' in item):
                            audio_file = item
                            break
                elif isinstance(result, dict) and 'path' in result:
                    audio_file = result['path']

                if audio_file and isinstance(audio_file, str):
                    # Copy the generated audio to output path
                    shutil.copy(audio_file, output_path)
                    # Check file size
                    file_size = os.path.getsize(output_path)
                    print(f"[NeuTTS DEBUG] Audio file saved: {output_path}, Size: {file_size} bytes")
                    return True, f"✓ Speech generated: {output_path}"
                else:
                    print(f"[NeuTTS DEBUG] Unexpected result type from gradio_client: {type(result)} = {result}")

            except ImportError:
                # gradio_client not installed, try requests
                print(f"[NeuTTS DEBUG] gradio_client not installed, falling back to requests")
                pass
            except Exception as e:
                # Log but try fallback
                print(f"[NeuTTS DEBUG] gradio_client error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                print(f"[NeuTTS DEBUG] Trying fallback methods...")

            # Fallback: Use requests with the correct endpoint
            # Format: /call/generate_speech or /api/predict with api_name
            endpoints_to_try = [
                # New Gradio 4.x format
                ("/call/generate_speech", None),
                # Gradio API predict with api_name
                ("/api/predict", "/generate_speech"),
                # Direct run endpoint
                ("/run/generate_speech", None),
            ]

            for endpoint, api_name in endpoints_to_try:
                try:
                    # Prepare payload based on endpoint type
                    if api_name:
                        payload = {
                            "data": [text, voice_name, speed],
                            "api_name": api_name
                        }
                    else:
                        payload = {
                            "data": [text, voice_name, speed]
                        }

                    print(f"[NeuTTS DEBUG] Trying endpoint: {self.server_url}{endpoint}")
                    print(f"[NeuTTS DEBUG] Payload data: text_len={len(text)}, voice={voice_name}, speed={speed}")

                    response = requests.post(
                        f"{self.server_url}{endpoint}",
                        json=payload,
                        timeout=120
                    )

                    print(f"[NeuTTS DEBUG] Response status: {response.status_code}")

                    if response.status_code == 200:
                        result = response.json()

                        # Extract audio from response
                        if 'data' in result and len(result['data']) > 0:
                            audio_data = result['data'][0]

                            # Handle different response formats
                            if isinstance(audio_data, dict) and 'data' in audio_data:
                                # Base64 encoded audio
                                audio_b64 = audio_data['data'].split(',')[1] if ',' in audio_data['data'] else audio_data['data']
                                audio_bytes = base64.b64decode(audio_b64)
                            elif isinstance(audio_data, str) and audio_data.startswith('data:'):
                                # Data URL format
                                audio_b64 = audio_data.split(',')[1]
                                audio_bytes = base64.b64decode(audio_b64)
                            elif isinstance(audio_data, str):
                                # File path returned - download it
                                file_url = f"{self.server_url}/file={audio_data}"
                                file_response = requests.get(file_url, timeout=30)
                                if file_response.status_code == 200:
                                    audio_bytes = file_response.content
                                else:
                                    continue
                            else:
                                continue

                            # Save audio to file
                            with open(output_path, 'wb') as f:
                                f.write(audio_bytes)

                            file_size = os.path.getsize(output_path)
                            print(f"[NeuTTS DEBUG] Audio saved via requests: {output_path}, Size: {file_size} bytes")
                            return True, f"✓ Speech generated: {output_path}"

                except requests.exceptions.RequestException as e:
                    print(f"[NeuTTS DEBUG] Request failed for {endpoint}: {type(e).__name__}: {e}")
                    continue
                except Exception as e:
                    print(f"[NeuTTS DEBUG] Unexpected error for {endpoint}: {type(e).__name__}: {e}")
                    continue

            print(f"[NeuTTS DEBUG] All endpoints failed - no successful response")
            return False, "✗ Could not generate speech - all API endpoints failed. Check server logs."

        except Exception as e:
            return False, f"✗ Exception: {str(e)}"

    def generate_speech_chunked(self,
                               text: str,
                               voice_name: str,
                               output_folder: str,
                               max_chunk_length: int = 500,
                               speed: float = 1.0,
                               pitch: float = 1.0) -> Tuple[bool, List[str], str]:
        """
        Generate speech for long text by chunking into smaller parts

        Args:
            text: Long text to convert
            voice_name: Name of cloned voice
            output_folder: Folder to save audio chunks
            max_chunk_length: Maximum characters per chunk
            speed: Speech speed multiplier
            pitch: Pitch adjustment

        Returns:
            Tuple of (success: bool, audio_files: List[str], message: str)
        """
        try:
            # Create output folder
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)

            # Split text into chunks (by sentences)
            chunks = self._chunk_text(text, max_chunk_length)

            audio_files = []

            for i, chunk in enumerate(chunks):
                chunk_file = output_path / f"chunk_{i+1:03d}.wav"

                success, msg = self.generate_speech(
                    text=chunk,
                    voice_name=voice_name,
                    output_path=str(chunk_file),
                    speed=speed,
                    pitch=pitch
                )

                if not success:
                    return False, audio_files, f"Failed at chunk {i+1}: {msg}"

                audio_files.append(str(chunk_file))

            return True, audio_files, f"✓ Generated {len(chunks)} audio chunks"

        except Exception as e:
            return False, [], f"✗ Exception: {str(e)}"

    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """
        Split text into chunks at sentence boundaries

        Args:
            text: Text to split
            max_length: Maximum chunk length

        Returns:
            List of text chunks
        """
        # Split by sentences
        sentences = text.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence exceeds limit, save current chunk
            if len(current_chunk) + len(sentence) > max_length and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def get_available_voices(self) -> Dict[str, dict]:
        """
        Get all cloned voices in library

        Returns:
            Dictionary of voice_name -> voice_info
        """
        return self.voices_library.copy()

    def save_voice_library(self, filepath: str = "neutts_voices.json"):
        """
        Save voice library to JSON file

        Args:
            filepath: Path to save library
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.voices_library, f, indent=2)
            return True, f"✓ Library saved to {filepath}"
        except Exception as e:
            return False, f"✗ Save failed: {str(e)}"

    def load_voice_library(self, filepath: str = "neutts_voices.json"):
        """
        Load voice library from JSON file

        Args:
            filepath: Path to library file
        """
        try:
            if Path(filepath).exists():
                with open(filepath, 'r') as f:
                    self.voices_library = json.load(f)
                return True, f"✓ Loaded {len(self.voices_library)} voices"
            else:
                return False, f"✗ Library file not found: {filepath}"
        except Exception as e:
            return False, f"✗ Load failed: {str(e)}"

    def add_voice_to_library(self, voice_name: str, voice_info: dict = None):
        """
        Manually add a voice to the library (for voices cloned in Gradio UI)

        Args:
            voice_name: Name of the cloned voice
            voice_info: Optional additional info about the voice
        """
        if voice_info is None:
            voice_info = {}

        self.voices_library[voice_name] = {
            'voice_id': voice_name,
            'language': voice_info.get('language', 'en'),
            'reference_text': voice_info.get('reference_text', ''),
            'audio_file': voice_info.get('audio_file', ''),
            'created_at': voice_info.get('created_at', time.strftime('%Y-%m-%d %H:%M:%S'))
        }
        return True, f"✓ Voice '{voice_name}' added to library"

    def delete_voice(self, voice_name: str) -> Tuple[bool, str]:
        """
        Remove a voice from library

        Args:
            voice_name: Name of voice to delete

        Returns:
            Tuple of (success: bool, message: str)
        """
        if voice_name in self.voices_library:
            del self.voices_library[voice_name]
            return True, f"✓ Voice '{voice_name}' deleted"
        else:
            return False, f"✗ Voice '{voice_name}' not found"

    def test_voice(self, voice_name: str, test_text: str = None) -> Tuple[bool, str, str]:
        """
        Generate a test audio with the voice

        Args:
            voice_name: Name of voice to test
            test_text: Optional custom test text

        Returns:
            Tuple of (success: bool, audio_path: str, message: str)
        """
        if not test_text:
            test_text = "Hello! This is a test of my cloned voice. How does it sound?"

        # Generate in temp folder
        temp_folder = Path("temp_neutts_tests")
        temp_folder.mkdir(exist_ok=True)

        output_file = temp_folder / f"test_{voice_name}_{int(time.time())}.wav"

        success, msg = self.generate_speech(
            text=test_text,
            voice_name=voice_name,
            output_path=str(output_file)
        )

        if success:
            return True, str(output_file), msg
        else:
            return False, "", msg


# Async wrapper for GUI integration
class AsyncNeuTTSHelper:
    """Thread-safe async wrapper for NeuTTS operations"""

    def __init__(self, server_url: str = "http://localhost:7860"):
        self.helper = NeuTTSHelper(server_url)
        self.result_queue = queue.Queue()

    def check_status_async(self, callback):
        """Check server status in background thread"""
        def worker():
            result = self.helper.check_server_status()
            callback(result)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def clone_voice_async(self, voice_name, audio_file, ref_text, language, callback):
        """Clone voice in background thread"""
        def worker():
            result = self.helper.clone_voice(voice_name, audio_file, ref_text, language)
            callback(result)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def generate_speech_async(self, text, voice_name, output_path, speed, pitch, callback):
        """Generate speech in background thread"""
        def worker():
            result = self.helper.generate_speech(text, voice_name, output_path, speed, pitch)
            callback(result)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def test_voice_async(self, voice_name, test_text, callback):
        """Test voice in background thread"""
        def worker():
            result = self.helper.test_voice(voice_name, test_text)
            callback(result)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()


if __name__ == "__main__":
    # Test the helper
    helper = NeuTTSHelper("http://localhost:7860")

    print("Checking NeuTTS server status...")
    is_running, status = helper.check_server_status()
    print(f"Status: {status}")

    if is_running:
        print("\nServer is ready!")
        print("\nTo use a voice you cloned in the Gradio UI:")
        print("1. Add it to library: helper.add_voice_to_library('jordanpeterson')")
        print("2. Save library: helper.save_voice_library()")
        print("3. Generate speech: helper.generate_speech('Hello', 'jordanpeterson', 'output.wav')")
    else:
        print("\nPlease start NeuTTS server")
