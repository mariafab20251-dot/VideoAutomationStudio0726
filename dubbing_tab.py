#!/usr/bin/env python
"""
Dubbing Tab (UI)
================
A self-contained Tk tab that dubs a chosen video's ORIGINAL dialogue into a
target language.  Deliberately kept OUT of ``complete_automation_gui.py`` so
the main file stays small and the Our Script tab is never touched.

Wiring (two lines in the main file):

    from dubbing_tab import DubbingTabMixin
    class VideoAutomationGUI(DubbingTabMixin, ...):   # add the mixin
        ...
        self.create_dubbing_tab()                     # after the other tabs

The mixin reuses the host GUI's ``self.settings``, ``self.update_setting``,
``self.notebook``, plus ``AppStyles`` / ``ModernButton`` imported here.  It
runs the dub on a worker thread and streams progress into its own log box.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import dubbing_engine


# Populated by create_dubbing_tab() via _styles(); declared here so the bare
# ``AppStyles`` / ``ModernButton`` references in the helper methods resolve as
# module globals rather than raising NameError at import time.
AppStyles = None
ModernButton = None


def _styles():
    """Lazy import of the host GUI's shared UI classes.

    Imported inside functions (not at module top) so that
    ``complete_automation_gui`` can ``from dubbing_tab import DubbingTabMixin``
    at class-definition time without a circular-import deadlock — by the time
    any of these run, the main module is fully loaded.
    """
    from complete_automation_gui import AppStyles, ModernButton
    return AppStyles, ModernButton


# Language menu — label shown to the user is what gets sent to the translator.
DUB_LANGUAGES = [
    'English', 'Urdu', 'Hindi', 'Arabic', 'Spanish', 'French', 'German', 'Italian',
    'Portuguese', 'Indonesian', 'Malay', 'Turkish', 'Russian', 'Persian',
    'Bengali', 'Punjabi', 'Tamil', 'Telugu', 'Japanese', 'Korean',
    'Chinese', 'Vietnamese', 'Thai',
]

# Source language options — "Auto-detect" lets whisper figure it out.
SOURCE_LANGUAGES = ['Auto-detect', 'English', 'Urdu', 'Hindi', 'Arabic',
    'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Indonesian',
    'Malay', 'Turkish', 'Russian', 'Persian', 'Bengali', 'Punjabi', 'Tamil',
    'Telugu', 'Japanese', 'Korean', 'Chinese', 'Vietnamese', 'Thai']


class DubbingTabMixin:
    """Adds a 🎙️ Dubbing tab to the host VideoAutomationGUI."""

    # ── Tab construction ────────────────────────────────────────────────
    def create_dubbing_tab(self):
        global AppStyles, ModernButton
        AppStyles, ModernButton = _styles()
        tab = tk.Frame(self.notebook, bg=AppStyles.BG_CARD)
        self.notebook.add(tab, text='🎙️ Dubbing')

        # Header
        header = tk.Frame(tab, bg=AppStyles.BG_CARD)
        header.pack(fill='x', padx=20, pady=(14, 4))
        tk.Label(header, text='🎙️ Dub Original Dialogue',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_DARK,
                 font=('Segoe UI', 14, 'bold')).pack(anchor='w')
        tk.Label(header,
                 text='Pick any video → its spoken dialogue is transcribed, '
                      'translated, re-voiced with your TTS engine, and muxed '
                      'back over the (ducked) original audio. No Excel or '
                      'script needed.',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 9), justify='left', wraplength=760).pack(
                     anchor='w', pady=(2, 0))

        body = tk.Frame(tab, bg=AppStyles.BG_CARD)
        body.pack(fill='both', expand=True, padx=20, pady=10)

        # ── 1) Source video ────────────────────────────────────────────
        vid_card = self._dub_card(body, '🎬 Source Video')
        row = tk.Frame(vid_card, bg=AppStyles.BG_CARD)
        row.pack(fill='x', padx=8, pady=6)
        self._dub_video_var = tk.StringVar(
            value=self.settings.get('dub_last_video', ''))
        tk.Entry(row, textvariable=self._dub_video_var, bg=AppStyles.BG_INPUT,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9), relief='flat').pack(
                     side='left', fill='x', expand=True, padx=(0, 6), ipady=3)

        def _browse_video():
            f = filedialog.askopenfilename(
                title='Select a video to dub',
                filetypes=[('Video files', '*.mp4 *.mov *.mkv *.avi *.webm'),
                           ('All files', '*.*')])
            if f:
                self._dub_video_var.set(f)
                self.update_setting('dub_last_video', f)
        ModernButton(row, text='📁 Browse', bg_color=AppStyles.ACCENT_INFO,
                     font=('Segoe UI', 9, 'bold'), padx=10, pady=3,
                     command=_browse_video).pack(side='left')

        # ── 1b) Source language ────────────────────────────────────────
        src_card = self._dub_card(body, '🔊 Source Language')
        srow = tk.Frame(src_card, bg=AppStyles.BG_CARD)
        srow.pack(fill='x', padx=8, pady=6)
        tk.Label(srow, text='Video is in:', bg=AppStyles.BG_CARD,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9)).pack(side='left')
        self._dub_src_lang_var = tk.StringVar(
            value=self.settings.get('dub_source_language', 'Auto-detect'))
        src_combo = ttk.Combobox(srow, textvariable=self._dub_src_lang_var,
                                 values=SOURCE_LANGUAGES, width=20)
        src_combo.pack(side='left', padx=(6, 0))
        src_combo.bind('<<ComboboxSelected>>', lambda e: self.update_setting(
            'dub_source_language', self._dub_src_lang_var.get()))
        src_combo.bind('<FocusOut>', lambda e: self.update_setting(
            'dub_source_language', self._dub_src_lang_var.get()))
        tk.Label(srow,
                 text='  "Auto-detect" lets whisper figure it out.',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 8, 'italic')).pack(side='left', padx=(8, 0))

        # ── 3) Target language ─────────────────────────────────────────
        lang_card = self._dub_card(body, '🌐 Target Language')
        lrow = tk.Frame(lang_card, bg=AppStyles.BG_CARD)
        lrow.pack(fill='x', padx=8, pady=6)
        tk.Label(lrow, text='Dub into:', bg=AppStyles.BG_CARD,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9)).pack(side='left')
        self._dub_lang_var = tk.StringVar(
            value=self.settings.get('dub_target_language', 'Urdu'))
        lang_combo = ttk.Combobox(lrow, textvariable=self._dub_lang_var,
                                  values=DUB_LANGUAGES, width=20)
        lang_combo.pack(side='left', padx=(6, 0))
        lang_combo.bind('<<ComboboxSelected>>', lambda e: self.update_setting(
            'dub_target_language', self._dub_lang_var.get()))
        lang_combo.bind('<FocusOut>', lambda e: self.update_setting(
            'dub_target_language', self._dub_lang_var.get()))
        tk.Label(lrow,
                 text='  ⚠ Make sure the 🗣 TTS tab has a voice that speaks '
                      'this language.',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 8, 'italic')).pack(side='left', padx=(8, 0))

        # ── 3) Output + mix controls ───────────────────────────────────
        opt_card = self._dub_card(body, '⚙️ Options')

        orow = tk.Frame(opt_card, bg=AppStyles.BG_CARD)
        orow.pack(fill='x', padx=8, pady=(6, 2))
        tk.Label(orow, text='Output folder:', bg=AppStyles.BG_CARD,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9)).pack(side='left')
        self._dub_out_var = tk.StringVar(
            value=self.settings.get('dub_output_folder', ''))
        tk.Entry(orow, textvariable=self._dub_out_var, bg=AppStyles.BG_INPUT,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9), relief='flat').pack(
                     side='left', fill='x', expand=True, padx=(6, 6), ipady=3)

        def _browse_out():
            d = filedialog.askdirectory(title='Select output folder')
            if d:
                self._dub_out_var.set(d)
                self.update_setting('dub_output_folder', d)
        ModernButton(orow, text='📁', bg_color=AppStyles.ACCENT_INFO,
                     font=('Segoe UI', 9, 'bold'), padx=8, pady=3,
                     command=_browse_out).pack(side='left')
        tk.Label(opt_card,
                 text='   Leave blank to write next to the source video as '
                      '<name>_dubbed_<lang>.mp4',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 8, 'italic')).pack(anchor='w', padx=8)

        # Duck sliders
        mrow = tk.Frame(opt_card, bg=AppStyles.BG_CARD)
        mrow.pack(fill='x', padx=8, pady=(8, 4))
        self._dub_duck_var = tk.DoubleVar(
            value=float(self.settings.get('dub_original_duck', 0.12)))
        self._dub_bg_var = tk.DoubleVar(
            value=float(self.settings.get('dub_original_bg', 0.55)))
        self._dub_slider(mrow, 'Original under dub:', self._dub_duck_var,
                         'dub_original_duck')
        self._dub_slider(mrow, 'Original elsewhere:', self._dub_bg_var,
                         'dub_original_bg')

        # Max dubbing speed — how much a long translated line may be sped up
        # (pitch-preserving) to stay on the video timeline.  1.0 = never speed
        # up (may drift/overlap); higher = tighter sync, more compressed voice.
        srow = tk.Frame(opt_card, bg=AppStyles.BG_CARD)
        srow.pack(fill='x', padx=8, pady=(2, 4))
        self._dub_speed_var = tk.DoubleVar(
            value=float(self.settings.get('dub_max_speed', 1.6)))
        tk.Label(srow, text='Max voice speed-up:', bg=AppStyles.BG_CARD,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 8),
                 width=18, anchor='w').pack(side='left')
        _spd_lbl = tk.Label(srow, text=f'{self._dub_speed_var.get():.2f}×',
                            bg=AppStyles.BG_CARD, fg=AppStyles.ACCENT_PRIMARY,
                            font=('Segoe UI', 8), width=5)
        _spd_lbl.pack(side='right')

        def _on_speed(v):
            fv = float(v)
            _spd_lbl.config(text=f'{fv:.2f}×')
            self.update_setting('dub_max_speed', round(fv, 2))
        ttk.Scale(srow, from_=1.0, to=2.5, variable=self._dub_speed_var,
                  orient='horizontal', command=_on_speed).pack(
                      side='left', fill='x', expand=True, padx=6)
        tk.Label(opt_card,
                 text='   1.0 = natural (may lag behind scene) · higher = '
                      'tighter sync, more compressed. ~1.6 recommended for '
                      'Urdu/Hindi.',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 8, 'italic'), justify='left',
                 wraplength=740).pack(anchor='w', padx=8)

        self._dub_keep_audio_var = tk.BooleanVar(
            value=self.settings.get('dub_keep_audio_file', False))
        tk.Checkbutton(opt_card,
                       text='Also keep the dubbed audio as a separate .mp3',
                       variable=self._dub_keep_audio_var,
                       bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_DARK,
                       activebackground=AppStyles.BG_CARD,
                       selectcolor=AppStyles.BG_INPUT,
                       font=('Segoe UI', 8),
                       command=lambda: self.update_setting(
                           'dub_keep_audio_file',
                           self._dub_keep_audio_var.get())).pack(
                               anchor='w', padx=8, pady=(2, 6))

        # ── 4) Run button + progress ───────────────────────────────────
        run_row = tk.Frame(body, bg=AppStyles.BG_CARD)
        run_row.pack(fill='x', pady=(6, 4))
        self._dub_run_btn = ModernButton(
            run_row, text='▶  Dub Video', bg_color=AppStyles.ACCENT_SUCCESS,
            hover_color='#059669', font=('Segoe UI', 11, 'bold'),
            padx=18, pady=8, command=self._dub_start)
        self._dub_run_btn.pack(side='left')

        self._dub_progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(run_row, mode='determinate',
                        variable=self._dub_progress_var,
                        style='Modern.Horizontal.TProgressbar').pack(
                            side='left', fill='x', expand=True, padx=(12, 0))

        self._dub_status_var = tk.StringVar(value='Ready.')
        tk.Label(body, textvariable=self._dub_status_var,
                 bg=AppStyles.BG_CARD, fg=AppStyles.ACCENT_PRIMARY,
                 font=('Segoe UI', 9)).pack(anchor='w', pady=(0, 4))

        # ── 5) Log box (expands to fill remaining space) ────────────
        log_card = self._dub_card(body, '📋 Log')
        log_card.pack_forget()
        log_card.pack(fill='both', expand=True, pady=6)
        self._dub_log_widget = tk.Text(
            log_card, height=6, wrap='word', bg=AppStyles.BG_INPUT,
            fg=AppStyles.TEXT_DARK, font=('Consolas', 8), relief='flat', bd=4)
        self._dub_log_widget.pack(fill='both', expand=True, padx=6, pady=6)
        self._dub_running = False

    # ── Small UI helpers ────────────────────────────────────────────────
    def _dub_card(self, parent, title):
        card = tk.Frame(parent, bg=AppStyles.BG_CARD,
                        highlightbackground='#30363d', highlightthickness=1)
        card.pack(fill='x', pady=6)
        tk.Label(card, text=title, bg=AppStyles.BG_CARD,
                 fg=AppStyles.ACCENT_PRIMARY,
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', padx=8, pady=(6, 0))
        return card

    def _dub_slider(self, parent, label, var, key):
        row = tk.Frame(parent, bg=AppStyles.BG_CARD)
        row.pack(fill='x', pady=1)
        tk.Label(row, text=label, bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_DARK,
                 font=('Segoe UI', 8), width=18, anchor='w').pack(side='left')
        val_lbl = tk.Label(row, text=f'{var.get():.0%}', bg=AppStyles.BG_CARD,
                           fg=AppStyles.ACCENT_PRIMARY, font=('Segoe UI', 8),
                           width=5)
        val_lbl.pack(side='right')

        def _on_move(v):
            fv = float(v)
            val_lbl.config(text=f'{fv:.0%}')
            self.update_setting(key, round(fv, 3))
        ttk.Scale(row, from_=0.0, to=1.0, variable=var, orient='horizontal',
                  command=_on_move).pack(side='left', fill='x', expand=True,
                                         padx=6)

    # ── Logging (thread-safe via after) ─────────────────────────────────
    def _dub_log(self, level, msg):
        icons = {'ok': '✅', 'error': '❌', 'warn': '⚠', 'info': 'ℹ',
                 'path': '📁', 'header': '━'}
        line = f"{icons.get(level, '·')} {msg}\n"

        def _append():
            try:
                self._dub_log_widget.insert('end', line)
                self._dub_log_widget.see('end')
            except Exception:
                pass
        try:
            self._dub_log_widget.after(0, _append)
        except Exception:
            print(line, end='')

    def _dub_set_status(self, text):
        try:
            self._dub_status_var.set(text)
        except Exception:
            pass

    def _dub_set_progress(self, done, total, note=''):
        try:
            pct = (done / total * 100) if total else 0
            self._dub_progress_var.set(pct)
            if note:
                self._dub_set_status(note)
        except Exception:
            pass

    # ── Run handler ─────────────────────────────────────────────────────
    def _dub_start(self):
        if getattr(self, '_dub_running', False):
            self._dub_log('warn', 'A dub is already running — please wait.')
            return

        video = (self._dub_video_var.get() or '').strip()
        lang = (self._dub_lang_var.get() or '').strip()
        src_lang = (self._dub_src_lang_var.get() or '').strip()
        if not video or not Path(video).is_file():
            messagebox.showerror('Dubbing', 'Please pick a valid video file.')
            return
        if not lang:
            messagebox.showerror('Dubbing', 'Please pick a target language.')
            return

        # Resolve output path
        out_folder = (self._dub_out_var.get() or '').strip()
        src = Path(video)
        safe_lang = lang.lower().replace(' ', '_')
        out_name = f'{src.stem}_dubbed_{safe_lang}.mp4'
        out_video = (Path(out_folder) / out_name) if out_folder else \
            src.with_name(out_name)

        # Persist selections
        self.update_setting('dub_last_video', video)
        self.update_setting('dub_target_language', lang)
        self.update_setting('dub_source_language', src_lang)

        self._dub_running = True
        self._dub_run_btn.config(state='disabled')
        self._dub_progress_var.set(0)
        self._dub_log('header', f'Dub started: {src.name} → {lang}')
        self._dub_set_status('Starting…')

        t = threading.Thread(
            target=self._dub_worker,
            args=(src, out_video, lang, src_lang), daemon=True)
        t.start()

    def _dub_worker(self, src: Path, out_video: Path, lang: str,
                    src_lang: str = 'Auto-detect'):
        try:
            result = dubbing_engine.dub_video(
                src, out_video, lang, self.settings,
                log=self._dub_log,
                progress=self._dub_set_progress,
                keep_audio_file=bool(self._dub_keep_audio_var.get()),
                source_language=src_lang if src_lang != 'Auto-detect' else None)
            if result is not None:
                self._dub_log('ok', f'Done → {result}')
                self._dub_set_status(f'✅ Done: {Path(result).name}')
                self._dub_video_widget_after(
                    lambda: messagebox.showinfo(
                        'Dubbing complete',
                        f'Dubbed video written to:\n{result}'))
            else:
                self._dub_log('error', 'Dub failed — see log above.')
                self._dub_set_status('❌ Failed. See log.')
        except Exception as e:
            self._dub_log('error', f'Unexpected error: {e}')
            for ln in traceback.format_exc().splitlines():
                self._dub_log('error', f'  {ln}')
            self._dub_set_status('❌ Error. See log.')
        finally:
            self._dub_running = False
            self._dub_video_widget_after(
                lambda: self._dub_run_btn.config(state='normal'))

    def _dub_video_widget_after(self, fn):
        """Run *fn* on the Tk main thread."""
        try:
            self._dub_log_widget.after(0, fn)
        except Exception:
            try:
                fn()
            except Exception:
                pass