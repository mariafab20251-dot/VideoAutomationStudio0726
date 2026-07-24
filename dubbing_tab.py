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

import os
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
                 font=('Segoe UI', 9), justify='left', wraplength=520).pack(
                     anchor='w', pady=(2, 0))

        body = tk.Frame(tab, bg=AppStyles.BG_CARD)
        body.pack(fill='both', expand=True, padx=20, pady=(10, 0))

        # Two responsive columns: controls on the LEFT (scrollable, grows to
        # fill), log panel on the RIGHT (narrow, fixed-ish).  Using pack so the
        # left column takes all leftover width at any screen size.
        left_col = tk.Frame(body, bg=AppStyles.BG_CARD)
        left_col.pack(side='left', fill='both', expand=True)

        right_col = tk.Frame(body, bg=AppStyles.BG_CARD, width=300)
        right_col.pack(side='right', fill='y', padx=(12, 0))
        right_col.pack_propagate(False)   # keep the log panel at its set width

        # Scrollable canvas for the input cards (so they don't compete with log)
        canvas = tk.Canvas(left_col, bg=AppStyles.BG_CARD, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_col, orient='vertical', command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=AppStyles.BG_CARD)
        scrollable.bind('<Configure>',
                        lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        _win = canvas.create_window((0, 0), window=scrollable, anchor='nw')
        # Bind the inner frame's width to the canvas width so cards reflow to the
        # available space instead of keeping their natural width and OVERLAPPING.
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(_win, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        # Mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', _on_mousewheel)
        tab.bind('<Destroy>', lambda e: canvas.unbind_all('<MouseWheel>'))

        # ── 1) Source video ────────────────────────────────────────────
        vid_card = self._dub_card(scrollable, '🎬 Source Video')
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

        # ── 1a) Batch folder (optional) ─────────────────────────────────
        # If set, EVERY video in this folder is dubbed one-by-one and the
        # single "Source Video" above is ignored.
        frow = tk.Frame(vid_card, bg=AppStyles.BG_CARD)
        frow.pack(fill='x', padx=8, pady=(0, 4))
        tk.Label(frow, text='or folder:', bg=AppStyles.BG_CARD,
                 fg=AppStyles.TEXT_MEDIUM, font=('Segoe UI', 8),
                 width=8, anchor='w').pack(side='left')
        self._dub_folder_var = tk.StringVar(
            value=self.settings.get('dub_batch_folder', ''))
        tk.Entry(frow, textvariable=self._dub_folder_var, bg=AppStyles.BG_INPUT,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9), relief='flat').pack(
                     side='left', fill='x', expand=True, padx=(0, 6), ipady=3)

        def _browse_folder():
            d = filedialog.askdirectory(
                title='Select a folder of videos to dub (batch)')
            if d:
                self._dub_folder_var.set(d)
                self.update_setting('dub_batch_folder', d)
                self._dub_refresh_batch_count()

        def _clear_folder():
            self._dub_folder_var.set('')
            self.update_setting('dub_batch_folder', '')
            self._dub_refresh_batch_count()
        ModernButton(frow, text='📂 Folder', bg_color=AppStyles.ACCENT_INFO,
                     font=('Segoe UI', 9, 'bold'), padx=10, pady=3,
                     command=_browse_folder).pack(side='left')
        ModernButton(frow, text='✖', bg_color=AppStyles.TEXT_MEDIUM,
                     font=('Segoe UI', 9, 'bold'), padx=8, pady=3,
                     command=_clear_folder).pack(side='left', padx=(4, 0))

        # Recurse into subfolders?
        self._dub_batch_recursive_var = tk.BooleanVar(
            value=bool(self.settings.get('dub_batch_recursive', False)))
        # Skip videos whose output already exists (resume-friendly)?
        self._dub_batch_skip_done_var = tk.BooleanVar(
            value=bool(self.settings.get('dub_batch_skip_done', True)))
        crow2 = tk.Frame(vid_card, bg=AppStyles.BG_CARD)
        crow2.pack(fill='x', padx=8, pady=(0, 2))
        tk.Checkbutton(crow2, text='include subfolders',
                       variable=self._dub_batch_recursive_var,
                       bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_DARK,
                       activebackground=AppStyles.BG_CARD,
                       selectcolor=AppStyles.BG_INPUT, font=('Segoe UI', 8),
                       command=lambda: (self.update_setting(
                           'dub_batch_recursive',
                           self._dub_batch_recursive_var.get()),
                           self._dub_refresh_batch_count())).pack(side='left')
        tk.Checkbutton(crow2, text='skip already-dubbed',
                       variable=self._dub_batch_skip_done_var,
                       bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_DARK,
                       activebackground=AppStyles.BG_CARD,
                       selectcolor=AppStyles.BG_INPUT, font=('Segoe UI', 8),
                       command=lambda: self.update_setting(
                           'dub_batch_skip_done',
                           self._dub_batch_skip_done_var.get())).pack(
                               side='left', padx=(12, 0))
        self._dub_batch_count_var = tk.StringVar(value='')
        tk.Label(vid_card, textvariable=self._dub_batch_count_var,
                 bg=AppStyles.BG_CARD, fg=AppStyles.ACCENT_PRIMARY,
                 font=('Segoe UI', 8, 'italic')).pack(anchor='w', padx=8)
        tk.Label(vid_card,
                 text='   Batch mode: every video in the folder is dubbed into '
                      'the target language one-by-one. Multi-speaker voice '
                      'assignments are per-video, so unmapped speakers use your '
                      'default TTS voice.',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 8, 'italic'), justify='left',
                 wraplength=500).pack(anchor='w', padx=8, pady=(0, 4))
        self._dub_refresh_batch_count()

        # ── 1b) Source language ────────────────────────────────────────
        src_card = self._dub_card(scrollable, '🔊 Source Language')
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
        lang_card = self._dub_card(scrollable, '🌐 Target Language')
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

        # ── 2b) Speaker voices (multi-speaker dubbing) ──────────────────
        spk_card = self._dub_card(scrollable, '🎭 Speaker Voices')

        # Master toggle
        self._dub_multi_var = tk.BooleanVar(
            value=bool(self.settings.get('dub_multispeaker', False)))
        tk.Checkbutton(spk_card,
                       text='Multi-speaker dubbing (detect & assign a voice per speaker)',
                       variable=self._dub_multi_var,
                       bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_DARK,
                       activebackground=AppStyles.BG_CARD,
                       selectcolor=AppStyles.BG_INPUT,
                       font=('Segoe UI', 9),
                       command=lambda: self.update_setting(
                           'dub_multispeaker',
                           self._dub_multi_var.get())).pack(
                               anchor='w', padx=8, pady=(2, 2))

        # HF token (needed only if the local pyannote bundle is missing)
        hrow = tk.Frame(spk_card, bg=AppStyles.BG_CARD)
        hrow.pack(fill='x', padx=8, pady=(2, 2))
        tk.Label(hrow, text='HF Token:', bg=AppStyles.BG_CARD,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9),
                 width=10, anchor='w').pack(side='left')
        self._dub_hf_var = tk.StringVar(value=self.settings.get('hf_token', ''))
        hf_entry = tk.Entry(hrow, textvariable=self._dub_hf_var, show='•',
                            bg=AppStyles.BG_INPUT, fg=AppStyles.TEXT_DARK,
                            font=('Segoe UI', 9), relief='flat')
        hf_entry.pack(side='left', fill='x', expand=True, padx=(0, 6), ipady=3)
        hf_entry.bind('<FocusOut>', lambda e: self.update_setting(
            'hf_token', self._dub_hf_var.get().strip()))
        tk.Label(spk_card,
                 text='   Optional — only needed if the bundled pyannote models '
                      'are missing (see MULTISPEAKER_DUBBING_PLAN.md §2).',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 8, 'italic'), justify='left',
                 wraplength=500).pack(anchor='w', padx=8)

        # Exact speaker count — forces pyannote instead of letting it guess
        # (auto-detection often over-splits one voice into several).
        crow = tk.Frame(spk_card, bg=AppStyles.BG_CARD)
        crow.pack(fill='x', padx=8, pady=(4, 2))
        tk.Label(crow, text='Speakers in video:', bg=AppStyles.BG_CARD,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9),
                 width=16, anchor='w').pack(side='left')
        self._dub_nspk_var = tk.StringVar(
            value=str(self.settings.get('dub_num_speakers') or 'Auto'))
        nspk = ttk.Combobox(
            crow, textvariable=self._dub_nspk_var, width=8, state='readonly',
            values=['Auto', '1', '2', '3', '4', '5', '6', '7', '8'])
        nspk.pack(side='left', padx=(6, 0))
        nspk.bind('<<ComboboxSelected>>', lambda e: self._dub_save_num_speakers())
        tk.Label(crow, text='  (set the exact count for best accuracy)',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 8, 'italic')).pack(side='left')

        # Detect button + rows container
        drow = tk.Frame(spk_card, bg=AppStyles.BG_CARD)
        drow.pack(fill='x', padx=8, pady=(6, 2))
        self._dub_detect_btn = ModernButton(
            drow, text='🔍 Detect Speakers', bg_color=AppStyles.ACCENT_INFO,
            font=('Segoe UI', 9, 'bold'), padx=10, pady=3,
            command=self._dub_detect_speakers)
        self._dub_detect_btn.pack(side='left')
        self._dub_detect_status = tk.StringVar(value='')
        tk.Label(drow, textvariable=self._dub_detect_status,
                 bg=AppStyles.BG_CARD, fg=AppStyles.ACCENT_PRIMARY,
                 font=('Segoe UI', 8, 'italic')).pack(side='left', padx=(10, 0))

        # Container that per-speaker rows get added to (rebuilt on each detect)
        self._dub_speaker_rows = tk.Frame(spk_card, bg=AppStyles.BG_CARD)
        self._dub_speaker_rows.pack(fill='x', padx=8, pady=(2, 6))
        self._dub_speaker_vars = {}   # speaker label → StringVar(voice key)

        # Rebuild rows from any previously-saved mapping
        _saved_map = self.settings.get('dub_speaker_voices') or {}
        if _saved_map:
            self._dub_build_speaker_rows(sorted(_saved_map.keys()))

        # ── 3) Output + mix controls ───────────────────────────────────
        opt_card = self._dub_card(scrollable, '⚙️ Options')

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
                 text='   Leave blank: writes into '
                      '<parent-folder>_<lang>/ with original filename',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 8, 'italic')).pack(anchor='w', padx=8)

        # Duck sliders
        mrow = tk.Frame(opt_card, bg=AppStyles.BG_CARD)
        mrow.pack(fill='x', padx=8, pady=(8, 4))
        self._dub_duck_var = tk.DoubleVar(
            value=float(self.settings.get('dub_original_duck', 0.12)))
        self._dub_bg_var = tk.DoubleVar(
            value=float(self.settings.get('dub_original_bg', 0.55)))
        self._dub_slider(mrow, 'Orig. vol. during dub:', self._dub_duck_var,
                         'dub_original_duck')
        self._dub_slider(mrow, 'Orig. vol. (music/gaps):', self._dub_bg_var,
                         'dub_original_bg')
        tk.Label(opt_card,
                 text='   During dub = how loud the original stays UNDER the '
                      'dubbed voice.  Music/gaps = its volume where nobody is '
                      'being dubbed (background music, pauses).',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 8, 'italic'), justify='left',
                 wraplength=500).pack(anchor='w', padx=8)

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
                 wraplength=500).pack(anchor='w', padx=8)

        # Keep original music & SFX (Demucs vocal removal) ---------------
        self._dub_keep_music_var = tk.BooleanVar(
            value=bool(self.settings.get('dub_keep_music', False)))
        tk.Checkbutton(opt_card,
                       text='Keep original music & sound effects (remove only voices)',
                       variable=self._dub_keep_music_var,
                       bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_DARK,
                       activebackground=AppStyles.BG_CARD,
                       selectcolor=AppStyles.BG_INPUT,
                       font=('Segoe UI', 8),
                       command=lambda: self.update_setting(
                           'dub_keep_music',
                           self._dub_keep_music_var.get())).pack(
                               anchor='w', padx=8, pady=(6, 0))
        tk.Label(opt_card,
                 text='   Uses AI (Demucs) to strip the actors’ speech while '
                      'keeping the score, ambience & effects at full quality — '
                      'the dub then sits on a clean music/SFX bed. Adds a short '
                      'GPU pass per video; falls back to ducking if unavailable.',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 8, 'italic'), justify='left',
                 wraplength=500).pack(anchor='w', padx=8, pady=(0, 2))

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
        run_row = tk.Frame(scrollable, bg=AppStyles.BG_CARD)
        run_row.pack(fill='x', pady=(4, 2))
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
        tk.Label(scrollable, textvariable=self._dub_status_var,
                 bg=AppStyles.BG_CARD, fg=AppStyles.ACCENT_PRIMARY,
                 font=('Segoe UI', 9)).pack(anchor='w', pady=(0, 4))

        # ── 5) Log box — small panel on the RIGHT, fills its column height ──
        log_card = tk.Frame(right_col, bg=AppStyles.BG_CARD,
                            highlightbackground='#30363d', highlightthickness=1)
        log_card.pack(fill='both', expand=True)
        tk.Label(log_card, text='📋 Log', bg=AppStyles.BG_CARD,
                 fg=AppStyles.ACCENT_PRIMARY,
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', padx=8, pady=(6, 0))
        _log_wrap = tk.Frame(log_card, bg=AppStyles.BG_CARD)
        _log_wrap.pack(fill='both', expand=True, padx=6, pady=6)
        _log_scroll = ttk.Scrollbar(_log_wrap, orient='vertical')
        _log_scroll.pack(side='right', fill='y')
        self._dub_log_widget = tk.Text(
            _log_wrap, width=1, wrap='word', bg=AppStyles.BG_INPUT,
            fg=AppStyles.TEXT_DARK, font=('Consolas', 8), relief='flat', bd=4,
            yscrollcommand=_log_scroll.set)
        self._dub_log_widget.pack(side='left', fill='both', expand=True)
        _log_scroll.config(command=self._dub_log_widget.yview)
        self._dub_running = False

    # ── Batch-folder helpers ────────────────────────────────────────────
    VIDEO_EXTS = ('.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.flv',
                  '.wmv', '.mpg', '.mpeg', '.ts')

    def _dub_scan_folder(self, folder: str, recursive: bool):
        """Return a sorted list of video Paths in *folder*.

        Skips our own outputs so a re-run over the same folder never dubs
        a dub — both legacy ``*_dubbed_*.mp4`` files and the new subfolder
        layout (``<parent>_<lang>/original_name.mp4``) are excluded.
        """
        from pathlib import Path as _P
        base = _P(folder)
        if not base.is_dir():
            return []
        # Compute current output-subfolder suffix (e.g. "_english") so
        # files sitting inside a previously-dubbed output folder are skipped.
        _lang = (self._dub_lang_var.get() or '').strip()
        _safe_lang = _lang.lower().replace(' ', '_') if _lang else ''
        it = base.rglob('*') if recursive else base.glob('*')
        vids = []
        for p in it:
            if not p.is_file():
                continue
            if p.suffix.lower() not in self.VIDEO_EXTS:
                continue
            if '_dubbed_' in p.stem.lower():
                continue
            # Skip files inside a previous dub-output subfolder
            if _safe_lang and p.parent.name.endswith(f'_{_safe_lang}'):
                continue
            vids.append(p)
        return sorted(vids, key=lambda p: str(p).lower())

    def _dub_batch_out_path(self, src, lang: str):
        """Where the dub for *src* is written.

        Creates a subfolder named ``<parent-folder>_<lang>`` next to the
        source video and keeps the **original filename** (no ``_dubbed_``
        suffix).  If the user set a custom output-folder override in the UI
        that folder is used as the base instead of the source's parent.

        Example:
            ``videos/MyChannel/video.mp4`` dubbed to ``english`` →
            ``videos/MyChannel/MyChannel_english/video.mp4``
        """
        from pathlib import Path as _P
        src = _P(src)
        out_folder = (self._dub_out_var.get() or '').strip()
        safe_lang = lang.lower().replace(' ', '_')
        # The "channel name" is the source video's parent folder name.
        parent_name = src.parent.name
        subfolder_name = f'{parent_name}_{safe_lang}'
        base_dir = _P(out_folder) if out_folder else src.parent
        output_dir = base_dir / subfolder_name
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass  # will fail downstream if the dir is unwritable
        return output_dir / src.name  # keep original filename

    def _dub_refresh_batch_count(self):
        """Update the '(N videos found)' hint next to the folder picker."""
        try:
            folder = (self._dub_folder_var.get() or '').strip()
        except Exception:
            return
        if not folder:
            self._dub_batch_count_var.set('')
            return
        vids = self._dub_scan_folder(
            folder, bool(self._dub_batch_recursive_var.get()))
        n = len(vids)
        if n == 0:
            self._dub_batch_count_var.set('   ⚠ no videos found in that folder')
        else:
            self._dub_batch_count_var.set(
                f'   📂 batch mode ON — {n} video(s) queued')

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

    # ── Multi-speaker: detect + voice mapping ───────────────────────────
    def _dub_voice_keys(self):
        """Gemini TTS voice keys for the speaker dropdowns (safe fallback).

        The child-voice presets are appended so a speaker can be voiced as a
        boy/girl (adult voice pitch-shifted up, or Edge's real child voice).
        """
        keys = None
        try:
            from gemini_api_tts_helper import get_voice_keys
            keys = get_voice_keys()
        except Exception:
            keys = None
        if not keys:
            keys = ['Zephyr', 'Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede']
        try:
            from dubbing_engine import CHILD_VOICE_PRESETS
            child = list(CHILD_VOICE_PRESETS.keys())
        except Exception:
            child = ['Child girl (Gemini)', 'Child boy (Gemini)',
                     'Child girl (Edge)', 'Child boy (Edge)']
        return list(keys) + child

    def _dub_voice_label(self, key):
        """'Puck' → 'Puck (Male) — upbeat · ads/reactions' for display.

        Appends gender + a best-use hint so voices aren't picked blind.  Falls
        back to bare key if unknown.  Child presets already read naturally
        (e.g. 'Child girl (Gemini)'), so they are shown as-is.
        """
        if str(key).startswith('Child '):
            return str(key)
        try:
            from gemini_api_tts_helper import get_voice_gender, get_voice_use
            g = get_voice_gender(key)
            use = get_voice_use(key)
            base = f'{key} ({g})' if g else str(key)
            return f'{base} — {use}' if use else base
        except Exception:
            return key

    def _dub_build_speaker_rows(self, speakers, genders=None):
        """Render one label + voice Combobox per detected speaker.

        ``genders`` maps speaker→'Male'/'Female' (from pitch analysis).  The
        default voice is picked to MATCH that gender when known, cycling within
        the matching-gender voices so two same-gender speakers still differ.
        Any voice already saved for a speaker is preserved.  Every change
        persists the whole mapping to ``settings['dub_speaker_voices']``.
        """
        genders = genders or {}
        # Clear previous rows
        for child in list(self._dub_speaker_rows.winfo_children()):
            child.destroy()
        self._dub_speaker_vars = {}

        voice_keys = self._dub_voice_keys()
        saved = self.settings.get('dub_speaker_voices') or {}
        self._dub_speaker_genders = genders

        # Group voice keys by gender so defaults can be gender-matched.
        try:
            from gemini_api_tts_helper import get_voice_gender
        except Exception:
            get_voice_gender = lambda k: ''
        by_gender = {'Male': [], 'Female': []}
        for k in voice_keys:
            g = get_voice_gender(k)
            if g in by_gender:
                by_gender[g].append(k)

        # Display labels carry the gender, e.g. "Puck (Male)"; map back to keys.
        display_values = [self._dub_voice_label(k) for k in voice_keys]
        self._dub_label_to_key = {self._dub_voice_label(k): k for k in voice_keys}

        _gender_counts = {'Male': 0, 'Female': 0}
        for idx, spk in enumerate(speakers):
            row = tk.Frame(self._dub_speaker_rows, bg=AppStyles.BG_CARD)
            row.pack(fill='x', pady=2)
            g = genders.get(spk, '')
            spk_label = f'{spk} ({g}):' if g else f'{spk}:'
            tk.Label(row, text=spk_label, bg=AppStyles.BG_CARD,
                     fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9),
                     width=18, anchor='w').pack(side='left')
            # Pre-fill: saved choice, else a gender-matched voice (cycling within
            # that gender so two same-gender speakers get distinct voices), else
            # fall back to cycling the whole list.
            default = saved.get(spk)
            if not default:
                pool = by_gender.get(g) if g else None
                if pool:
                    default = pool[_gender_counts[g] % len(pool)]
                    _gender_counts[g] += 1
                elif voice_keys:
                    default = voice_keys[idx % len(voice_keys)]
            # The visible StringVar shows the labelled form; the bare key is
            # recovered in _dub_save_speaker_map via _dub_label_to_key.
            var = tk.StringVar(value=self._dub_voice_label(default) if default else '')
            combo = ttk.Combobox(row, textvariable=var, values=display_values,
                                 width=42, state='readonly')
            combo.pack(side='left', padx=(6, 0))
            combo.bind('<<ComboboxSelected>>',
                       lambda e: self._dub_save_speaker_map())
            self._dub_speaker_vars[spk] = var

            # Preview buttons: hear the ACTOR's real voice vs. the ASSIGNED
            # voice, so voices aren't assigned blind.
            ModernButton(row, text='🎬 Actor', bg_color=AppStyles.ACCENT_INFO,
                         font=('Segoe UI', 8), padx=6, pady=2,
                         command=lambda s=spk: self._dub_preview_actor(s)).pack(
                             side='left', padx=(6, 0))
            ModernButton(row, text='▶ Voice', bg_color=AppStyles.ACCENT_SUCCESS,
                         font=('Segoe UI', 8), padx=6, pady=2,
                         command=lambda s=spk: self._dub_preview_voice(s)).pack(
                             side='left', padx=(4, 0))

        # Persist the (possibly auto-suggested) mapping immediately
        self._dub_save_speaker_map()

    def _dub_save_num_speakers(self):
        """Persist the exact-speaker-count choice.

        'Auto' clears the hint (pyannote guesses); a number N pins BOTH the min
        and max so the diarizer returns exactly N speakers.
        """
        raw = (self._dub_nspk_var.get() or 'Auto').strip()
        if raw.isdigit():
            n = int(raw)
            self.update_setting('dub_num_speakers', n)
            self.update_setting('dub_min_speakers', n)
            self.update_setting('dub_max_speakers', n)
        else:
            self.update_setting('dub_num_speakers', 0)
            self.update_setting('dub_min_speakers', 0)
            self.update_setting('dub_max_speakers', 0)

    def _dub_save_speaker_map(self):
        """Write the current speaker→voice dropdown state to settings.

        The comboboxes show labelled forms like "Puck (Male)"; store the bare
        voice key so the TTS engine gets exactly what it expects.
        """
        label_to_key = getattr(self, '_dub_label_to_key', {})
        mapping = {}
        for spk, var in self._dub_speaker_vars.items():
            disp = var.get()
            if not disp:
                continue
            mapping[spk] = label_to_key.get(disp, disp)
        self.update_setting('dub_speaker_voices', mapping)

    # ── Speaker/voice preview ───────────────────────────────────────────
    def _dub_preview_actor(self, speaker):
        """Play ~10s of the ACTOR's real voice from the video, so the user can
        hear who SPEAKER_xx actually is (and their true gender)."""
        if getattr(self, '_dub_running', False):
            self._dub_log('warn', 'Busy — please wait.')
            return
        video = (self._dub_video_var.get() or '').strip()
        if not video or not Path(video).is_file():
            messagebox.showerror('Preview', 'Pick a valid video file first.')
            return
        segs = getattr(self, '_dub_detected_segments', None)
        if not segs:
            messagebox.showinfo('Preview',
                                'Run "Detect Speakers" first so I know each '
                                'actor\'s timing.')
            return
        self._dub_log('info', f'Preview: extracting {speaker} audio…')
        t = threading.Thread(
            target=self._dub_preview_actor_worker,
            args=(Path(video), list(segs), speaker), daemon=True)
        t.start()

    def _dub_preview_actor_worker(self, video, segs, speaker):
        try:
            import tempfile
            out = Path(tempfile.gettempdir()) / f'_dub_actor_{speaker}.wav'
            res = dubbing_engine.preview_actor_clip(
                video, segs, speaker, out, log=self._dub_log, max_dur=10.0)
            if res and Path(res).is_file():
                os.startfile(str(res))  # play in default audio player
            else:
                self._dub_log('warn', f'Preview: no audio for {speaker}')
        except Exception as e:
            self._dub_log('error', f'Preview actor failed: {e}')

    def _dub_preview_voice(self, speaker):
        """Render + play a short TTS sample of the voice currently assigned to
        this speaker — exactly as the dub will produce it."""
        if getattr(self, '_dub_running', False):
            self._dub_log('warn', 'Busy — please wait.')
            return
        var = self._dub_speaker_vars.get(speaker)
        disp = var.get() if var else ''
        label_to_key = getattr(self, '_dub_label_to_key', {})
        voice = label_to_key.get(disp, disp)
        if not voice:
            messagebox.showinfo('Preview', 'Pick a voice for this speaker first.')
            return
        self._dub_log('info', f'Preview: rendering "{voice}" for {speaker}…')
        t = threading.Thread(
            target=self._dub_preview_voice_worker,
            args=(speaker, voice), daemon=True)
        t.start()

    def _dub_preview_voice_worker(self, speaker, voice):
        try:
            import tempfile
            # A short line in the target language reads more naturally than
            # English when auditioning a dub voice.
            sample = self._dub_preview_sample_text()
            out = Path(tempfile.gettempdir()) / f'_dub_voice_{speaker}.mp3'
            res = dubbing_engine.preview_voice(
                sample, voice, self.settings, out, log=self._dub_log)
            if res and Path(res).is_file():
                os.startfile(str(res))
            else:
                self._dub_log('warn', f'Preview: could not render {voice}')
        except Exception as e:
            self._dub_log('error', f'Preview voice failed: {e}')

    def _dub_preview_sample_text(self):
        """A one-line audition sample; uses the target language when known."""
        tgt = (self._dub_lang_var.get() or '').strip().lower() if hasattr(
            self, '_dub_lang_var') else ''
        samples = {
            'urdu': 'السلام علیکم، یہ میری آواز کا نمونہ ہے۔',
            'hindi': 'नमस्ते, यह मेरी आवाज़ का नमूना है।',
            'arabic': 'مرحبا، هذه عينة من صوتي.',
        }
        for k, v in samples.items():
            if k in tgt:
                return v
        return 'Hello, this is a sample of my dubbing voice.'

    def _dub_detect_speakers(self):
        """Run diarized transcription on the chosen video → list speakers."""
        if getattr(self, '_dub_running', False):
            self._dub_log('warn', 'A dub is already running — please wait.')
            return
        video = (self._dub_video_var.get() or '').strip()
        if not video or not Path(video).is_file():
            messagebox.showerror('Detect Speakers',
                                 'Please pick a valid video file first.')
            return
        # Persist the toggle + token before detecting
        self.update_setting('dub_multispeaker', self._dub_multi_var.get())
        self.update_setting('hf_token', self._dub_hf_var.get().strip())

        self._dub_running = True
        self._dub_detect_btn.config(state='disabled')
        self._dub_detect_status.set('Detecting… (this can take a minute)')
        self._dub_log('header', 'Detecting speakers…')
        src_lang = (self._dub_src_lang_var.get() or '').strip()
        t = threading.Thread(
            target=self._dub_detect_worker,
            args=(Path(video),
                  src_lang if src_lang != 'Auto-detect' else None),
            daemon=True)
        t.start()

    def _dub_detect_worker(self, video: Path, src_lang):
        try:
            words = dubbing_engine.transcribe_video(
                video, self.settings, log=self._dub_log,
                source_language=src_lang, diarize=True,
                hf_token=self._dub_hf_var.get().strip() or None,
                min_spk=self.settings.get('dub_min_speakers') or None,
                max_spk=self.settings.get('dub_max_speakers') or None)
            segs = dubbing_engine.group_words_into_segments(words) if words else []
            speakers = dubbing_engine.distinct_speakers(segs) if segs else []
            if not speakers:
                self._dub_log('warn', 'No speakers detected — check the video/log.')
                self._dub_video_widget_after(
                    lambda: self._dub_detect_status.set('No speakers detected.'))
                return
            self._dub_log('ok', f'Detected {len(speakers)} speaker(s): {speakers}')
            # Cache the diarized segments + video so the per-speaker preview
            # buttons can extract each actor's ORIGINAL voice on demand.
            self._dub_detected_segments = segs
            self._dub_detected_video = video
            # Estimate each speaker's gender from voice pitch so the default
            # voice can be gender-matched instead of assigned by index.
            genders = {}
            try:
                genders = dubbing_engine.estimate_speaker_genders(
                    video, segs, log=self._dub_log)
                if genders:
                    self._dub_log('info', f'Dub: estimated genders {genders}')
            except Exception as e:
                self._dub_log('warn', f'Dub: gender estimate failed ({e})')
            self._dub_video_widget_after(
                lambda: (self._dub_build_speaker_rows(speakers, genders),
                         self._dub_detect_status.set(
                             f'{len(speakers)} speaker(s) — pick a voice each.')))
        except Exception as e:
            self._dub_log('error', f'Detect speakers failed: {e}')
            for ln in traceback.format_exc().splitlines():
                self._dub_log('error', f'  {ln}')
            self._dub_video_widget_after(
                lambda: self._dub_detect_status.set('❌ Failed. See log.'))
        finally:
            self._dub_running = False
            self._dub_video_widget_after(
                lambda: self._dub_detect_btn.config(state='normal'))

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

        lang = (self._dub_lang_var.get() or '').strip()
        src_lang = (self._dub_src_lang_var.get() or '').strip()
        if not lang:
            messagebox.showerror('Dubbing', 'Please pick a target language.')
            return

        # ── Batch-folder mode takes priority when a folder is set ─────────
        folder = (self._dub_folder_var.get() or '').strip()
        if folder:
            if not Path(folder).is_dir():
                messagebox.showerror('Dubbing', 'The batch folder does not exist.')
                return
            recursive = bool(self._dub_batch_recursive_var.get())
            vids = self._dub_scan_folder(folder, recursive)
            if not vids:
                messagebox.showerror(
                    'Dubbing', 'No videos found in that folder.')
                return
            self.update_setting('dub_batch_folder', folder)
            self.update_setting('dub_target_language', lang)
            self.update_setting('dub_source_language', src_lang)

            self._dub_running = True
            self._dub_run_btn.config(state='disabled')
            self._dub_progress_var.set(0)
            self._dub_log('header',
                          f'Batch dub started: {len(vids)} video(s) → {lang}')
            self._dub_set_status(f'Batch: 0/{len(vids)}…')
            t = threading.Thread(
                target=self._dub_batch_worker,
                args=(vids, lang, src_lang), daemon=True)
            t.start()
            return

        # ── Single-video mode ─────────────────────────────────────────────
        video = (self._dub_video_var.get() or '').strip()
        if not video or not Path(video).is_file():
            messagebox.showerror(
                'Dubbing', 'Please pick a valid video file, or choose a '
                           'folder for batch mode.')
            return

        src = Path(video)
        out_video = self._dub_batch_out_path(src, lang)

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

    def _dub_batch_worker(self, vids, lang: str, src_lang: str):
        """Dub every video in *vids* one-by-one on this worker thread.

        Each video is independent: a failure on one is logged and the batch
        continues with the next. The overall progress bar tracks video count;
        per-video engine progress is streamed to the log/status line.
        """
        total = len(vids)
        done = ok = skipped = failed = 0
        skip_done = bool(self._dub_batch_skip_done_var.get())
        try:
            for idx, src in enumerate(vids, 1):
                if not getattr(self, '_dub_running', False):
                    self._dub_log('warn', 'Batch cancelled.')
                    break
                out_video = self._dub_batch_out_path(src, lang)
                self._dub_set_status(
                    f'Batch {idx}/{total}: {src.name}')
                self._dub_set_progress(idx - 1, total,
                                       f'Batch {idx}/{total}: {src.name}')

                if skip_done and Path(out_video).is_file() \
                        and Path(out_video).stat().st_size > 0:
                    self._dub_log('info',
                                  f'[{idx}/{total}] ⏭ already dubbed — {src.name}')
                    skipped += 1
                    done += 1
                    continue

                self._dub_log('header', f'[{idx}/{total}] {src.name} → {lang}')
                try:
                    result = dubbing_engine.dub_video(
                        src, out_video, lang, self.settings,
                        log=self._dub_log,
                        progress=self._dub_set_progress,
                        keep_audio_file=bool(self._dub_keep_audio_var.get()),
                        source_language=src_lang
                        if src_lang != 'Auto-detect' else None)
                    if result is not None:
                        self._dub_log('ok',
                                      f'[{idx}/{total}] ✅ {Path(result).name}')
                        ok += 1
                    else:
                        self._dub_log('error',
                                      f'[{idx}/{total}] ❌ failed — {src.name}')
                        failed += 1
                except Exception as e:
                    self._dub_log('error',
                                  f'[{idx}/{total}] error on {src.name}: {e}')
                    for ln in traceback.format_exc().splitlines():
                        self._dub_log('error', f'  {ln}')
                    failed += 1
                done += 1
                self._dub_set_progress(done, total)

            summary = (f'Batch done: {ok} dubbed, {skipped} skipped, '
                       f'{failed} failed (of {total}).')
            self._dub_log('header', summary)
            self._dub_set_status(f'✅ {summary}')
            self._dub_video_widget_after(
                lambda: messagebox.showinfo('Batch dubbing complete', summary))
        finally:
            self._dub_running = False
            self._dub_video_widget_after(
                lambda: self._dub_run_btn.config(state='normal'))

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