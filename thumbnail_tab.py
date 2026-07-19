#!/usr/bin/env python
"""
Thumbnail Tab (UI)
==================
A self-contained Tk tab that designs a video thumbnail with Google's Gemini
image models (Nano Banana / Nano Banana Pro).  Kept OUT of
``complete_automation_gui.py`` so the main file stays small — same convention
as ``dubbing_tab.py``.

Wiring (two lines in the main file):

    from thumbnail_tab import ThumbnailTabMixin
    class VideoAutomationGUI(DubbingTabMixin, ThumbnailTabMixin):   # add mixin
        ...
        self.create_thumbnail_tab()                     # after the other tabs

Reuses the host GUI's ``self.settings`` (for the shared Gemini service account
/ API key), ``self.update_setting``, ``self.notebook``, plus AppStyles /
ModernButton.  All image work happens in ``thumbnail_designer.py``.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import thumbnail_designer


AppStyles = None
ModernButton = None


def _styles():
    from complete_automation_gui import AppStyles, ModernButton
    return AppStyles, ModernButton


# Aspect presets: label → (ratio string, friendly note).
ASPECTS = {
    '16:9  (YouTube)': '16:9',
    '9:16  (Shorts / Reels)': '9:16',
    '1:1  (Square)': '1:1',
    '4:3': '4:3',
}

# A sensible default styling prompt the user can edit per thumbnail.
DEFAULT_PROMPT = (
    "Design a bold, high-CTR YouTube thumbnail. Keep the main subject's face "
    "from the reference image sharp and well-lit, push the background into a "
    "dramatic, high-contrast scene, add cinematic color grading and a subtle "
    "glow around the subject. Leave clean space for a short punchy title. "
    "No watermarks, no gibberish text."
)


class ThumbnailTabMixin:
    """Adds a 🖼 Thumbnails tab to the host VideoAutomationGUI."""

    # ── Tab construction ────────────────────────────────────────────────
    def create_thumbnail_tab(self):
        global AppStyles, ModernButton
        AppStyles, ModernButton = _styles()
        tab = tk.Frame(self.notebook, bg=AppStyles.BG_CARD)
        self.notebook.add(tab, text='🖼 Thumbnails')

        header = tk.Frame(tab, bg=AppStyles.BG_CARD)
        header.pack(fill='x', padx=20, pady=(14, 4))
        tk.Label(header, text='🖼 Thumbnail Designer',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_DARK,
                 font=('Segoe UI', 14, 'bold')).pack(anchor='w')
        tk.Label(header,
                 text='Grab a frame from any video (or start blank), tweak the '
                      'styling prompt, and Gemini paints a finished thumbnail. '
                      'Uses the same Gemini service account as your TTS — no '
                      'extra key needed.',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 9), justify='left', wraplength=760).pack(
                     anchor='w', pady=(2, 0))

        body = tk.Frame(tab, bg=AppStyles.BG_CARD)
        body.pack(fill='both', expand=True, padx=20, pady=10)

        # Scrollable canvas so everything stays accessible on small screens
        canvas = tk.Canvas(body, bg=AppStyles.BG_CARD, highlightthickness=0)
        scrollbar = ttk.Scrollbar(body, orient='vertical', command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=AppStyles.BG_CARD)
        scrollable.bind('<Configure>',
                        lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scrollable, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', _on_mousewheel)
        # Unbind on destroy to avoid leaking
        tab.bind('<Destroy>', lambda e: canvas.unbind_all('<MouseWheel>'))

        # ── 0) Load from a Case-Commentary Excel ───────────────────────
        excel_card = self._th_card(scrollable, '📄 Load from Excel (optional)')
        exrow = tk.Frame(excel_card, bg=AppStyles.BG_CARD)
        exrow.pack(fill='x', padx=8, pady=6)
        self._th_excel_var = tk.StringVar(
            value=self.settings.get('thumb_last_excel', ''))
        tk.Entry(exrow, textvariable=self._th_excel_var, bg=AppStyles.BG_INPUT,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9),
                 relief='flat').pack(side='left', fill='x', expand=True,
                                     padx=(0, 6), ipady=3)
        ModernButton(exrow, text='📂 Browse', bg_color=AppStyles.ACCENT_INFO,
                     font=('Segoe UI', 9, 'bold'), padx=10, pady=3,
                     command=self._th_browse_excel).pack(side='left')
        ModernButton(exrow, text='⤵ Load Row', bg_color=AppStyles.ACCENT_PRIMARY,
                     font=('Segoe UI', 9, 'bold'), padx=10, pady=3,
                     command=self._th_load_excel).pack(side='left', padx=(6, 0))
        tk.Label(excel_card,
                 text='   Auto-fills the reference frame, title text and aspect '
                      'from Gemini\'s thumbnail pick (Thumbnail Time/Text cols).',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 8, 'italic')).pack(anchor='w', padx=8)

        # ── 1) Source video + frame grab ───────────────────────────────
        vid_card = self._th_card(scrollable, '🎬 Reference Frame')
        row = tk.Frame(vid_card, bg=AppStyles.BG_CARD)
        row.pack(fill='x', padx=8, pady=6)
        self._th_video_var = tk.StringVar(
            value=self.settings.get('thumb_last_video', ''))
        tk.Entry(row, textvariable=self._th_video_var, bg=AppStyles.BG_INPUT,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9),
                 relief='flat').pack(side='left', fill='x', expand=True,
                                     padx=(0, 6), ipady=3)

        def _browse_video():
            f = filedialog.askopenfilename(
                title='Select a video (frame source)',
                filetypes=[('Video files', '*.mp4 *.mov *.mkv *.avi *.webm'),
                           ('All files', '*.*')])
            if f:
                self._th_video_var.set(f)
                self.update_setting('thumb_last_video', f)
        ModernButton(row, text='📁 Browse', bg_color=AppStyles.ACCENT_INFO,
                     font=('Segoe UI', 9, 'bold'), padx=10, pady=3,
                     command=_browse_video).pack(side='left')

        # timestamp + grab
        trow = tk.Frame(vid_card, bg=AppStyles.BG_CARD)
        trow.pack(fill='x', padx=8, pady=(0, 6))
        tk.Label(trow, text='Frame at (sec):', bg=AppStyles.BG_CARD,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9)).pack(side='left')
        self._th_time_var = tk.StringVar(
            value=str(self.settings.get('thumb_frame_time', '2')))
        tk.Spinbox(trow, from_=0, to=100000, increment=1,
                   textvariable=self._th_time_var, width=8,
                   font=('Segoe UI', 9)).pack(side='left', padx=(6, 8))
        ModernButton(trow, text='📸 Grab Frame', bg_color=AppStyles.ACCENT_INFO,
                     font=('Segoe UI', 9, 'bold'), padx=10, pady=3,
                     command=self._th_grab_frame).pack(side='left')
        tk.Label(trow, text='  → grabbed frame becomes the reference below',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 8, 'italic')).pack(side='left', padx=(6, 0))

        # editable reference image path
        frow = tk.Frame(vid_card, bg=AppStyles.BG_CARD)
        frow.pack(fill='x', padx=8, pady=(0, 8))
        tk.Label(frow, text='Reference image:', bg=AppStyles.BG_CARD,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9)).pack(side='left')
        self._th_ref_var = tk.StringVar(value='')
        tk.Entry(frow, textvariable=self._th_ref_var, bg=AppStyles.BG_INPUT,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9),
                 relief='flat').pack(side='left', fill='x', expand=True,
                                     padx=(6, 6), ipady=3)

        def _browse_ref():
            f = filedialog.askopenfilename(
                title='Select a reference image (optional)',
                filetypes=[('Images', '*.png *.jpg *.jpeg *.webp'),
                           ('All files', '*.*')])
            if f:
                self._th_ref_var.set(f)
        ModernButton(frow, text='📁', bg_color=AppStyles.ACCENT_INFO,
                     font=('Segoe UI', 9, 'bold'), padx=8, pady=3,
                     command=_browse_ref).pack(side='left')
        tk.Label(vid_card,
                 text='   Leave blank to generate purely from the prompt (no '
                      'reference frame).',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 8, 'italic')).pack(anchor='w', padx=8)

        # ── 2) Model + aspect ──────────────────────────────────────────
        model_card = self._th_card(scrollable, '🤖 Model & Format')
        mrow = tk.Frame(model_card, bg=AppStyles.BG_CARD)
        mrow.pack(fill='x', padx=8, pady=6)
        tk.Label(mrow, text='Model:', bg=AppStyles.BG_CARD,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9)).pack(side='left')
        _model_labels = list(thumbnail_designer.MODELS.keys())
        _saved_model = self.settings.get('thumb_model', _model_labels[0])
        if _saved_model not in _model_labels:
            _saved_model = _model_labels[0]
        self._th_model_var = tk.StringVar(value=_saved_model)
        mcombo = ttk.Combobox(mrow, textvariable=self._th_model_var,
                              values=_model_labels, width=28, state='readonly')
        mcombo.pack(side='left', padx=(6, 16))
        mcombo.bind('<<ComboboxSelected>>', lambda e: self.update_setting(
            'thumb_model', self._th_model_var.get()))

        tk.Label(mrow, text='Aspect:', bg=AppStyles.BG_CARD,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9)).pack(side='left')
        _aspect_labels = list(ASPECTS.keys())
        _saved_aspect = self.settings.get('thumb_aspect', _aspect_labels[0])
        if _saved_aspect not in _aspect_labels:
            _saved_aspect = _aspect_labels[0]
        self._th_aspect_var = tk.StringVar(value=_saved_aspect)
        acombo = ttk.Combobox(mrow, textvariable=self._th_aspect_var,
                              values=_aspect_labels, width=20, state='readonly')
        acombo.pack(side='left', padx=(6, 0))
        acombo.bind('<<ComboboxSelected>>', lambda e: self.update_setting(
            'thumb_aspect', self._th_aspect_var.get()))
        tk.Label(model_card,
                 text='   Nano Banana = cheap & fast · Pro = crisper text, '
                      'pricier (preview regions only).',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 8, 'italic')).pack(anchor='w', padx=8)

        # ── 3) Title + styling prompt (editable) ───────────────────────
        prompt_card = self._th_card(scrollable, '✏️ Title & Styling Prompt')
        titrow = tk.Frame(prompt_card, bg=AppStyles.BG_CARD)
        titrow.pack(fill='x', padx=8, pady=(6, 2))
        tk.Label(titrow, text='Title / hook text:', bg=AppStyles.BG_CARD,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9)).pack(side='left')
        self._th_title_var = tk.StringVar(
            value=self.settings.get('thumb_title', ''))
        tk.Entry(titrow, textvariable=self._th_title_var, bg=AppStyles.BG_INPUT,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9),
                 relief='flat').pack(side='left', fill='x', expand=True,
                                     padx=(6, 0), ipady=3)
        tk.Label(prompt_card,
                 text='   Optional — the words to render big on the thumbnail. '
                      'Leave blank for image-only.',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 8, 'italic')).pack(anchor='w', padx=8)

        self._th_prompt_widget = tk.Text(
            prompt_card, height=5, wrap='word', bg=AppStyles.BG_INPUT,
            fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9), relief='flat', bd=4)
        self._th_prompt_widget.pack(fill='x', padx=8, pady=(6, 4))
        self._th_prompt_widget.insert(
            '1.0', self.settings.get('thumb_prompt', DEFAULT_PROMPT))

        # ── 4) Output ───────────────────────────────────────────────────
        out_card = self._th_card(scrollable, '💾 Output')
        orow = tk.Frame(out_card, bg=AppStyles.BG_CARD)
        orow.pack(fill='x', padx=8, pady=6)
        tk.Label(orow, text='Save to:', bg=AppStyles.BG_CARD,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9)).pack(side='left')
        self._th_out_var = tk.StringVar(
            value=self.settings.get('thumb_output', ''))
        tk.Entry(orow, textvariable=self._th_out_var, bg=AppStyles.BG_INPUT,
                 fg=AppStyles.TEXT_DARK, font=('Segoe UI', 9),
                 relief='flat').pack(side='left', fill='x', expand=True,
                                     padx=(6, 6), ipady=3)

        def _browse_out():
            f = filedialog.asksaveasfilename(
                title='Save thumbnail as', defaultextension='.png',
                filetypes=[('PNG image', '*.png'), ('JPEG image', '*.jpg')])
            if f:
                self._th_out_var.set(f)
                self.update_setting('thumb_output', f)
        ModernButton(orow, text='📁', bg_color=AppStyles.ACCENT_INFO,
                     font=('Segoe UI', 9, 'bold'), padx=8, pady=3,
                     command=_browse_out).pack(side='left')
        tk.Label(out_card,
                 text='   Leave blank to save next to the video as '
                      '<name>_thumbnail.png',
                 bg=AppStyles.BG_CARD, fg=AppStyles.TEXT_MEDIUM,
                 font=('Segoe UI', 8, 'italic')).pack(anchor='w', padx=8)

        # ── 5) Run button + progress ────────────────────────────────────
        run_row = tk.Frame(scrollable, bg=AppStyles.BG_CARD)
        run_row.pack(fill='x', pady=(6, 4))
        self._th_run_btn = ModernButton(
            run_row, text='🎨  Generate Thumbnail',
            bg_color=AppStyles.ACCENT_SUCCESS, hover_color='#059669',
            font=('Segoe UI', 11, 'bold'), padx=18, pady=8,
            command=self._th_start)
        self._th_run_btn.pack(side='left')
        self._th_open_btn = ModernButton(
            run_row, text='🖼 Open Last', bg_color=AppStyles.ACCENT_INFO,
            font=('Segoe UI', 10, 'bold'), padx=12, pady=8,
            command=self._th_open_last)
        self._th_open_btn.pack(side='left', padx=(8, 0))

        self._th_status_var = tk.StringVar(value='Ready.')
        tk.Label(scrollable, textvariable=self._th_status_var,
                 bg=AppStyles.BG_CARD, fg=AppStyles.ACCENT_PRIMARY,
                 font=('Segoe UI', 9)).pack(anchor='w', pady=(4, 4))

        # ── 6) Log box ──────────────────────────────────────────────────
        log_card = self._th_card(scrollable, '📋 Log')
        # Make the log card expand vertically
        log_card.pack_forget()
        log_card.pack(fill='both', expand=True, pady=6)
        log_frame = tk.Frame(log_card, bg=AppStyles.BG_CARD)
        log_frame.pack(fill='both', expand=True, padx=6, pady=6)
        self._th_log_widget = tk.Text(
            log_frame, height=6, wrap='word', bg=AppStyles.BG_INPUT,
            fg=AppStyles.TEXT_DARK, font=('Consolas', 8), relief='flat', bd=4)
        self._th_log_widget.pack(side='left', fill='both', expand=True)
        log_scroll = ttk.Scrollbar(log_frame, orient='vertical',
                                   command=self._th_log_widget.yview)
        log_scroll.pack(side='right', fill='y')
        self._th_log_widget.configure(yscrollcommand=log_scroll.set)
        self._th_running = False
        self._th_last_out = None

    # ── Small UI helpers ────────────────────────────────────────────────
    def _th_card(self, parent, title):
        card = tk.Frame(parent, bg=AppStyles.BG_CARD,
                        highlightbackground='#30363d', highlightthickness=1)
        card.pack(fill='x', pady=6)
        tk.Label(card, text=title, bg=AppStyles.BG_CARD,
                 fg=AppStyles.ACCENT_PRIMARY,
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', padx=8,
                                                     pady=(6, 0))
        return card

    def _th_log(self, level, msg):
        icons = {'ok': '✅', 'error': '❌', 'warn': '⚠', 'info': 'ℹ',
                 'path': '📁', 'header': '━'}
        line = f"{icons.get(level, '·')} {msg}\n"

        def _append():
            try:
                self._th_log_widget.insert('end', line)
                self._th_log_widget.see('end')
            except Exception:
                pass
        try:
            self._th_log_widget.after(0, _append)
        except Exception:
            print(line, end='')

    def _th_set_status(self, text):
        try:
            self._th_status_var.set(text)
        except Exception:
            pass

    def _th_after(self, fn):
        try:
            self._th_log_widget.after(0, fn)
        except Exception:
            try:
                fn()
            except Exception:
                pass

    # ── Load from Excel ─────────────────────────────────────────────────
    def _th_browse_excel(self):
        f = filedialog.askopenfilename(
            title='Select a Case-Commentary Excel',
            filetypes=[('Excel files', '*.xlsx *.xls'), ('All files', '*.*')])
        if f:
            self._th_excel_var.set(f)
            self.update_setting('thumb_last_excel', f)

    def _th_load_excel(self):
        """Pull the Gemini thumbnail pick (frame + text) from an Excel row."""
        xls = (self._th_excel_var.get() or '').strip()
        if not xls or not Path(xls).is_file():
            messagebox.showerror('Thumbnails', 'Pick a valid Excel file first.')
            return
        try:
            import pandas as pd
        except ImportError:
            messagebox.showerror('Thumbnails',
                                 'pandas is required to read Excel.')
            return
        try:
            df = pd.read_excel(xls)
        except Exception as e:
            messagebox.showerror('Thumbnails', f'Could not read Excel:\n{e}')
            return
        if df.empty:
            messagebox.showinfo('Thumbnails', 'That Excel has no rows.')
            return

        # Prefer the first row that actually has a thumbnail pick.
        def _col(row, *names):
            for n in names:
                if n in row and pd.notna(row[n]) and str(row[n]).strip():
                    return str(row[n]).strip()
            return ''

        chosen = None
        for _, r in df.iterrows():
            if _col(r, 'Thumbnail Prompt', 'Thumbnail Text',
                    'Thumbnail Time', 'Thumbnail Frame'):
                chosen = r
                break
        if chosen is None:
            chosen = df.iloc[0]

        frame = _col(chosen, 'Thumbnail Frame')
        text = _col(chosen, 'Thumbnail Text')
        ts = _col(chosen, 'Thumbnail Time')
        # Tool 1's "recreate the original cover" columns (may be absent on
        # older Excel files, or blank when the video had no proper thumbnail).
        thumb_prompt = _col(chosen, 'Thumbnail Prompt')
        thumb_ref = _col(chosen, 'Thumbnail Ref')
        vertical = _col(chosen, 'Vertical Format').lower() in (
            'yes', 'true', '1', 'y', 'vertical', '9:16')

        # Reference image priority: the downloaded ORIGINAL cover (Thumbnail
        # Ref) beats a plain frame grab (Thumbnail Frame) — we want to recreate
        # the real cover, not a random frame.  Relative paths resolve next to
        # the Excel.
        def _resolve(p):
            if p and not Path(p).is_absolute():
                _cand = Path(xls).with_name(Path(p).name)
                if _cand.is_file():
                    return str(_cand)
            return p

        ref_img = _resolve(thumb_ref) or ''
        frame = _resolve(frame)
        if ref_img and Path(ref_img).is_file():
            self._th_ref_var.set(ref_img)
            self._th_log('ok', f'Loaded original thumbnail: {Path(ref_img).name}')
        elif frame and Path(frame).is_file():
            self._th_ref_var.set(frame)
            self._th_log('ok', f'Loaded reference frame: {Path(frame).name}')
        elif ts:
            # No frame file saved — leave the timestamp so the user can grab it
            # from the video with 📸 Grab Frame.
            self._th_time_var.set(str(self._ts_to_seconds(ts)))
            self._th_log('warn',
                         f'No frame file in Excel — set grab time to {ts}. '
                         f'Pick the video and click 📸 Grab Frame.')

        # Styling prompt: Gemini's "recreate this cover, but better" prompt
        # wins over the static default when present.
        if thumb_prompt:
            try:
                self._th_prompt_widget.delete('1.0', 'end')
                self._th_prompt_widget.insert('1.0', thumb_prompt)
                self._th_log('ok', 'Loaded Gemini styling prompt from Excel '
                                   '(recreate original cover).')
            except Exception:
                pass

        if text:
            self._th_title_var.set(text)
            self._th_log('ok', f'Loaded title text: "{text}"')

        # Aspect: Shorts → 9:16, else 16:9.
        for label, ratio in ASPECTS.items():
            if vertical and ratio == '9:16':
                self._th_aspect_var.set(label)
                break
            if not vertical and ratio == '16:9':
                self._th_aspect_var.set(label)
                break

        self._th_set_status('Loaded from Excel — review and Generate.')

    @staticmethod
    def _ts_to_seconds(ts):
        """'MM:SS' or 'HH:MM:SS' → int seconds; bare number passes through."""
        ts = str(ts).strip()
        if ':' in ts:
            parts = [int(p) for p in ts.split(':')]
            sec = 0
            for p in parts:
                sec = sec * 60 + p
            return sec
        try:
            return int(float(ts))
        except ValueError:
            return 2

    # ── Frame grab ──────────────────────────────────────────────────────
    def _th_grab_frame(self):
        video = (self._th_video_var.get() or '').strip()
        if not video or not Path(video).is_file():
            messagebox.showerror('Thumbnails', 'Pick a valid video first.')
            return
        try:
            ts = float(self._th_time_var.get())
        except ValueError:
            ts = 2.0
        self.update_setting('thumb_frame_time', self._th_time_var.get())
        src = Path(video)
        frame_out = src.with_name(f'{src.stem}_thumbframe.jpg')
        self._th_log('info', f'Grabbing frame from {src.name} @ {ts:.1f}s…')

        def _work():
            out = thumbnail_designer.extract_frame(
                src, ts, frame_out, log=self._th_log)
            if out:
                self._th_after(lambda: self._th_ref_var.set(str(out)))
                self._th_set_status(f'Frame ready: {out.name}')
            else:
                self._th_set_status('❌ Frame grab failed. See log.')
        threading.Thread(target=_work, daemon=True).start()

    # ── Run handler ─────────────────────────────────────────────────────
    def _th_start(self):
        if getattr(self, '_th_running', False):
            self._th_log('warn', 'A thumbnail is already generating — wait.')
            return

        prompt = self._th_prompt_widget.get('1.0', 'end').strip()
        if not prompt:
            messagebox.showerror('Thumbnails', 'Enter a styling prompt.')
            return
        title = (self._th_title_var.get() or '').strip()
        ref = (self._th_ref_var.get() or '').strip() or None
        video = (self._th_video_var.get() or '').strip()

        # Resolve output path.
        out = (self._th_out_var.get() or '').strip()
        if not out:
            if video and Path(video).is_file():
                out = str(Path(video).with_name(
                    f'{Path(video).stem}_thumbnail.png'))
            else:
                # Use video ID from settings as fallback
                vid_id = (self.settings.get('our_script_video_id') or '').strip()
                if vid_id:
                    out = str(Path.cwd() / f'{vid_id}_thumbnail.png')
                else:
                    out = str(Path.cwd() / 'thumbnail.png')

        model_label = self._th_model_var.get()
        model_id = thumbnail_designer.MODELS.get(
            model_label, thumbnail_designer.DEFAULT_MODEL)
        aspect = ASPECTS.get(self._th_aspect_var.get(), '16:9')

        # Fold the title into the prompt so the model renders it.
        full_prompt = prompt
        if title:
            full_prompt = (f'{prompt}\n\nRender this exact title text large and '
                           f'legible on the thumbnail: "{title}"')

        # Persist choices.
        self.update_setting('thumb_prompt', prompt)
        self.update_setting('thumb_title', title)
        self.update_setting('thumb_model', model_label)
        self.update_setting('thumb_aspect', self._th_aspect_var.get())
        self.update_setting('thumb_output', self._th_out_var.get())

        self._th_running = True
        self._th_run_btn.config(state='disabled')
        self._th_log('header', f'Generating thumbnail with {model_label}')
        self._th_set_status('Generating…')

        threading.Thread(
            target=self._th_worker,
            args=(full_prompt, out, model_id, ref, aspect), daemon=True).start()

    def _th_worker(self, prompt, out, model_id, ref, aspect):
        try:
            result = thumbnail_designer.generate_thumbnail(
                prompt, out, model=model_id, frame_path=ref, aspect=aspect,
                settings=self.settings, log=self._th_log)
            if result is not None:
                self._th_last_out = str(result)
                self._th_set_status(f'✅ Done: {Path(result).name}')
                self._th_after(lambda: messagebox.showinfo(
                    'Thumbnail complete',
                    f'Thumbnail saved to:\n{result}'))
            else:
                self._th_set_status('❌ Failed. See log.')
        except Exception as e:
            self._th_log('error', f'Unexpected error: {e}')
            for ln in traceback.format_exc().splitlines():
                self._th_log('error', f'  {ln}')
            self._th_set_status('❌ Error. See log.')
        finally:
            self._th_running = False
            self._th_after(lambda: self._th_run_btn.config(state='normal'))

    def _th_open_last(self):
        path = self._th_last_out or (self._th_out_var.get() or '').strip()
        if not path or not Path(path).is_file():
            messagebox.showinfo('Thumbnails', 'No thumbnail generated yet.')
            return
        import os
        import subprocess
        import sys
        try:
            if sys.platform.startswith('win'):
                os.startfile(path)          # noqa: S606
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', path])
            else:
                subprocess.Popen(['xdg-open', path])
        except Exception as e:
            self._th_log('error', f'Could not open image: {e}')
