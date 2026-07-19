#!/usr/bin/env python3
"""Generate a 1-2 page feature-summary PDF for the Video Automation Tool."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos

class FeaturePDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-10)
        self.set_font(_F, 'I', 7)
        self.set_text_color(160, 160, 160)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')


pdf = FeaturePDF(orientation='P', unit='mm', format='A4')
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=18)
# Register a Unicode TTF font so em-dashes, apostrophes etc. render
pdf.add_font('Arial', '', r'C:\Windows\Fonts\Arial.ttf')
pdf.add_font('Arial', 'B', r'C:\Windows\Fonts\Arialbd.ttf')
pdf.add_font('Arial', 'I', r'C:\Windows\Fonts\Ariali.ttf')
pdf.add_font('Arial', 'BI', r'C:\Windows\Fonts\Arialbi.ttf')
_F = 'Arial'   # font family shortcut
pdf.add_page()

# ── Colors ─────────────────────────────────────────────────────────
C_DARK   = (20,   22,  28)
C_ACCENT = (108, 135, 255)   # soft indigo
C_WHITE  = (255, 255, 255)
C_MUTED  = (120, 125, 135)
C_LINE   = (55,   58,  65)
w = pdf.w - pdf.l_margin - pdf.r_margin

# ── HERO BAND ───────────────────────────────────────────────────────
pdf.set_fill_color(*C_DARK)
pdf.rect(0, 0, 210, 84, 'F')

pdf.set_y(14)
pdf.set_font(_F, 'B', 26)
pdf.set_text_color(*C_ACCENT)
pdf.cell(0, 10, 'VIDEO AUTOMATION STUDIO', align='C', new_x=XPos.LMARGIN)
pdf.ln(12)

pdf.set_font(_F, 'B', 16)
pdf.set_text_color(*C_WHITE)
pdf.multi_cell(0, 7.5,
    'Almost Impossible.\nYou Will Never Need a Video Editor Again.',
    align='C')
pdf.ln(4)

pdf.set_font(_F, '', 9.5)
pdf.set_text_color(*C_MUTED)
pdf.multi_cell(0, 5,
    'One click — quotes, transitions, voiceovers, captions, effects, AI lip-sync, cleaning.\n'
    'Two years of relentless iteration. From raw footage to platform-ready content in minutes.',
    align='C')
pdf.ln(4)

# separator line
pdf.set_draw_color(*C_LINE)
pdf.set_line_width(0.3)
pdf.line(pdf.l_margin + 35, pdf.get_y(), pdf.l_margin + w - 35, pdf.get_y())
pdf.ln(6)

# ── HELPERS ─────────────────────────────────────────────────────────
def section(title):
    pdf.set_font(_F, 'B', 10.5)
    pdf.set_text_color(*C_ACCENT)
    pdf.cell(0, 6.5, title.upper(), new_x=XPos.LMARGIN)
    pdf.ln(6)
    pdf.set_draw_color(*C_LINE)
    pdf.set_line_width(0.2)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + w, pdf.get_y())
    pdf.ln(2.5)


def bullet(bold, rest=''):
    """One bullet: bullet-char, bold name, rest text."""
    pdf.set_font(_F, '', 8)
    pdf.set_text_color(*C_ACCENT)
    pdf.cell(3.5, 4.4, '•')
    pdf.set_font(_F, 'B', 8)
    pdf.set_text_color(*C_WHITE)
    bw = pdf.get_string_width(bold)
    pdf.cell(bw + 1, 4.4, bold)
    if rest:
        pdf.set_font(_F, '', 8)
        pdf.set_text_color(205, 208, 215)
        used = 3.5 + bw + 2
        pdf.multi_cell(w - used, 4.4, rest, new_x=XPos.LMARGIN)
    else:
        pdf.ln(4.4)


def blist(items_):
    for i, (b, r) in enumerate(items_):
        bullet(b, r)
        # auto page-break if too close to bottom
        if i < len(items_) - 1 and pdf.get_y() > 258:
            pdf.add_page()
            pdf.set_y(pdf.get_y() + 8)


# ── CONTENT ─────────────────────────────────────────────────────────

section('Core Pipeline — One Click, Full Video')
blist([
    ('Auto Quote Engine', 'transcript or Excel → timed text overlays with per-niche prompts (movies, courtroom, heartwarming, voiceover)'),
    ('23+ Transitions', 'Fade, Zoom, Blur, Slide, Wipe, Glitch, Bounce, Mask Reveal, Radial Wipe, Split Wipe, Luma Wipe, Cinematic Bars, Color Dissolve — each with 15+ SFX sounds'),
    ('Smart CTA System', 'auto call-to-action with spoken/shown label stripping, multi-line wrapping, freeze-frame tail extension'),
    ('Scene-by-Scene Presets', 'per-niche script modes (Rewrite / Write Story) with dual prompt-store architecture'),
    ('Multi-Platform Output', 'Instagram Reels, TikTok, YouTube Shorts, YouTube, Facebook — custom resolution & crop modes'),
])

section('Artificial Intelligence')
blist([
    ('4 TTS Engines', 'Microsoft Edge Cloud (41+ voices, 17 locales), Kokoro ONNX (42 voices), Qwen3-TTS (emotion + voice cloning), NeuTTS'),
    ('Gemini Case Commentary', 'watches courtroom/movie video → structured Summary + Montage Clips + Commentary Spots → auto-builds the montage'),
    ('Whisper Speech-to-Text', 'word-level timing for TTS, dialogue captions from original audio, overlap suppression with voiceover'),
    ('Wav2Lip AI Avatar', 'lip-sync any face image to speech, overlay or standalone mode, position & size controls'),
    ('LLM Rerank', 'optional Gemini / OpenAI / Anthropic reranking over heuristic scene-cut scoring'),
])

section('Captions — Three Independent Stacks')
blist([
    ('Simple Captions', 'style presets, font/color/size, background, stroke, position, sync offset, emoji presets'),
    ('Highlight Captions (CapCut Style)', 'word-by-word colour-highlight animation, independent font & animation type'),
    ('Dialogue Captions (Whisper)', 'auto-transcribe original audio, third caption layer, smart word-level overlap suppression'),
    ('Live Preview', 'phone-aspect canvas with toggle layers, scrubber, crosshair alignment guide'),
])

section('Audio Engineering')
blist([
    ('BGM System', 'file or folder, volume, auto-loop, intelligent ducking that drops during speech'),
    ('Voiceover Ducking', 'per-frame amplitude check — original audio drops to 15% during TTS segments'),
    ('Voice Effects', 'Deep / High / Robot / Echo / Whisper / Radio / Chipmunk + SSML speaking styles'),
    ('Audio Normalization', 'consistent loudness with configurable dB target'),
    ('Auto Silence Removal', 'threshold, min-duration, transition smoothing'),
])

section('Visual Effects')
blist([
    ('Particle Systems', 'Glitter, Stars, Hearts, Confetti — all with animated motion paths'),
    ('Light Effects', 'Light Leaks, Lens Flare, Film Burn — configurable intensity & repeat interval'),
    ('Blur Engine', 'region blur, custom rectangular blur regions, feather edge per-pixel control'),
    ('Alight Motion Look Builder', '14+ AM-style presets: hue, saturation, gamma, shadows, VHS, scanlines, RGB split, bloom, colorama, grain, vignette, film burn & more'),
    ('20+ Overlay Elements', 'progress bar, watermark, vignette, dim, grain, drop shadow, glitch, chromatic aberration, text glow, neon glow, gradient, border, crosshair, spotlight, dual-video'),
])

section('Performance — GPU Renderer')
blist([
    ('OpenCV + NVENC Pipeline', 'auto-detects NVENC (RTX 2060+) vs CPU. 30 fps 1080p at 50–100 fps vs 1–2 fps MoviePy fallback.'),
    ('Phase 2 Optimizations', 'AV1 seek optimisation, static overlay pre-merge, density-adaptive blend, vectorised feather cache. 3-min render: ~60 min → ~7 min.'),
])

section('Post-Processing & Cleanup')
blist([
    ('Quick Wipe', 'instant platform-metadata strip, no re-encode'),
    ('Full Spoof', 'deep re-encode with mirror, smart zoom, unique colour profile, pitch-shifted audio, borders'),
    ('Cleanup Effects', 'speed/rotation, letterbox, colour wash, watermark, RGB glitch, Ken Burns zoom, reverb, transitions re-apply'),
    ('YouTube Uploader', 'upload processed videos directly from the app'),
])

section('Specialised Modules')
blist([
    ('Our Script Tab', 'channel-aware single-video pipeline, per-channel presets, batch + skip-done'),
    ('Xiaohongshu / RedNote Scraper', 'Selenium + CDP API interception for hidden note IDs, cookies.txt auth, cursor pagination'),
    ('Instagram Auth', 'unified cookies.txt for instaloader + yt-dlp, auto re-login on session expiry'),
    ('AI Avatar Tab', 'full Wav2Lip pipeline: face image, TTS audio, lip-sync, composite back onto video'),
])

section('Quality of Life')
blist([
    ('Dark-Theme GUI', 'scrollable canvases, 10 notebook tabs, color pickers everywhere, live preview'),
    ('Preset Manager', 'save/load/delete full effect combinations as named templates'),
    ('Platform Quick-Select', 'one-click resolution presets with live aspect-ratio display'),
    ('Settings Persistence', 'overlay_settings.json + processing_paths.json auto-save/load'),
    ('Portable Copy', 'standalone deployment ready on any machine'),
])

# ── CLOSING ─────────────────────────────────────────────────────────
pdf.ln(3)
pdf.set_draw_color(*C_LINE)
pdf.set_line_width(0.3)
pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + w, pdf.get_y())
pdf.ln(4)

pdf.set_font(_F, 'I', 9)
pdf.set_text_color(*C_MUTED)
pdf.multi_cell(0, 4.5,
    'Four TTS engines, five AI integrations, 23 transitions, three caption stacks, '
    'a GPU render pipeline, and a cleaning engine that strips every fingerprint.\n'
    'All behind one button.',
    align='C')
pdf.ln(4)

pdf.set_font(_F, 'B', 12)
pdf.set_text_color(*C_ACCENT)
pdf.cell(0, 7, 'One Click. Ready to Post.', align='C', new_x=XPos.LMARGIN)

# ── SAVE ───────────────────────────────────────────────────────────
out = r'd:\GitHub\ChangeGUI\Video_Automation_Studio_Features.pdf'
pdf.output(out)
print(f'PDF saved to {out}')
print(f'Pages: {pdf.page_no()}')
