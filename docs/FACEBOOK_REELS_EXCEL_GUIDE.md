# 📊 Facebook Reels Excel — Step-by-Step Guide

**What it is:** a panel inside the **🌫️ Blur Effects** tab that swaps the
standard `Quotes.txt` workflow for an **Excel-driven batch workflow**. Instead
of cycling through one text file, every video gets:

- its **caption** read from a row in your `.xlsx`
- its **voiceover MP3** auto-picked from a folder by matching the **Video ID**
- its **per-row text styling** (font, color, size, position, outline) pulled
  from the panel and used as the override for that row

This is the recommended path for **Facebook Reels production runs** because:

- It scales to hundreds of videos with zero copy-paste.
- It guarantees the right MP3 plays over the right caption.
- It lets you tweak visuals in one panel without editing text files.

---

## 0. Concepts you need to know first

| Term | Meaning |
|---|---|
| **Video ID** | A number (positive or negative) that uniquely identifies a video. It is whatever appears as the filename stem — e.g. `-111224006530392.mp4` → `-111224006530392`, `1.mp4` → `1`, `video_42.mp4` → `42`. Extracted at runtime by `ExcelIntegration.extract_video_id()` ([youtube_video_automation_enhanced.py:6094](../youtube_video_automation_enhanced.py#L6094)). |
| **Excel row** | A single row in your `.xlsx`. Row 1 is the header (skipped by pandas). Data starts on row 2. |
| **Matching mode** | `Row Index` (video 1 → row 2, video 2 → row 3, …) or `Video ID` (lookup by ID in the chosen ID column). |
| **Override** | When Excel mode is on, the panel's font / color / size / position / outline values are written into `self.settings` for the duration of that one video's render. They are *not* persisted back to disk. |

---

## 1. Prepare your Excel file

The renderer reads your file with **pandas** + **openpyxl** at startup, so any
regular `.xlsx` works.

### Recommended layout

| A (Video ID) | B (Caption) | (optional) C (Subtitle) |
|---|---|---|
| `-111224006530392` | `You only need hunger strong enough` | `— Boss Quote` |
| `581727869405526` | `Marriage is waking up every day` | `— Anonymous` |
| `42` | `The strongest souls aren't born in peace` | `— Forged in Chaos` |

Rules of thumb:
- Row 1 is the header row. Do not put data there.
- Column A is the **Video ID** by default — this must match the **stem** of
  the video filename in your source folder.
- Column B is the **Caption** by default — this is what gets painted on the
  video.
- No blank rows in the middle. Blank rows shift the row-index matching.

You can change which column is which from the panel (Text Overlay Column /
Video ID Column dropdowns) — the defaults are A=ID, B=Text.

### Save

`File → Save As → .xlsx` (NOT `.csv`). Close Excel before running the GUI;
otherwise pandas may not pick up the latest values on Windows.

---

## 2. Prepare your Audio (MP3) folder

The renderer looks for a file whose name is **`<video_id>.mp3`** (exact match
first, then `video_id in stem` as a fallback). Create a folder like:

```
E:\MyAutomations\ScriptAutomations\VideoFolder\AudioReels\
  -111224006530392.mp3
  581727869405526.mp3
  42.mp3
```

If a video has no matching MP3, the renderer logs `No audio file found for
video ID '…'` and falls back to **TTS or silence** depending on your other
settings — but the **caption** is still drawn from Excel.

---

## 3. Open the panel

1. Start the GUI.
2. Click the **🌫️ Blur Effects** tab (the tab right after *Quick Start*).
3. In the middle card labelled **📊 Facebook Reels Excel**:

   ✅ Tick **✓ Enable Excel Integration**.
4. Click **📁 Browse** next to *Excel File*, select your `.xlsx`.
5. Click **📁 Browse** next to *Audio Folder (MP3s)*, select the folder from
   step 2.
6. Pick the columns:
   - **Text Overlay Column** — letter of the column with your caption
     text. Default `B`.
   - **Video ID Column** — letter of the column with the per-video ID.
     Default `A`.
7. Pick a **Matching Mode**:
   - **By Row Index** — easier. Video N in your folder ↔ row N in Excel.
     Choose this if your videos are in the same order as the Excel rows.
   - **By Video ID** — safer. The renderer extracts the ID from the
     filename and looks it up in the ID column. Choose this when videos
     may not be in order.
8. Optionally set a default **Text Position** (top / center / bottom) and
   the rest of the styling. These become the **per-row override** values
   applied for Excel rows.
9. Click **👁️ Live Preview** in the card to see a side-by-side sample of how
   the caption will look with the current font / color / outline choices.

---

## 4. (Optional) Customize per-row text styling

All controls inside the **📊 Facebook Reels Excel** card are read **only when
the renderer is in Excel mode** ([youtube_video_automation_enhanced.py:7207-7266](../youtube_video_automation_enhanced.py#L7207)).
You can change them between runs and they are stored in
`overlay_settings.json` under keys like `excel_font_size`, `excel_text_color`,
`excel_outline_size`, etc.

Tip: if you have multiple campaigns (Reels vs. TikToks vs. YouTube Shorts) with
different fonts, save the panel as a **Template** using the right-side card on
the Quick Start tab. Then load the matching template before each run.

---

## 5. Run the pipeline

1. Go to the **🎬 Quick Start** tab.
2. Make sure the **Output Folder** points to where you want the final MP4s.
3. The `Quotes` field can be **empty** when Excel mode is on — the Excel
   sheet is the source of truth. The renderer even shows a friendly popup
   in that case ([complete_automation_gui.py:10049-10069](../complete_automation_gui.py#L10049)):
   *"Using Excel Integration mode - Quotes file not required"*.
4. Click **▶ Execute**. Watch the stdout for:
   - `[OK] Excel Integration Enabled`
   - `[OK] Ready to process N videos from Excel`
   - `[EXCEL] Found audio file: -111224006530392.mp3`
   - `Output: <caption>.mp4` (or `<source>.mp4` if you switched the
     Video Title Source to *Use source video filename* in the Quick Start
     tab).

---

## 6. What gets written into the final MP4

For every video in your folder, the renderer does:

1. **Resolve the quote** — `VideoQuoteAutomation.get_quote_for_video()`
   ([youtube_video_automation_enhanced.py:6536-6602](../youtube_video_automation_enhanced.py#L6536)).
   - If `excel_data` is loaded and `match_mode=='row_index'`, the caption is
     `self.excel_text_list[video_index]`.
   - If `match_mode=='video_id'`, it does `self.excel_data[extract_video_id(filename)]`.
2. **Resolve the audio** — `ExcelIntegration.find_audio_file(video_id, audio_folder)`
   ([youtube_video_automation_enhanced.py:6207-6247](../youtube_video_automation_enhanced.py#L6207)).
   - First tries `<video_id>.mp3` (exact match).
   - Then any MP3 whose stem contains `video_id`.
   - Returns `None` if nothing matches.
3. **Apply per-row styling override** ([youtube_video_automation_enhanced.py:7207-7266](../youtube_video_automation_enhanced.py#L7207)) — `font_size`, `font_style`, `bold`, `italic`, `text_color`, `bg_color`, `bg_opacity`, `text_outline`, `outline_color`, `outline_size`, `position` are all overwritten for the duration of the render.
4. **Run the rest of the pipeline** — same caption drawing, same region blur,
   same cleanup, same encoder. Nothing else changes.

The output filename is built by `create_filename()` with the active
**Video Title Source** (Quick Start tab → 🎞 Video Title Source):

- `Use quote text` (default) → named after the caption.
- `Use source video filename` → named after the input video's stem.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Excel Integration is enabled but Excel file is missing!` | The path in `excel_file_path` doesn't exist. | Re-pick the file in the panel. |
| `[ERROR] Video index N exceeds Excel rows (M)` | You have more videos than rows. | Add more rows to Excel, or remove videos from the folder. |
| `[ERROR] Video ID 'X' not found in Excel data` (Video ID mode) | The extracted ID doesn't appear in the ID column. | Open the Excel sheet, check the ID column matches the filename stems. The `extract_video_id` function returns the **last** number in the stem. |
| `[EXCEL] No audio file found` | Audio folder is wrong / file name doesn't match. | Rename MP3 to `<video_id>.mp3`, or check the Audio Folder path. |
| Caption font / color not changing | Excel mode is off, or the panel setting got reset by a Template load. | Tick `Enable Excel Integration`, re-save the panel as a template, and make sure you didn't pick a template that sets `excel_font_size=0`. |
| Renderer uses `Quotes.txt` instead | `excel_integration_enabled` is `False` in `overlay_settings.json`. | Tick the checkbox; restart if it was loaded before the change. |
| Pandas error at startup | `pandas` / `openpyxl` not installed. | `pip install pandas openpyxl`. |

---

## 8. End-to-end quickstart (TL;DR)

```
1.  Build the .xlsx:
      A: video_id   B: caption
      -111224006530392  You only need hunger strong enough
      581727869405526   Marriage is waking up every day

2.  Put the MP3s in a folder:
      E:\…\AudioReels\-111224006530392.mp3
      E:\…\AudioReels\581727869405526.mp3

3.  Open GUI → 🌫️ Blur Effects tab → 📊 Facebook Reels Excel card
      ☑ Enable Excel Integration
      📁 pick the .xlsx
      📁 pick the AudioReels folder
      A=ID column, B=text column
      ☑ By Video ID

4.  Optional: tweak font / color / outline in the same card.

5.  Quick Start tab → Execute. Final videos land in your Output Folder.
```

That's it — Excel drives both the caption and the audio for every video in
the run.
