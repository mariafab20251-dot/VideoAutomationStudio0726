# 🎙️ OurScript Tab — Step-by-Step Guide

**What it is:** a panel inside the **🎙️ OurScript** tab (between *🌫️ Blur
Effects* and *🎵 Audio*) that lets you process **one video at a time** out of
a downloaded YouTube channel folder. You pick the channel, pick a video id, the
tab reads the right row from your **`results_clean` Excel** file, swaps the
on-screen caption for your own text, mixes a voiceover (either an MP3 you
already have or a freshly-generated TTS), and writes the result into a
channel-local **output / done** folder pair so you always know what's left.

This is the panel you want when:
- you have a tool that downloads YouTube channels into
  `channels/youtube/<ChannelName>/` with `videos/`, `audio/`, and a
  `results_clean <date>.xlsx`; and
- you want to put your **own caption** on each video, optionally with your
  **own voiceover**, in a single click per video.

> Source of truth for the render path:
> `_os_run_single()` and `_os_render_one_video()` in
> [complete_automation_gui.py](../complete_automation_gui.py) (the OurScript
> tab's two `_*` methods). The renderer composes existing settings from the
> 💬 Captions, 🌫️ Blur Effects, 🎨 Cleanup, and 🗣 TTS tabs.

---

## 0. Concepts you need to know first

| Term | Meaning |
|---|---|
| **Channel Folder** | A directory like `channels/youtube/The_Law_Office_of_Dennis_R__Vetrano/` that contains `videos/`, optionally `audio/`, and one or more `results_clean*.xlsx` files. |
| **Video ID** | The YouTube ID (e.g. `mxcMthreXwE`). The renderer finds `<channel>/videos/<id>.mp4` and matches the row in the `results_clean` xlsx. |
| **`results_clean` xlsx** | The Excel file produced by the downloader. Its columns are `Video ID, Platform, URL, Overlay Text, Speech Transcript, Captions, Hashtags, Date Processed`. **Only this file is read** — `metadata_*.xlsx` is ignored by the OurScript tab. |
| **Caption Source** | Which Excel column holds the text you want to paint on the video. Default `Captions`. You can switch to `Overlay Text` or `Speech Transcript` from the dropdown. |
| **VoiceOver Mode** | One of four: `channel_mp3` (use the audio the downloader already put in `<channel>/audio/`), `folder_mp3` (use a different folder you point at), `tts` (generate an MP3 from the caption text using the TTS tab's engine), or `none` (keep the original video audio). |
| **Manual override** | The text you type into the OurScript text box. If non-empty, it wins over whatever the Excel column said. Leave it blank to use the column value. |
| **Output folders** | The tab auto-creates `<channel>/_processed_output/` (final videos) and `<channel>/_processed_done/` (moved source mp4s) on first run. The second folder is your "what's left to process" ledger. |

---

## 1. Prepare your channel folder

The expected layout is whatever the downloader produces:

```
D:\GitHub\pythonprojects\VideoTextExtractor\channels\youtube\
  The_Law_Office_of_Dennis_R__Vetrano\
    videos\
      mxcMthreXwE.mp4
      abcDEF1234.mp4
      ...
    audio\                          (optional, only if downloader produced it)
      mxcMthreXwE.mp3
      ...
    results_clean 22.xlsx           (or any results_clean*.xlsx)
    metadata_*.xlsx                 (ignored by OurScript)
    urls_*.txt                      (ignored)
    reports\                        (ignored)
```

Rules:
- The xlsx file must start with `results_clean` (case-insensitive) and end
  with `.xlsx`. The newest one by mtime wins.
- The `Video ID` column is matched case-insensitively, so `Video ID`,
  `video_id`, `videoid` all work.
- The video file must live at `<channel>/videos/<video_id>.mp4` — exact name
  match. If it's missing, the tab shows a clear error and refuses to run.
- The voiceover MP3 (in `channel_mp3` mode) must live at
  `<channel>/audio/<video_id>.mp3`. If it's missing, the tab silently falls
  back to the folder you configured for `folder_mp3` (if any).

---

## 2. Open the tab

1. Start the GUI (`python complete_automation_gui.py`).
2. Click the **📜 OurScript** tab (between *🌫️ Blur Effects* and *🎵 Audio*).
3. The tab uses a **two-pane split** so everything fits on one screen
   without long scrolling:
   - **Left ~70%** (scrollable) — Channel & Video, OurScript Caption,
     VoiceOver, Blur & Border, Blur Text Overlay, Title Text, Run / Batch
   - **Right ~30%** (fixed) — 👁 Live Preview (top) + 📋 OurScript Log
     (bottom), separated by a draggable sash
   The vertical sash between the panes is draggable — pull it left/right
   to give the controls more room if you have a wide monitor.
4. In the **📁 Channel & Video** card:
   - **Channel Folder** — click **📁 Browse** and pick the channel directory
     described in step 1. The path is stored in
     `settings['our_script_channel_path']`.
5. Click **🔄 Refresh Video List**. The **🎬 Video ID** dropdown is
   populated with every `video_id` it can find in the latest
   `results_clean*.xlsx`.

> The dropdown also filters to **videos that actually exist in
> `videos/`**, so a stale row in the xlsx never shows up as a runnable
> option.

---

## 3. Pick a video and inspect its row

1. Pick a Video ID from the dropdown. The right-hand **🔎 Preview** card
   fills in:
   - **Title** — the value of the selected `Caption Source` column from
     `results_clean`.
   - **Source video** path — `<channel>/videos/<id>.mp4`.
   - **Available MP3** — the path the voiceover resolver will pick (or
     *none*, depending on mode).
2. The big text box under the dropdown shows the resolved caption text
   (whatever the Excel column said). **Edit it if you want a manual
   override** — non-empty text here wins over the Excel value.

### Caption Source dropdown
- `Captions` (default) — the cleaned, time-aligned captions column.
- `Overlay Text` — the text the original video paints on the frame.
- `Speech Transcript` — the raw speech-to-text transcript.

You can change this freely between runs — it's read on click of
**▶ Run This Video**.

---

## 4. Pick a voiceover mode

The **🎙️ VoiceOver** card has a 4-way radio. Pick the one that fits your
workflow:

| Mode | What it does | When to use |
|---|---|---|
| **channel_mp3** (default) | Looks for `<channel>/audio/<video_id>.mp3`. Falls back to the **Folder MP3** path if not found there. | You let the downloader produce audio and want to use it as-is. |
| **folder_mp3** | Looks for `<user folder>/<video_id>.mp3`. The path is configured in the sub-panel that appears under the radio. | You have a separate library of MP3s (e.g. one folder per campaign). |
| **tts** | Generates an MP3 from the resolved caption text using the **🗣 TTS** tab's engine/voice/rate. The MP3 is cached at `<channel>/_tts_cache/<video_id>.mp3` so re-runs don't re-synthesize. | You want a fresh TTS read of the caption, using whatever TTS engine you already configured. |
| **none** | Keeps the source video's original audio. | You only want to repaint the caption and don't want to touch the audio. |

Each mode shows a sub-panel with the path / settings it needs. The sub-panel
appears as soon as you select the mode (the others are hidden).

---

## 5. Style the caption (pointers, not duplicates)

The OurScript tab **doesn't re-invent** the caption controls. The
on-screen caption uses the **exact same** settings the 💬 Captions tab
exposes. To tweak font, color, size, position, or **background color**:

> **Go to the 💬 Captions tab.** All the styling you set there —
> `caption_text_color`, `caption_bg_color`, `caption_bg_opacity`,
> `caption_bg_enabled`, `caption_font_family`, `caption_font_size`,
> `caption_position` — is what the OurScript render will use.

The OurScript tab shows a small **ⓘ** hint right above the Run button
reminding you of this. We don't duplicate the controls so you never have
to wonder "which one wins?" — there's only one source of truth.

---

## 6. Enable blur & border (optional)

The **🌫️ Blur & Border** card at the top of the tab is a quick toggle
panel for the two effects most users want to add per-video:

- **🎯 Quick Blur** — `region_blur_enabled` + a count of *active* custom
  blur regions. If any custom region is `enabled=True` in the 🌫️ Blur
  Effects tab, the count shows that number and the renderer applies them.
- **🖼 Border Color** — uses `cleanup_border_enabled` and
  `cleanup_border_color` from the 🎨 Cleanup tab. Tick the checkbox and
  pick a color.

Everything else (custom blur regions themselves, region shape, intensity,
feather, etc.) lives in the 🌫️ Blur Effects tab — this card is just the
"on/off" switch for the OurScript run.

---

## 7. Run it

1. Confirm **✅ Enable OurScript** is ticked (it is by default).
2. Click **▶ Run This Video** (or **📦 Process Full Channel** — see §7b).
3. Watch the console:

   ```
   [OurScript] Channel : D:\...\The_Law_Office_of_Dennis_R__Vetrano
   [OurScript] Video   : mxcMthreXwE.mp4
   [OurScript] Caption : The strongest souls aren't born in peace...
   [OurScript] Voice   : mxcMthreXwE.mp3
   [OurScript] Output  : D:\...\_processed_output\The strongest souls....mp4
   [OurScript] Writing D:\...\_processed_output\The strongest souls....mp4 ...
   [OurScript] ✅ Done — output: D:\...\_processed_output\The strongest souls....mp4
   [OurScript] 📦 Source moved to: D:\...\_processed_done\mxcMthreXwE.mp4
   ```

4. The original `mxcMthreXwE.mp4` is moved from `videos/` into
   `_processed_done/` — that's how you tell what's left to process at a
   glance: anything still in `videos/` is unprocessed; anything in
   `_processed_done/` is done.

---

## 7b. Process the whole channel in one click

When you don't want to babysit the dropdown, the **📦 Process Full Channel**
button (under the green **🚀 Process Selected Video** button, in the
**📁 Channel & Video** card) walks through every video in the channel for
you. It reuses **every** setting above — voiceover mode, caption source,
blur/border, title — so the batch is exactly as many single-video runs
chained together, in the **Excel row order** the dropdown already shows.

### Layout

```
📁 Channel & Video
   Channel Folder:  [______________________________] [📁] [🔄]
   Video ID:        [▼ abc123XYZ                      ]
   Title:           …

   [ 🚀 Process Selected Video ]

   📦 Batch — process every unprocessed video in this channel:
   [ 📦 Process Full Channel ]   [ ⏹ Stop ]

   ☐ ⏭ Skip videos already in _processed_output/
   If no MP3 in channel_mp3 mode: [▼ skip ▾]
   Running 3/12: mxcMthreXwE     ← live status
```

### Batch options

| Option | What it does | Default |
|---|---|---|
| **⏭ Skip videos already in `_processed_output/`** | If a video's source is no longer in `videos/` (i.e. it was processed in a previous run), skip it. | ✅ on |
| **If no MP3 in channel_mp3 mode** | Policy when a video has no `<channel>/audio/<id>.mp3`: <br>• `skip` — leave it for later, continue with the next video<br>• `abort` — stop the whole batch on the first missing audio<br>• `tts` — auto-switch just that video to TTS mode<br>• `none` — auto-switch just that video to "no voiceover" | `skip` |

Both options persist into `settings.json` so they survive a restart.

### Live status & log

- The grey italic line under the buttons shows live progress, e.g.
  `Running 3/12: mxcMthreXwE`. When the batch is done it becomes
  `Idle — last batch: 10 ok, 1 skipped, 1 failed`.
- The **📋 OurScript Log** panel at the bottom of the tab prints a
  per-video header so you can scan it later:

  ```
  [14:23:01] ═══ BATCH START: 12 video(s) ═══
  [14:23:01] ── Batch 1/12: mxcMthreXwE ──
  [14:23:01] ═══ Run started: video_id=mxcMthreXwE ═══
  …
  [14:23:18] ═══ Run complete: mxcMthreXwE ═══
  [14:23:18] ── Batch 2/12: abcDEF1234 ──
  …
  [14:25:42] ═══ BATCH DONE: 10 ok, 1 skipped, 1 failed (of 12) in 161.2s ═══
  [14:25:42] [ERROR]    Failed video_ids: 7646032404536118550
  ```

### ⏹ Stop button

Click **⏹ Stop** at any time — the batch finishes the **current** video
cleanly (so you don't get a half-written MP4) and then halts. The status
bar updates immediately and the buttons re-enable.

### When the user clicks the single-video button during a batch

The dropdown's `<<ComboboxSelected>>` handler still works (it only edits
the preview). The actual run is owned by the worker thread, so picking
a different Video ID from the dropdown while a batch is running does
**not** disrupt the batch — the batch sets the Video ID itself before
each call to `_os_run_single()`.

### Use case: bake a whole channel with a TTS voiceover

1. Pick the channel folder.
2. Set VoiceOver mode to **🗣️ TTS**.
3. Set On-missing-audio to **`tts`** (so a missing MP3 still gets a voice).
4. Click **📦 Process Full Channel**.
5. Walk away — the log panel will print a `BATCH DONE` summary line
   with `ok / skipped / failed` counts when it finishes.

---

## 8. What gets written into the final MP4

For the selected video, the OurScript render does this in order:

1. **Open the source** with MoviePy (`VideoFileClip`).
2. **Apply region blur** if `region_blur_enabled=True` or any
   `custom_blur_regions[i].enabled=True` — same per-frame hook used by
   the main pipeline
   ([`VideoEffects.apply_region_blur`](../youtube_video_automation_enhanced.py#L1325)).
3. **Apply border color** if `cleanup_border_enabled=True` — pads the
   frame with the chosen color to the configured thickness.
4. **Apply the caption** using the resolved text + the 💬 Captions tab's
   `caption_text_color` / `caption_bg_color` / `caption_bg_opacity` /
   `caption_bg_enabled` / `caption_font_family` / `caption_font_size` /
   `caption_position`. The text is also promoted to `caption_text`,
   `subtitle`, `voiceover_text`, and `quote` in `settings` so any other
   pipeline that reads those keys sees it.
5. **Mix the voiceover**:
   - If the resolved mode produces a file path → replace (or add) audio
     on the clip with that MP3, trimmed to the clip duration.
   - If mode is `none` → keep the source's original audio untouched.
6. **Write** to `<channel>/_processed_output/<sanitized caption>.mp4` with
   `libx264` video + `aac` audio. Filename falls back to `<video_id>.mp4`
   if the caption is empty.
7. **Move** the source mp4 from `videos/` into `_processed_done/` with
   `Path.replace()` — atomic on Windows. Sibling files in `videos/`
   aren't touched.

The output filename uses the **caption text** (sanitized for Windows
filenames) when available, so the folder of completed videos is
human-readable: every filename is a sentence the user will see on
screen.

---

## 9. TTS cache hygiene

When you run in **tts** mode, the generated MP3s go into
`<channel>/_tts_cache/<video_id>.mp3`. They are reused on subsequent
runs of the same video — the renderer doesn't re-synthesize. To force
a fresh read:

- delete the file in `_tts_cache/`, **or**
- change the engine/voice/rate in the **🗣 TTS** tab before clicking
  **▶ Run This Video** again.

The `_tts_cache` folder is created on first use and never auto-deleted
by the OurScript tab — clean it up manually if you change voices and
want a hard reset.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| *"Please pick a Channel Folder and a Video ID first"* | One of the two is empty. | Pick a channel folder, click **🔄 Refresh Video List**, then pick a video. |
| *"Video file not found: <path>"* | The mp4 isn't at `<channel>/videos/<id>.mp4`. | The dropdown is filtered to existing files, but the file may have been moved manually. Re-pick a video id or restore the file. |
| VoiceOver dropdown says "None" but I picked a folder | The folder path is wrong or `<id>.mp3` doesn't exist there. | Click into the **Folder MP3** sub-panel, confirm the path, and verify the file exists. |
| TTS run printed "No text for <id>" | The resolved caption text is empty (the Excel column was blank **and** the manual override is empty). | Type something in the text box, or pick a different `Caption Source` column. |
| Output filename looks like `<id>.mp4` instead of a sentence | The resolved caption text was empty. | Type a manual override, or fix the Excel row so the chosen column has content. |
| *"Output written, but could not move source"* | The source mp4 was open in another process (e.g. a video player). | Close the player and click **▶ Run This Video** again. The output is safe; only the move failed. |
| My caption has no background color on the final video | `caption_bg_enabled` is `False` in the 💬 Captions tab. | Go to 💬 Captions, tick **Background**, pick a color and opacity, then re-run. |
| The caption color doesn't match what I set in the OurScript text box | The OurScript tab doesn't expose text color — it's read from the 💬 Captions tab so you only have one place to set it. | Change `caption_text_color` in the 💬 Captions tab. |

---

## 11. End-to-end quickstart (TL;DR)

```
1.  Point the tab at your channel folder
       🎙️ OurScript → 📂 Channel → 📁 Browse → <channel dir>
       🔄 Refresh Video List

2.  Pick a video id from the dropdown.
       (preview fills in on the right)

3.  Pick a voiceover mode:
       ● channel_mp3  → use <channel>/audio/<id>.mp3
       ○ folder_mp3   → point at any folder of <id>.mp3 files
       ○ tts          → generate from the caption text
       ○ none         → keep the original audio

4.  (Optional) Type a manual caption override in the big text box.

5.  (Optional) Style the caption in the 💬 Captions tab — that's where
    text color, font, size, position, and background color live.

6.  ▶ Run This Video.

7.  Find the result in  <channel>/_processed_output/
    Find the moved source in  <channel>/_processed_done/
```

That's it — one click per video, with all the data coming from the
`results_clean` Excel you already have.
