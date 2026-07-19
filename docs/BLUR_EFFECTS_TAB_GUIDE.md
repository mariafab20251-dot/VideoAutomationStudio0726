# 🌫️ Blur Effects Tab — Panel-by-Panel Guide

This document walks through every panel inside the **🌫️ Blur Effects** tab of the
GUI, in the order they appear (left → right, top → bottom). For each control it
states **(a) what it does, (b) whether it is actually wired into the render path,
and (c) the visual effect on the final video**.

> Source of truth for renderer behaviour: `VideoEffects.apply_region_blur()`
> in [youtube_video_automation_enhanced.py:1325](../youtube_video_automation_enhanced.py#L1325)
> and the per-frame hook at
> [youtube_video_automation_enhanced.py:7774-7788](../youtube_video_automation_enhanced.py#L7774).
>
> Source of truth for the UI: `create_blur_tab()` in
> [complete_automation_gui.py:2841](../complete_automation_gui.py#L2841).

---

## Layout (3-column grid + one full-width card)

```
Row 0 ┌──────────────────┬──────────────────┬──────────────────┐
      │ 🌫️ Region Blur   │ 📊 Facebook Reels│ 📝 Blur Text     │
      │                  │    Excel         │    Overlay       │
      ├──────────────────┴──────────────────┴──────────────────┤
Row 1 │ 🎯 Custom Blur Regions (Hide Logos) — colspan 3        │
      └─────────────────────────────────────────────────────────┘
```

---

## 1. 🌫️ Region Blur (Row 0, Col 0) — "preset shape" card

A **predefined shape** of the frame gets blurred in a single region. Useful for
the bottom black-bar look or to push attention to a center crop.

> 💡 **This card and 🎯 Custom Blur Regions below do the same physical thing
> (cv2.GaussianBlur on a rectangle) but solve different problems:**
>
> | Use this card (Region Blur) when you want… | Use the Custom Blur Regions card when you want… |
> |---|---|
> | A single, **preset** shape (top / bottom / left / right / center / top_bottom / left_right) | Multiple rectangles at **arbitrary X/Y/W/H** positions |
> | A quick "bottom black bar" or "lower third" with one switch | To **hide a specific logo / watermark** at a pixel position |
> | A single tint + feather that applies to the band | Per-region text, font, color, intensity, optional Excel text |
>
> Both can be enabled at the same time — they fire in the same render pass
> ([youtube_video_automation_enhanced.py:1376-1490](../youtube_video_automation_enhanced.py#L1376)) and do not conflict.

| Control | Setting key | Status | Render impact |
|---|---|---|---|
| `Enable` checkbox | `region_blur_enabled` | ✅ wired | Turns the whole card on. If unchecked, the renderer skips the entire region-blur branch ([youtube_video_automation_enhanced.py:1328-1333](../youtube_video_automation_enhanced.py#L1328)). |
| `Region` combobox (`top / bottom / left / right / center / top_bottom / left_right`) | `blur_region` | ✅ wired | Picks which edge(s) of the frame are blurred. The renderer computes pixel rectangles per region ([youtube_video_automation_enhanced.py:1355-1374](../youtube_video_automation_enhanced.py#L1355)). |
| `Size` slider (10–50 %) | `blur_region_size` | ✅ wired | How far the blurred band extends from the edge (or how big the center square is). |
| `Intensity` slider (1–100) | `blur_intensity` | ✅ wired | Strength of the Gaussian blur. Internally `kernel_size = intensity * 2 + 1` ([youtube_video_automation_enhanced.py:1343-1345](../youtube_video_automation_enhanced.py#L1343)). |
| `Color Tint` checkbox | `blur_color_tint_enabled` | ✅ wired | After blurring, the band is blended with a solid color for an extra stylized look. |
| `🎨 Tint color` (hex) | `blur_tint_color` | ✅ wired | Default `#000000` (gives the classic "black-bar" cap). |
| `Tint` opacity slider (0–100 %) | `blur_tint_opacity` | ✅ wired | How strong the tint overlay is. |
| `Feather Edge` checkbox | `blur_feather_edge` | ✅ wired | Softens the boundary between blurred and sharp zones with a 1/4-region feather mask ([youtube_video_automation_enhanced.py:1466-1488](../youtube_video_automation_enhanced.py#L1466)). Note: only applies to **predefined** regions, not custom ones. |

**Steps to use**
1. Tick **Enable**.
2. Pick a **Region** (e.g. `bottom`).
3. Move **Size** to ~30 % and **Intensity** to ~15 for a subtle bar.
4. Optionally enable **Color Tint** (`#000000`, opacity 50 %) for a darker
   bottom cap, or change the color for a colored "news-lower-third" look.
5. Leave **Feather Edge** on so the blur dissolves into the video smoothly.

**Render impact** — every frame's edges are Gaussian-blurred (cv2) with the
chosen color tint and feather mask. Adds ~5-15 % to per-frame processing time
depending on intensity; the rest of the pipeline is unchanged.

---

## 2. 📊 Facebook Reels Excel (Row 0, Col 1)

Replaces the standard **Quotes.txt** workflow with an **Excel-driven** one. The
Excel file is the single source of truth for: (i) the on-screen overlay text
per video, (ii) which MP3 voiceover to use, and (iii) the per-row text styling
override. This is the panel you want for **Facebook Reels batch production**.

> Full step-by-step walkthrough is in
> [docs/FACEBOOK_REELS_EXCEL_GUIDE.md](FACEBOOK_REELS_EXCEL_GUIDE.md).

| Control | Setting key | Status | Render impact |
|---|---|---|---|
| `Enable Excel Integration` | `excel_integration_enabled` | ✅ wired | When off, the renderer uses `Quotes.txt` as before. When on, `Quotes.txt` is **ignored** ([youtube_video_automation_enhanced.py:10049-10069](../youtube_video_automation_enhanced.py#L10049)). |
| `Excel File` path + 📁 Browse | `excel_file_path` | ✅ wired | The `.xlsx/.xls` to load with pandas at startup ([youtube_video_automation_enhanced.py:6301-6322](../youtube_video_automation_enhanced.py#L6301)). |
| `Text Overlay Column` (A–H) | `excel_text_column` | ✅ wired | Which Excel column holds the quote / caption text. Default `B`. |
| `Video ID Column` (A–H) | `excel_id_column` | ✅ wired | Which Excel column holds the per-video ID used to match an MP3 in the audio folder. Default `A`. |
| Matching Mode radio (`Row Index` / `Video ID`) | `excel_match_mode` | ✅ wired | `row_index` = video N ↔ row N (simple, just process the videos in folder order). `video_id` = use `ExcelIntegration.extract_video_id` on the filename and look it up in the Excel column ([youtube_video_automation_enhanced.py:6539-6602](../youtube_video_automation_enhanced.py#L6539)). |
| `Audio Folder (MP3s)` path | `excel_audio_folder` | ✅ wired | Folder containing per-video MP3s named `<video_id>.mp3` (e.g. `-111224006530392.mp3`). `ExcelIntegration.find_audio_file` does an exact match first, then a partial `video_id in stem` fallback ([youtube_video_automation_enhanced.py:6207-6247](../youtube_video_automation_enhanced.py#L6207)). |
| `Text Position` radio (top/center/bottom) | `excel_text_position` | ✅ wired | Overrides the global `position` setting **for this row's caption** ([youtube_video_automation_enhanced.py:6588, 7211-7214](../youtube_video_automation_enhanced.py#L6588)). |
| `Font Size` slider (20–100) | `excel_font_size` | ✅ wired | Sets both `font_size` and `quote_font_size` for that row ([youtube_video_automation_enhanced.py:7217-7221](../youtube_video_automation_enhanced.py#L7217)). |
| `Font Family` combobox | `excel_font_style` | ✅ wired | e.g. `Arial Bold`, `Impact`, `Verdana`, `Comic Sans MS`, `Times New Roman`. |
| `Bold` / `Italic` checkboxes | `excel_bold`, `excel_italic` | ✅ wired |  |
| `Text Color` color picker | `excel_text_color` | ✅ wired |  |
| `Background` color picker | `excel_bg_color` | ✅ wired |  |
| `BG Opacity` slider (0–100 %) | `excel_bg_opacity` | ✅ wired |  |
| `Text Outline` checkbox | `excel_text_outline` | ✅ wired |  |
| `Outline Color` color picker | `excel_outline_color` | ✅ wired |  |
| `Outline Size` slider (1–10) | `excel_outline_size` | ✅ wired |  |
| `👁️ Live Preview` button | — | ✅ wired | Opens `show_excel_text_preview` ([complete_automation_gui.py:2649](../complete_automation_gui.py#L2649)) — a side-by-side canvas using the exact same settings. |

**What you get on render**
- The **caption** for each video comes from the configured Excel column.
- The **MP3** for each video is auto-picked from the audio folder by ID.
- All font / color / outline / position values from the panel **override** the
  global ones for the Excel-driven row — see lines 7211-7266 of the renderer.

---

## 3. 📝 Blur Text Overlay (Row 0, Col 2)

A standalone caption block that is rendered **on top of the blurred region**.
Distinct from the main quote caption: this text is the *secondary* line that
appears *inside* the blurred area (e.g. "SUBSCRIBE", a tagline, your handle).

| Control | Setting key | Status | Render impact |
|---|---|---|---|
| `✓ Enable Blur_Text` checkbox | `blur_text_enabled` | ✅ wired | Master toggle. If off, the entire `blur_text_*` block is skipped ([youtube_video_automation_enhanced.py:1629-1661](../youtube_video_automation_enhanced.py#L1629)). |
| Multi-line `Text Content` box | `blur_text_content` | ✅ wired | Saved on focus-out. Use `\n` for line breaks. |
| `create_text_controls(blur_text_card, 'blur_text')` — full set | `blur_text_*` (size, color, font, position, bold, italic, outline, etc.) | ✅ wired | The exact same comprehensive set used by the Quote Settings card. |
| `💾 Save Template` / `📂 Load Template` | — | ✅ wired | Templates card on the right of Quick Start saves the `blur_text_*` set. |

**Render impact** — when enabled and there's a region to blur, cv2 writes the
configured text on top of the **first** blurred rectangle, centered. No effect
on captions, color, or audio.

---

## 4. 🎯 Custom Blur Regions (Hide Logos) (Row 1, full width) — "free rectangles" card

The most powerful panel in the tab: define any number of **rectangular
blur boxes at exact X/Y/W/H coordinates** to erase channel logos, TikTok
watermarks, or in-video text. Each region can also drop a **replacement
text box** (or pull the text from a per-row column in a separate Excel
spreadsheet).

> 💡 **This card and 🌫️ Region Blur above do the same physical thing
> (cv2.GaussianBlur on a rectangle) but solve different problems:**
>
> | Use the Region Blur card (above) when you want… | Use this card when you want… |
> |---|---|
> | A single, **preset** shape (top / bottom / left / right / center / top_bottom / left_right) | Multiple rectangles at **arbitrary X/Y/W/H** positions |
> | A quick "bottom black bar" or "lower third" with one switch | To **hide a specific logo / watermark** at a pixel position |
> | A single tint + feather that applies to the band | Per-region text, font, color, intensity, optional Excel text |
>
> Both can be enabled at the same time — they fire in the same render pass
> ([youtube_video_automation_enhanced.py:1376-1490](../youtube_video_automation_enhanced.py#L1376)) and do not conflict.

| Control | Setting key | Status | Render impact |
|---|---|---|---|
| `👁️ Live Preview` | — | ✅ wired | Opens `show_blur_regions_preview` ([complete_automation_gui.py:1866](../complete_automation_gui.py#L1866)) with the current settings drawn on the first frame. |
| `📊 Excel Spreadsheet` (file + browse) | `blur_regions_spreadsheet_file` | ✅ wired | If a region has `use_spreadsheet=True`, the renderer calls `VideoEffects.get_text_from_spreadsheet(video_path, file, column)` once per region, then caches it in `settings[f"_cached_text_{id}"]` so it isn't re-fetched per frame ([youtube_video_automation_enhanced.py:1507-1526](../youtube_video_automation_enhanced.py#L1507)). |
| Per-region block: name, description, `✓ enabled` | each `custom_blur_regions[i]` | ✅ wired | The enable flag is read at [youtube_video_automation_enhanced.py:1383, 1499](../youtube_video_automation_enhanced.py#L1383). |
| `X / Y / W / H` (all in %) | per-region | ✅ wired | The renderer treats 0-100 values as percentages of the frame, anything else as raw pixels ([youtube_video_automation_enhanced.py:1393-1419](../youtube_video_automation_enhanced.py#L1393)). |
| `Blur` intensity (1–100) | per-region | ✅ wired | Per-region kernel size; falls back to the global `blur_intensity` if missing. |
| `📝 Replacement Text` + `📏 Auto-expand` checkbox | per-region | ✅ wired | When `auto_expand=True`, the box shrinks to fit the text exactly ([youtube_video_automation_enhanced.py:1565-1580](../youtube_video_automation_enhanced.py#L1565)). |
| `Font Size / Font Style / Bg color / Text color / Bg opacity` | per-region | ✅ wired | All used by the cv2 text render block ([youtube_video_automation_enhanced.py:1549-1626](../youtube_video_automation_enhanced.py#L1549)). |
| `Use Spreadsheet` + `Column` | per-region | ✅ wired | Tells the renderer to pull the text from the configured Excel sheet instead of the manual `Text` field. |

### How a custom region is rendered

For every frame the renderer ([youtube_video_automation_enhanced.py:1325-1665](../youtube_video_automation_enhanced.py#L1325)) does:

1. Skip the entire path unless `region_blur_enabled=True` **or** at least one
   custom region is enabled.
2. For each enabled custom region, compute the (x1, y1, x2, y2) box in pixels
   from the percentage values.
3. `cv2.GaussianBlur(roi, (kernel, kernel), 0)` on that sub-rectangle and
   write it back.
4. Resolve the text: spreadsheet (cached) → manual `text` field → empty.
5. If text is present and `auto_expand=False`, draw a background rectangle at
   the configured color+opacity, then cv2 the text centered with an auto
   outline (black on light, white on dark).
6. If text is present and `auto_expand=True`, measure the text first, then
   draw the box tight to the text + 20%/30% padding.

**How to add a region** — the panel auto-renders one card per entry in
`settings['custom_blur_regions']`. There is no GUI "+ Add region" button in
this version; regions are seeded by your templates/saved data. If you want
to add a new one, paste a dict into `overlay_settings.json` like:

```json
{
  "name": "TikTok Watermark",
  "description": "Bottom-right corner",
  "enabled": true,
  "x": 80, "y": 85, "width": 18, "height": 12,
  "intensity": 35,
  "auto_expand": false,
  "text": "",
  "text_color": "#FFFFFF",
  "bg_color": "#000000",
  "bg_opacity": 180,
  "use_spreadsheet": false,
  "spreadsheet_column": "B"
}
```

Then restart the GUI — the new card appears under the existing regions.

**Steps to use the existing regions**
1. Set `✓ Region Name` on the card you want active.
2. Adjust X / Y / W / H so the rectangle covers the watermark. (Use
   **👁️ Live Preview** to see it on the first frame.)
3. Raise **Blur** intensity until the logo is unreadable (start ~25, push to
   ~50 for aggressive).
4. Optionally add a **Replacement Text** to drop your own branding in the
   same spot. Tick **📏 Auto-expand** so the box shrinks to fit.
5. Save the card with **💾 Save Template** (top right of the tab) so all
   custom regions persist across restarts.

**Render impact** — every enabled region adds one cv2 GaussianBlur call per
frame plus, if there's text, one cv2.rectangle + one or two cv2.putText
calls. With 2-3 regions you should not see any wall-clock difference; with
8+ regions the per-frame cost climbs but stays well under one MoviePy
encode pass.

---

## Quick "is it working?" checklist

| Test | How | Expected |
|---|---|---|
| Region blur visible? | Enable `Region Blur` + pick `bottom`, Size 30, Intensity 30, run on a single video. | Bottom 30 % of the frame is fuzzy. |
| Tint visible? | Tick `Color Tint`, color `#FF0000`, opacity 60. | The bottom 30 % has a red wash over the blur. |
| Excel text loaded? | Enable Excel, point at an `.xlsx`, run with `--debug` or watch stdout. | `Ready to process N videos from Excel`. |
| Audio file matched? | Put `-111224006530392.mp3` in the audio folder for a video named `-111224006530392.mp4`. | `Found audio file: -111224006530392.mp3`. |
| Custom region hit? | Enable a region, run on one video, open the result. | A blurred rectangle appears at the configured position. |
| Custom region text? | Add `text="HANDLE"` to a region. | The rectangle has a dark background and "HANDLE" centered inside. |
| Blur Text Overlay? | Enable `Blur_Text`, set content to `WATCH TILL END`. | The text is drawn on top of the first blurred region. |
