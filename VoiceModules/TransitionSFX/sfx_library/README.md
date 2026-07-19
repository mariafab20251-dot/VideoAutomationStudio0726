# Transition SFX Library

Drop your custom `.wav` files in this folder to override the generated SFX.

## Filename -> Transition mapping

| Filename      | Used for transitions                              |
|---------------|---------------------------------------------------|
| `whoosh.wav`  | zoom, slide                                       |
| `swoosh.wav`  | wipe                                              |
| `boom.wav`    | blur                                              |
| `chime.wav`   | cinematic bars                                    |
| `click.wav`   | (utility)                                         |
| `zap.wav`     | (utility)                                         |
| `sparkle.wav` | lens flare                                        |
| `hiss.wav`    | light leak                                        |
| `rumble.wav`  | film burn                                         |
| `shimmer.wav` | fade                                              |
| `glitch.wav`  | glitch (uses click+zap if missing)                |

## Format

- WAV (PCM 16-bit or 32-bit float)
- Any sample rate (gets resampled to match video)
- Mono or stereo (mono gets duplicated)

## Defaults

If a file is missing, the `transition_sfx.py` generator creates a synthesized version
on the fly. Generated sounds are short, basic, and clean - good for prototyping.
Replace them with your own high-quality recordings for production use.
