"""Fixed audio format and concealment parameters.

The input format is part of the API contract: 16 kHz, mono, signed 16-bit
little-endian PCM, 20 ms per numbered frame (320 samples / 640 bytes).
"""

SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320
FRAME_BYTES = FRAME_SAMPLES * 2  # 640

HISTORY_MS = 40
HISTORY_SAMPLES = SAMPLE_RATE * HISTORY_MS // 1000  # 640

# Pitch search range: 40..200 samples == 80..400 Hz at 16 kHz.
MIN_PERIOD_SAMPLES = 40
MAX_PERIOD_SAMPLES = 200
MIN_PEAK_CORRELATION = 0.6

# The recovered frame's first 5 ms fade from concealment into real audio.
FADE_MS = 5
FADE_SAMPLES = SAMPLE_RATE * FADE_MS // 1000  # 80

# Concealment repeats the last period at half amplitude.
CONCEAL_GAIN = 0.5

# Reordering buffer bound.  A permanently missing sequence would otherwise
# let every later packet accumulate for ever; at the cap new *future* frames
# are refused (429) instead of evicting already-accepted data.
MAX_PENDING_FRAMES = 5_000  # 100 s of audio
