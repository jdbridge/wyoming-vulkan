"""Pocket TTS on onnxruntime (vendored).

Runtime of https://github.com/thewh1teagle/pocket-tts-onnx at commit 9860fc0fe0ee5ffb13a4bae2fb99e9bd913acbe5
(CC-BY-4.0, see LICENSE; Pocket TTS itself: Kyutai, https://github.com/kyutai-labs/pocket-tts). Unmodified copies of
tts.py, text.py, audio.py and hub.py; this __init__ is ours and leaves out the Hebrew/IPA G2P (g2p.py), which only the
"english-ipa" model needs.
"""

from pocket_tts_onnx.tts import PocketTTS

__all__ = ["PocketTTS"]
