"""How each source ``cleaning_status`` may be used downstream.

The source pipeline compared the phonemes of the original line with the
phonemes of Whisper's (normalised) transcript. Both sides are presumably
produced by G2P from *text*, so a match confirms that the words are right but
not how kanji were read: a homograph misread by G2P is misread identically on
both sides. Readings therefore always come from LilyTTS's own audio-based
labeler, and the fields below only describe how far the *text* can be trusted.

Fields written to the manifest:

* ``text_check``: who confirmed the text (``human`` / ``asr_phone`` / ``ctc``).
* ``phrase_end_mismatch``: the audio may lengthen or cut the last mora
  (``ですかー`` / ``ですかっ``) compared with the text. Such lines are fine for
  TTS training but are kept out of recogniser teachers and accent gold labels.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StatusPolicy:
    text_check: str
    phrase_end_mismatch: bool = False


POLICIES: dict[str, StatusPolicy] = {
    "acoustic_and_asr_phone_match": StatusPolicy("asr_phone"),
    "phrase_end_timing": StatusPolicy("asr_phone", phrase_end_mismatch=True),
    "ctc_confirms_original": StatusPolicy("ctc"),
    "human_original": StatusPolicy("human"),
    "human_corrected": StatusPolicy("human"),
}

UNKNOWN = StatusPolicy("unknown")


def policy_for(status: str | None) -> StatusPolicy | None:
    """Return the policy for a status, or None if the status is not known."""
    if status is None:
        return None
    return POLICIES.get(status)
