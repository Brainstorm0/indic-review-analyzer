"""Text normalisation, script-based language detection and tokenisers for the RNN family."""
import re
import unicodedata
from collections import Counter
from .config import INDICBERT
_RANGES = [("hi", 0x0900, 0x097F), ("bn", 0x0980, 0x09FF), ("pa", 0x0A00, 0x0A7F),
           ("gu", 0x0A80, 0x0AFF), ("or", 0x0B00, 0x0B7F), ("ta", 0x0B80, 0x0BFF),
           ("te", 0x0C00, 0x0C7F), ("kn", 0x0C80, 0x0CFF), ("ml", 0x0D00, 0x0D7F),
           ("ur", 0x0600, 0x06FF)]
# Function words that separate Marathi from Hindi (both use Devanagari).
_MARATHI_WORDS = {"आहे", "आहेत", "नाही", "खूप", "आणि", "मला", "झाला", "झाले", "छान", "होते", "होता",
                  "केले", "पण", "हे", "ही", "तो", "ते", "चांगला", "चांगले", "चांगली", "आम्ही", "माझ्या", "त्याचा",
                  "किंवा", "अतिशय", "फक्त", "हा", "जास्त", "लहान", "करतो", "नाहीत", "वाईट", "तुम्ही",
                  "सर्वांत", "याचा", "देतो", "येतो", "ती", "मी", "होतो", "बराच", "मध्ये", "आम्हाला"}
_MARATHI_SUFFIXES = ("च्या", "ल्या", "ण्या", "साठी", "ण्यासाठी", "ण्याची", "ाचा", "ाची", "ाचे")
_HINDI_WORDS = {"है", "हैं", "नहीं", "और", "बहुत", "मुझे", "था", "थी", "थे", "गया", "किया", "यह", "वह",
                "का", "की", "के", "में", "से", "को", "अच्छा", "अच्छी", "लेकिन", "भी"}
# Roman-script Hindi words. Strong hints alone mark text as Hinglish; weak ones (also common
# typos or English fragments) need a second hint.
_HINGLISH_STRONG = {"hai", "nahi", "nahin", "bahut", "bohot", "accha", "acha", "achha", "bekar", "bakwas",
                    "kya", "paisa", "vasool", "yaar", "ekdum", "bilkul", "kharab", "hain", "nhi", "zyada",
                    "jyada", "mujhe", "lekin", "tha", "thi", "kaafi", "bekaar", "achi", "acchi", "bura"}
_HINGLISH_WEAK = {"bhi", "mein", "ko", "ki", "ke", "sahi", "mast", "se", "ka", "ye", "wo"}
def normalize_text(text) -> str:
    t = unicodedata.normalize("NFC", str(text))
    t = re.sub(r"https?://\S+", " ", t)
    return re.sub(r"\s+", " ", t).strip()
def detect_lang(text: str) -> str:
    counts, latin = {}, 0
    for ch in text:

        if "a" <= ch.lower() <= "z":
            latin += 1
            continue
        cp = ord(ch)
        for code, lo, hi in _RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if counts and max(counts.values()) >= latin:
        code = max(counts, key=counts.get)
        if code == "hi":
            toks = word_tokenize(text)
            mr = (sum(t in _MARATHI_WORDS or t.endswith(_MARATHI_SUFFIXES) for t in toks)
                  + 2 * ("ळ" in text))
            hi = sum(t in _HINDI_WORDS for t in toks)
            if mr > hi:
                return "mr"
        return code
    words = re.findall(r"[a-z]+", text.lower())
    strong = sum(w in _HINGLISH_STRONG for w in words)
    weak = sum(w in _HINGLISH_WEAK for w in words)
    return "hinglish" if strong >= 1 or weak >= 2 else "en"
# NOTE: Python's \w does not match Indic vowel signs (matras), so we split on whitespace/punctuation instead.
_SPLIT_RE = re.compile(r"[^\s\.,!?;:\"'()\[\]{}।॥|/\\\-]+|[!?]")
def word_tokenize(text: str):
    return _SPLIT_RE.findall(text.lower())
def char_tokenize(text: str):
    return list(text.lower().replace(" ", "▁"))
_SP = None
def subword_tokenize(text: str):
    """IndicBERT SentencePiece pieces (keep_accents=True is essential, otherwise matras are stripped)."""
    global _SP
    if _SP is None:
        from transformers import AutoTokenizer
        _SP = AutoTokenizer.from_pretrained(INDICBERT, keep_accents=True, use_fast=False)
    return _SP.tokenize(text)
TOKENIZERS = {"word": word_tokenize, "char": char_tokenize, "subword": subword_tokenize}
class Vocab:
    PAD, UNK = 0, 1
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
    @classmethod
    def build(cls, token_lists, min_freq: int = 2, max_size: int = 40000):
        c = Counter(t for toks in token_lists for t in toks)
        items = [t for t, f in c.most_common(max_size) if f >= min_freq]
        return cls(["<pad>", "<unk>"] + items)
    def encode(self, toks, max_len: int):

        ids = [self.stoi.get(t, self.UNK) for t in toks][:max_len]
        return ids or [self.UNK]
    def __len__(self):
        return len(self.itos)
