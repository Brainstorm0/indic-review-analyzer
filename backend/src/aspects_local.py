"""Offline aspect-based sentiment: split a review into clauses, find aspect words from a multilingual
word list, and score each clause's sentiment with the project's own trained model.

No internet or API key needed. Aspects not in the word list are not found (the LLM mode covers those)."""
import re

# Canonical aspect -> words that signal it, in English/Hinglish (Roman) and the five Indic scripts.
# Indic entries are word stems matched anywhere in a clause, so inflected forms (कैमरा/कैमरे) also match.
ASPECTS = {
    "price": ["price", "cost", "costly", "expensive", "worth", "value for money", "paisa", "paise", "daam", "keemat", "kimat", "mehenga", "mehnga", "sasta", "rate",
              "कीमत", "क़ीमत", "दाम", "पैसे", "पैसा", "महंगा", "महँगा", "सस्ता", "मूल्य", "किंमत", "किम्मत", "पैसे वसूल",
              "விலை", "காசு", "ధర", "డబ్బు", "দাম", "টাকা", "মূল্য"],
    "quality": ["quality", "kwality", "build", "गुणवत्ता", "क्वालिटी", "दर्जा", "தரம்", "నాణ్యత", "క్వాలిటీ", "মান", "কোয়ালিটি", "গুণমান"],
    "delivery": ["delivery", "deliver", "shipping", "courier", "arrived", "late", "डिलीवरी", "डिलिव्हरी", "पहुंचा", "पहुँचा",
                 "டெலிவரி", "డెలివరీ", "ডেলিভারি"],
    "packaging": ["packaging", "packing", "package", "box", "पैकिंग", "पैकेजिंग", "डिब्बा", "पॅकिंग", "பேக்கிங்", "ప్యాకింగ్", "প্যাকেজিং", "প্যাকিং"],
    "battery": ["battery", "charge", "charging", "बैटरी", "बॅटरी", "चार्ज", "பேட்டரி", "சார்ஜ்", "బ్యాటరీ", "ఛార్జ్", "ব্যাটারি", "চার্জ"],
    "camera": ["camera", "photo", "picture", "कैमरा", "कैमरे", "कॅमेरा", "फोटो", "கேமரா", "புகைப்பட", "కెమెరా", "ఫోటో", "ক্যামেরা", "ছবি"],
    "display": ["display", "screen", "डिस्प्ले", "स्क्रीन", "டிஸ்ப்ளே", "திரை", "స్క్రీన్", "డిస్ప్లే", "স্ক্রিন", "ডিসপ্লে"],
    "sound": ["sound", "audio", "speaker", "bass", "volume", "noise", "आवाज़", "आवाज", "साउंड", "ध्वनी", "ஒலி", "சத்தம்", "సౌండ్", "శబ్దం", "ধ্বনি", "শব্দ", "সাউন্ড"],
    "performance": ["performance", "speed", "slow", "lag", "hang", "smooth", "chalta", "chalti",
                   "परफॉर्मेंस", "धीमा", "तेज़", "हैंग", "स्पीड", "वेग", "செயல்திறன்", "வேகம்", "పనితీరు", "వేగం", "পারফরম্যান্স", "গতি"],
    "design": ["design", "look", "looks", "style", "color", "colour", "beautiful", "डिज़ाइन", "डिजाइन", "दिखने", "रंग", "सुंदर", "रूप", "डिझाइन",
               "வடிவமைப்பு", "நிறம்", "డిజైన్", "రంగు", "ডিজাইন", "রং", "দেখতে"],
    "size": ["size", "fit", "fitting", "small", "big", "साइज़", "साइज", "आकार", "माप", "அளவு", "సైజ్", "పరిమాణం", "সাইজ", "মাপ", "আকার"],
    "durability": ["durable", "durability", "broke", "broken", "lasting", "toot", "tut", "kharab ho", "टिकाऊ", "टूट", "खराब हो गया", "बंद पड",
                   "உடைந்து", "நீடித்த", "పగిలి", "మన్నిక", "ভেঙে", "নষ্ট", "টেকসই"],
    "material": ["material", "fabric", "cloth", "plastic", "मटेरियल", "कपड़ा", "कपडा", "सामग्री", "துணி", "பொருள்", "మెటీరియల్", "వస్త్రం", "কাপড়", "উপাদান"],
    "service": ["service", "customer care", "support", "staff", "seller", "installation", "refund", "return", "सर्विस", "सेवा", "कस्टमर केयर", "इंस्टॉलेशन", "विक्रेता",
                "சேவை", "సేవ", "సర్వీస్", "পরিষেবা", "সার্ভিস"],
    "taste": ["taste", "tasty", "flavour", "flavor", "swad", "स्वाद", "चव", "சுவை", "రుచి", "স্বাদ"],
    "comfort": ["comfort", "comfortable", "soft", "आराम", "आरामदायक", "मुलायम", "आरामदायी", "வசதி", "மென்மை", "సౌకర్య", "মোলায়েম", "আরাম", "নরম"],
    "ease of use": ["easy to use", "easy", "difficult", "user friendly", "आसान", "सोपे", "सोपा", "எளிது", "எளிதாக", "సులభం", "সহজ"],
    "content": ["story", "content", "book", "movie", "film", "acting", "कहानी", "किताब", "फिल्म", "पुस्तक", "कथा", "கதை", "புத்தக", "படம்",
                "కథ", "పుస్తకం", "సినిమా", "গল্প", "বই", "সিনেমা"],
}

# Words that join two separate opinions ("good camera BUT poor battery"). Splitting there gives one
# clause per opinion. Roman words use word boundaries; Indic ones are surrounded by spaces.
_SPLIT_ROMAN = ["but", "however", "although", "though", "and", "while", "lekin", "par", "magar", "aur", "or"]
_SPLIT_INDIC = ["लेकिन", "परंतु", "किंतु", "मगर", "पर", "और", "तथा", "एवं", "पण", "आणि", "मात्र",
                "ஆனால்", "மற்றும்", "కానీ", "మరియు", "కాని", "কিন্তু", "এবং", "আর"]
_PUNCT = re.compile(r"[,.;!?।॥|]+")


def split_clauses(text: str):
    parts = [p.strip() for p in _PUNCT.split(text) if p.strip()]
    out = []
    for part in parts:
        p = " " + part + " "
        for w in _SPLIT_INDIC:
            p = p.replace(f" {w} ", " \x00 ")
        p = re.sub(r"\b(" + "|".join(_SPLIT_ROMAN) + r")\b", "\x00", p, flags=re.IGNORECASE)
        out += [c.strip() for c in p.split("\x00") if len(c.strip()) >= 2]
    return out


# Sentiment words for Roman-script (English/Hinglish) clauses. The trained models are reliable on
# Indian scripts but often misjudge very short Roman-script phrases, so these words decide there.
_POS_ROMAN = {"good", "great", "nice", "best", "excellent", "amazing", "awesome", "perfect", "love", "loved", "fast", "quick",
              "worth", "reasonable", "affordable", "soft", "comfortable", "smooth", "beautiful", "superb", "fantastic",
              "bright", "clear", "sharp", "crisp", "loud", "sturdy", "strong", "fresh", "tasty", "helpful",
              "achha", "acha", "accha", "achhi", "acchi", "achi", "badiya", "badhiya", "mast", "zabardast", "shandar", "sahi", "badhia"}
_NEG_ROMAN = {"bad", "poor", "worst", "terrible", "horrible", "awful", "waste", "useless", "damaged", "broken", "broke", "slow",
              "late", "delayed", "costly", "overpriced", "disappointed", "disappointing", "faulty", "defective", "cheap",
              "bekar", "bekaar", "bakwas", "kharab", "ganda", "ghatiya", "barbaad", "bura", "buri"}
_NEGATION = {"not", "no", "never", "nahi", "nahin", "nhi", "mat"}


def _roman_polarity(clause: str):
    words = re.findall(r"[a-z]+", clause.lower())
    pos = sum(w in _POS_ROMAN for w in words)
    neg = sum(w in _NEG_ROMAN for w in words)
    if any(w in _NEGATION for w in words):
        pos, neg = neg, pos + (1 if pos == 0 and neg == 0 else 0)
    if pos > neg:
        return 0.9
    if neg > pos:
        return 0.1
    return None


def _find_aspects(clause: str):
    low = clause.lower()
    found = []
    for aspect, words in ASPECTS.items():
        for w in words:
            if w.isascii():
                if re.search(r"\b" + re.escape(w) + r"\b", low):
                    found.append(aspect)
                    break
            elif w in clause:
                found.append(aspect)
                break
    # "sound quality", "camera quality" ... describe the sound/camera, not a separate "quality" aspect.
    if "quality" in found and any(a in found for a in ("sound", "camera", "display")):
        found.remove("quality")
    return found


def extract_aspects_local(text: str, muril_proba, lr_proba=None, neutral_band: float = 0.2) -> dict:
    """muril_proba / lr_proba: functions mapping a list of texts to rows of [P(negative), P(positive)]."""
    clauses = split_clauses(text)
    tagged = [(c, _find_aspects(c)) for c in clauses if _find_aspects(c)]
    texts = [text] + [c for c, _ in tagged]
    pm = muril_proba(texts)[:, 1]
    pl = lr_proba(texts)[:, 1] if lr_proba else pm

    def clause_p(i, clause):
        if not re.search(r"[a-zA-Z]", clause) or len(re.findall(r"[a-zA-Z]", clause)) < len(clause) / 2:
            return float(pm[i])                      # Indian script: the transformer is reliable
        lex = _roman_polarity(clause)
        return lex if lex is not None else float((pm[i] + pl[i]) / 2)

    def label(p):
        if abs(p - 0.5) < neutral_band:
            return "neutral"
        return "positive" if p > 0.5 else "negative"

    aspects, seen = [], set()
    for i, (clause, names) in enumerate(tagged, start=1):
        p = clause_p(i, clause)
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            aspects.append({"aspect": name, "sentiment": label(p), "evidence": clause, "evidence_en": "",
                            "confidence": round(max(p, 1 - p), 3)})
    return {"overall": label(clause_p(0, text)), "translation_en": "", "aspects": aspects, "method": "local"}
