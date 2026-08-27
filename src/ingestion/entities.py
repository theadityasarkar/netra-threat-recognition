"""
Layer 1 (unstructured) — NLP entity extraction from free-text intelligence.

PS 26189 requires the system to *"extract important entities such as people,
locations, vehicles, phone numbers, and organizations"* from the unstructured
sources named in its Background section: FIR narratives, surveillance reports,
social-media intelligence and intelligence-agency reports.

Design decision — pattern + gazetteer NER, not a transformer
------------------------------------------------------------
A spaCy/BERT pipeline was deliberately rejected:

* **Deployability.** ``en_core_web_trf`` is ~430 MB and needs >1 GB RSS to run.
  Streamlit Community Cloud gives ~1 GB total, so a transformer NER would make
  the free deployment target in the brief impossible.
* **Determinism.** A seeded, rule-based extractor returns byte-identical output
  every run, which matters when the output is quoted in a case dossier.
* **Accuracy where it counts.** Generic English NER does *not* know Indian
  vehicle-registration format, IFSC codes, Indian mobile numbering, or Hindi
  honorifics — precisely the entities an NCRB analyst cares about. Purpose-built
  patterns beat a general model on this domain.

The honest cost: recall on *uncued* person names is lower than a transformer's.
A name that appears with no honorific, no role cue and no prior mention scores at
the bare-capitalisation floor and may fall below the confidence threshold. The
public signatures here are model-agnostic, so swapping in a fine-tuned NER later
is a drop-in change to :func:`extract_entities` alone.

Every mention carries a confidence score and the surrounding context so an
investigator can verify it against the source document — extraction is a lead
generator, never an assertion of fact.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from itertools import combinations

import pandas as pd

from src.config import CRIME_TYPES, ENTITY_LABELS, NLP_PARAMS, RELATIONSHIP_COLUMNS

# ──────────────────────────────────────────────────────────────────────────────
# Gazetteers
#
# Kept compact and India-specific. These are *negative* filters as much as
# positive ones: a token that is a known city must not also be scored as a
# person name.
# ──────────────────────────────────────────────────────────────────────────────

INDIAN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa",
    "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala",
    "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
    "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
    "uttar pradesh", "uttarakhand", "west bengal", "delhi", "jammu and kashmir",
    "ladakh", "puducherry", "chandigarh", "andaman and nicobar islands", "lakshadweep",
    "dadra and nagar haveli", "daman and diu",
}

INDIAN_CITIES = {
    "mumbai", "delhi", "new delhi", "bengaluru", "bangalore", "hyderabad", "chennai",
    "kolkata", "pune", "ahmedabad", "surat", "jaipur", "lucknow", "kanpur", "nagpur",
    "indore", "thane", "bhopal", "visakhapatnam", "patna", "vadodara", "ghaziabad",
    "ludhiana", "agra", "nashik", "faridabad", "meerut", "rajkot", "varanasi",
    "srinagar", "aurangabad", "dhanbad", "amritsar", "allahabad", "prayagraj",
    "ranchi", "howrah", "coimbatore", "jabalpur", "gwalior", "vijayawada", "jodhpur",
    "madurai", "raipur", "kota", "guwahati", "solapur", "hubli", "mysuru", "mysore",
    "tiruchirappalli", "bareilly", "aligarh", "moradabad", "gurugram", "gurgaon",
    "noida", "jalandhar", "bhubaneswar", "salem", "warangal", "guntur", "bhiwandi",
    "saharanpur", "gorakhpur", "bikaner", "amravati", "cuttack", "firozabad",
    "kochi", "cochin", "nellore", "bhavnagar", "dehradun", "durgapur", "asansol",
    "siliguri", "darbhanga", "muzaffarpur", "kurnool", "rourkela", "ajmer", "goa",
    "panaji", "shimla", "imphal", "shillong", "aizawl", "kohima", "gangtok",
    "agartala", "itanagar", "dispur", "puducherry", "port blair", "kakinada",
    "sangli", "belgaum", "jamnagar", "ujjain", "loni", "jhansi", "nanded",
    "kolhapur", "sonipat", "malegaon", "gaya", "jalgaon", "udaipur", "maheshtala",
    # Border / transit points that recur in trafficking-corridor casework.
    "raxaul", "siliguri corridor", "petrapole", "jogbani", "sunauli", "banbasa",
}

LOCATION_GAZETTEER = INDIAN_STATES | INDIAN_CITIES

# Organisation detection is suffix-driven: an entity is an org because of what it
# ends with, which generalises far better than any list of company names.
ORG_SUFFIXES = {
    "traders", "trading", "enterprises", "enterprise", "industries", "exports",
    "imports", "corporation", "corp", "company", "co", "pvt", "private", "limited",
    "ltd", "llp", "agency", "agencies", "associates", "brothers", "sons", "group",
    "holdings", "ventures", "solutions", "services", "systems", "technologies",
    "logistics", "transport", "transports", "travels", "tours", "hotel", "hotels",
    "lodge", "guesthouse", "restaurant", "bar", "cafe", "dhaba", "spa", "salon",
    "placement", "consultancy", "consultants", "recruiters", "manpower", "overseas",
    "exchange", "forex", "finance", "financials", "capital", "investments",
    "builders", "developers", "constructions", "motors", "automobiles", "garage",
    "pharma", "pharmaceuticals", "chemicals", "textiles", "mills", "studio",
    "productions", "entertainment", "network", "communications", "telecom",
}

# Placement agencies / spas / bars are the front-business archetypes in NCRB
# Women Safety casework — worth flagging as a distinct sub-signal.
FRONT_BUSINESS_SUFFIXES = {
    "placement", "consultancy", "manpower", "recruiters", "overseas", "travels",
    "tours", "spa", "salon", "bar", "lodge", "guesthouse", "hotel", "studio",
    "entertainment", "productions",
}

WEAPON_TERMS = {
    "pistol", "revolver", "rifle", "shotgun", "carbine", "ak-47", "ak47",
    "country-made pistol", "katta", "firearm", "cartridge", "cartridges",
    "live rounds", "ammunition", "magazine", "knife", "dagger", "chopper",
    "sword", "machete", "grenade", "explosive", "explosives", "detonator",
    "gelatin sticks", "ied", "acid", "acid bottle",
    # Hindi & regional terminology
    "तमंचा", "कट्टा", "पिस्तौल", "कारतूस", "चाकू", "हथियार", "छुरा", "तलवार", "बम",
}

CONTRABAND_TERMS = {
    "heroin", "brown sugar", "smack", "charas", "hashish", "ganja", "cannabis",
    "marijuana", "opium", "poppy husk", "cocaine", "mdma", "ecstasy", "lsd",
    "methamphetamine", "meth", "mephedrone", "md", "ketamine", "tramadol",
    "alprazolam", "codeine", "contraband", "narcotics", "psychotropic",
    "red sanders", "counterfeit currency", "fake currency", "fico", "fakes",
    # Hindi & regional terminology
    "चरस", "गांजा", "अफीम", "हेरोइन", "चिट्टा", "नकली नोट", "शराब", "नशीला पदार्थ",
}

# Words that are capitalised in running prose but are never person names. Used to
# suppress the commonest false-positive class in the PERSON extractor.
PERSON_STOPWORDS = {
    "the", "this", "that", "these", "those", "there", "then", "thereafter",
    "accused", "victim", "complainant", "witness", "suspect", "informant",
    "police", "station", "inspector", "constable", "sub", "officer", "court",
    "magistrate", "judge", "section", "act", "ipc", "crpc", "pocso", "itact",
    "fir", "case", "crime", "report", "statement", "investigation", "arrest",
    "arrested", "custody", "bail", "remand", "chargesheet", "charge", "sheet",
    "during", "after", "before", "further", "however", "subsequently", "meanwhile",
    "on", "in", "at", "near", "from", "to", "by", "with", "and", "or", "but",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "north", "south", "east", "west", "central", "district", "state", "city",
    "road", "street", "nagar", "colony", "market", "railway", "airport", "bus",
    "stand", "hospital", "school", "college", "university", "temple", "mosque",
    "church", "hotel", "lodge", "bar", "shop", "godown", "warehouse", "flat",
    "house", "building", "area", "sector", "block", "phase", "gate", "chowk",
    "mobile", "phone", "number", "vehicle", "car", "bike", "truck", "van",
    "cash", "amount", "rupees", "lakh", "crore", "bank", "account", "atm",
    "whatsapp", "telegram", "facebook", "instagram", "signal", "sim", "imei",
    "unknown", "unidentified", "one", "two", "three", "several", "many", "some",
    # Hindi grammatical particles & narrative stopwords
    "और", "तथा", "एवं", "को", "ने", "से", "का", "के", "की", "में", "पर", "द्वारा",
    "हेतु", "था", "थी", "थे", "है", "हैं", "गया", "गई", "पास", "नकद", "बरामद",
    "उर्फ", "पुत्र", "पुत्री", "पत्नी", "पिता",
}

# Offence vocabulary, derived from the canonical crime list rather than hand-typed.
# "Offence registered as Human Trafficking" would otherwise be read as a person
# name: it is two capitalised words following a role cue, which is exactly the
# PERSON signature. Deriving the blacklist from CRIME_TYPES keeps it in step with
# the vocabulary automatically as crime types are added.
_CRIME_TOKENS = {tok.lower()
                 for ct in CRIME_TYPES
                 for tok in re.findall(r"[A-Za-z]+", ct)}

# Report prose that is routinely capitalised at the start of a sentence and would
# otherwise be scored as a name.
_REPORT_NOUNS = {
    "payments", "payment", "offence", "offences", "consignment", "communications",
    "assessment", "movement", "preservation", "surveillance", "recruitment",
    "handset", "wallet", "profile", "advance", "employment", "placement",
    "contact", "content", "direct", "open", "source", "monitoring", "intelligence",
    "financial", "flows", "associates", "recommend", "cross", "reference",
    "subject", "vehicle", "number", "handle", "note", "input", "activity",
    "principal", "node", "coordinating", "confidence", "moderate", "continued",
    "watch", "tower", "dump", "packet", "premises", "advertisements", "messages",
    "request", "platform", "attributed", "interacts", "frequently", "listed",
    "solicited", "totalling", "traced", "maintained", "registered", "graded",
}

PERSON_STOPWORDS |= _CRIME_TOKENS | _REPORT_NOUNS

# Honorifics and cue words that make a following capitalised token a near-certain
# person name. This is what lifts precision without a statistical model.
#
# The two families below are matched differently, and the distinction matters. An
# *abbreviation* cue ("Smt.", "s/o", "a/k/a") legitimately carries a trailing full
# stop. A *role* cue ("accused", "subject") does not — so a period after it is a
# sentence boundary, and letting the pattern step over it turns
# "…recovered from the accused. Payments totalling…" into a person named
# "Payments". Role cues therefore require whitespace, never a period.
_ROLE_CUES = (
    r"accused|victim|complainant|witness|suspect|informant|deceased|absconder|"
    r"co-?accused|handler|kingpin|pandit|thakur|seth|bhai|one|named|namely|alias|"
    r"aaropi|aropi|peedita|pidita|gawah|shakki|sargana|dalal|thekedar|kotwali|thana|"
    # Surveillance-report register. The PS names surveillance and intelligence
    # reports as sources, and those write "Subject <Name>" / "Target <Name>" where
    # an FIR writes "Accused <Name>". Without these cues the capitalised-run tier
    # swallows the cue word, so "Subject Ramesh Kumar" becomes an entity distinct
    # from "Ramesh Kumar" and one person is split across two nodes.
    r"subject|target|detainee|arrestee|petitioner|respondent|proprietor|"
    r"recruiter|middleman|tout"
)
_ABBREV_CUES = (
    r"shri|smt|sri|mr|mrs|ms|dr|md|haji|maulana|kum|kumari|"
    r"aka|a/?k/?a|s/?o|d/?o|w/?o"
)

# Single public alternation, for callers and tests that reason about cue coverage.
PERSON_CUES = f"{_ROLE_CUES}|{_ABBREV_CUES}"

# Honorifics that may sit *between* a role cue and the actual name
# ("complainant Smt. Kavita Sharma"). Without this hop the cue consumes the
# honorific as its candidate, the candidate is rejected as too short, and the
# real name falls through to the low-confidence bare-capitalisation tier.
_HONORIFIC_HOP = r"(?:(?i:shri|smt|sri|mr|mrs|ms|dr|md|haji|kum|kumari)\.?\s+)?"

# ──────────────────────────────────────────────────────────────────────────────
# Patterns
# ──────────────────────────────────────────────────────────────────────────────

# Indian vehicle registration: two-letter state, RTO code, 1–3 letter series,
# 4 digits — with optional separators. Also the newer BH (Bharat) series.
_VEHICLE_RE = re.compile(
    r"\b(?:[A-Z]{2}[\s\-]?\d{1,2}[A-Z]?[\s\-]?[A-Z]{1,3}[\s\-]?\d{4}"
    r"|\d{2}[\s\-]?BH[\s\-]?\d{4}[\s\-]?[A-Z]{1,2})\b",
    re.IGNORECASE)

# 15-digit IMEI (optionally hyphen-grouped). Checked before generic digit runs.
_IMEI_RE = re.compile(r"\b(?:IMEI[\s:#]*)?(\d{15})\b", re.IGNORECASE)

# Indian mobile: optional +91/0091/0 prefix, then 6-9 followed by 9 digits. One
# internal separator is allowed so the "+91 98765 43210" grouping still matches.
_PHONE_RE = re.compile(
    r"(?<![\d])(?:\+?91[\s\-]?|0(?![\d]{10,}))?([6-9]\d{4}[\s\-]?\d{5})(?![\d])")

_IFSC_RE = re.compile(r"\b([A-Z]{4}0[A-Z0-9]{6})\b")
# Bank account: 9–18 digits, and only when a banking cue word is nearby — a bare
# long digit run is far more often a case number or a docket reference.
_ACCOUNT_RE = re.compile(
    r"\b(?:a/?c|acc(?:oun)?t|account\s*(?:no|number)|savings|current)\b[\s.:#no]*(\d{9,18})\b",
    re.IGNORECASE)

# Crypto wallets — the PS theme is Blockchain & Cybersecurity, and laundering
# through wallets is now routine in sextortion and trafficking-proceeds cases.
_CRYPTO_RE = re.compile(
    r"\b(?:0x[a-fA-F0-9]{40}"                      # Ethereum / EVM
    r"|bc1[a-z0-9]{25,62}"                          # Bech32 Bitcoin
    r"|[13][a-km-zA-HJ-NP-Z1-9]{25,34}"             # Legacy Bitcoin (base58)
    r"|T[a-zA-Z0-9]{33})\b")                        # Tron

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")
_HANDLE_RE = re.compile(r"(?<![\w.])@([A-Za-z][A-Za-z0-9_.]{3,29})\b")

# ₹ / Rs / INR / रु amounts, including the lakh/crore multipliers used in Indian FIRs.
_CURRENCY_RE = re.compile(
    r"(?:₹|\bRs\.?|\bINR\b|रु(?:पये|\.)?)\s*([\d,]+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|k|thousand|लाख|करोड़|हज़ार|हजार)?",
    re.IGNORECASE)

_CAPWORD = r"[A-Z][a-z]{1,}"

_PERSON_CUED_RE = re.compile(
    rf"\b(?:(?i:{_ABBREV_CUES})\b\.?|(?i:{_ROLE_CUES})\b)\s+"
    rf"{_HONORIFIC_HOP}((?:{_CAPWORD}\s+){{0,2}}{_CAPWORD})")
_PERSON_BARE_RE = re.compile(rf"\b({_CAPWORD}(?:\s+{_CAPWORD}){{1,2}})\b")

_LOCATION_CUED_RE = re.compile(
    r"\b(?i:at|in|near|from|to|towards|via|resident of|r/?o|posted at|"
    r"recovered (?:from|at)|arrested (?:at|in|from))\s+"
    rf"((?:{_CAPWORD}\s+){{0,3}}{_CAPWORD})")

# Devanagari / Hindi extraction patterns
_HINDI_PERSON_RE = re.compile(
    r"(?:आरोपी|अभियुक्त|पीड़िता|पीड़िता|गवाह|संदिग्ध|सरगना|दलाल|ठेकेदार|श्री|श्रीमती|सुश्री|उर्फ|पुत्र|पुत्री|पत्नी)\s+"
    r"([\u0900-\u097F]{2,20}(?:\s+[\u0900-\u097F]{2,20}){0,2})",
    re.UNICODE)

_HINDI_LOCATION_RE = re.compile(
    r"(?:थाना|कोतवाली|चौकी|गाँव|गांव|मोहल्ला|नगर|जिला|गली|तहसील)\s+"
    r"([\u0900-\u097F]{2,20})",
    re.UNICODE)

# Only the suffix is case-insensitive; the organisation's head must be capitalised.
_ORG_RE = re.compile(
    rf"\b((?:{_CAPWORD}\s+){{0,3}}{_CAPWORD})\s+"
    rf"(?i:({'|'.join(sorted(ORG_SUFFIXES, key=len, reverse=True))}))\b")

_WS = re.compile(r"\s+")

# Entity types that denote a *party* to an interaction — these are the only types
# that get co-occurrence edges. A weapon, a seized consignment, a cash amount or a
# place is context that describes an event, not a participant in it; linking
# "pistol" to "Rs 4,50,000" because they appear in one FIR would manufacture edges
# that carry no relational meaning and would swamp the real network.
LINKABLE_ENTITY_TYPES = frozenset({
    "PERSON", "PHONE", "ORGANISATION", "VEHICLE",
    "BANK_ACCOUNT", "CRYPTO_WALLET", "IMEI", "EMAIL",
})

# Context types — aggregated per document and shown as case attributes instead.
CONTEXT_ENTITY_TYPES = frozenset({"LOCATION", "WEAPON", "CONTRABAND", "CURRENCY"})


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Mention:
    """One occurrence of one entity inside one document."""

    doc_id: str
    entity_id: str
    entity_type: str
    raw: str            # exactly as it appeared in the source
    value: str          # normalised, used for deduplication
    start: int
    end: int
    confidence: float
    context: str        # ±60 chars, for analyst verification


@dataclass
class ExtractionResult:
    """Everything the entity page and the graph integration need."""

    entities: pd.DataFrame
    mentions: pd.DataFrame
    links: pd.DataFrame
    documents: pd.DataFrame
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_entities(self) -> int:
        return len(self.entities)

    @property
    def n_mentions(self) -> int:
        return len(self.mentions)

    def of_type(self, entity_type: str) -> pd.DataFrame:
        if not len(self.entities):
            return self.entities
        return self.entities[self.entities["entity_type"] == entity_type]

    def to_relationship_frame(self) -> pd.DataFrame:
        """Emit co-occurrence links in the canonical relationship schema.

        This is the bridge that lets unstructured intelligence flow into the same
        graph → centrality → GNN → anomaly pipeline as structured CSV evidence.
        """
        if not len(self.links):
            return pd.DataFrame(columns=RELATIONSHIP_COLUMNS)
        out = pd.DataFrame({
            "source_id": self.links["source_id"],
            "target_id": self.links["target_id"],
            "relation_type": "Communication",
            "strength": self.links["strength"],
            "interaction_count": self.links["n_documents"],
            "first_seen": pd.NaT,
            "last_seen": pd.NaT,
            "evidence_source": "Text Analytics",
            "location": "Unknown",
        })
        return out[RELATIONSHIP_COLUMNS]


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────────────────────

def _norm_generic(text: str) -> str:
    return _WS.sub(" ", str(text)).strip()


def normalise_value(entity_type: str, raw: str) -> str:
    """Canonical form used to merge mentions of the same real-world entity."""
    s = _norm_generic(raw)
    if entity_type == "PHONE":
        digits = re.sub(r"\D", "", s)
        return digits[-10:] if len(digits) >= 10 else digits
    if entity_type == "CRYPTO_WALLET":
        # Never upper-case a wallet address: EIP-55 encodes its checksum in the
        # letter casing, and Bitcoin base58 is case-sensitive too.
        return re.sub(r"[\s\-]", "", s)
    if entity_type in {"VEHICLE", "IMEI", "BANK_ACCOUNT"}:
        return re.sub(r"[\s\-]", "", s).upper()
    if entity_type == "EMAIL":
        return s.lower()
    if entity_type in {"PERSON", "LOCATION", "ORGANISATION"}:
        return s.title()
    if entity_type in {"WEAPON", "CONTRABAND"}:
        return s.lower()
    return s


def entity_key(entity_type: str, value: str) -> str:
    """Stable, content-derived entity id — deterministic across runs and machines."""
    digest = hashlib.sha1(f"{entity_type}|{value}".encode()).hexdigest()[:8]
    return f"E-{entity_type[:3]}-{digest}".upper()


def mask_value(entity_type: str, value: str) -> str:
    """Partially redact identifiers for on-screen display.

    Extraction output is shown in a UI that may be screen-shared during a demo or
    a briefing, so raw phone numbers and account numbers are masked by default.
    The unmasked value is retained in the frame for export under authorisation.
    """
    if entity_type == "PHONE" and len(value) >= 10:
        return f"{value[:3]}XXXX{value[-3:]}"
    if entity_type in {"BANK_ACCOUNT", "IMEI"} and len(value) > 6:
        return f"{value[:4]}{'X' * (len(value) - 8)}{value[-4:]}"
    if entity_type == "CRYPTO_WALLET" and len(value) > 14:
        return f"{value[:8]}…{value[-6:]}"
    if entity_type == "EMAIL":
        # A social handle ("@rampal_delhi") shares this type but has no local
        # part, so the e-mail branch would mask the wrong side of the "@".
        if value.startswith("@"):
            handle = value[1:]
            if len(handle) > 5:
                return f"@{handle[:3]}{'X' * (len(handle) - 5)}{handle[-2:]}"
            return value
        if "@" in value:
            local, _, domain = value.partition("@")
            head = local[:2] if len(local) > 2 else local
            return f"{head}{'X' * max(1, len(local) - 2)}@{domain}"
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Extraction
# ──────────────────────────────────────────────────────────────────────────────

def _context(text: str, start: int, end: int, width: int = 60) -> str:
    lo, hi = max(0, start - width), min(len(text), end + width)
    prefix = "…" if lo > 0 else ""
    suffix = "…" if hi < len(text) else ""
    return f"{prefix}{_norm_generic(text[lo:hi])}{suffix}"


class _SpanClaims:
    """Tracks claimed character spans so specific patterns beat generic ones."""

    def __init__(self) -> None:
        self._spans: list[tuple[int, int]] = []

    def free(self, start: int, end: int) -> bool:
        return not any(start < e and end > s for s, e in self._spans)

    def claim(self, start: int, end: int) -> None:
        self._spans.append((start, end))


def extract_entities(text: str, doc_id: str = "DOC-1") -> list[Mention]:
    """Extract every recognised entity from one document.

    Patterns run most-specific-first and claim their character spans, so a
    15-digit IMEI is never re-read as a phone number, and a labelled account
    number is never re-read as an IMEI.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    text = text[: NLP_PARAMS["max_doc_chars"]]

    claims = _SpanClaims()
    out: list[Mention] = []

    def add(entity_type: str, raw: str, start: int, end: int, confidence: float) -> None:
        raw = raw.strip()
        if not raw or not claims.free(start, end):
            return
        value = normalise_value(entity_type, raw)
        if not value:
            return
        claims.claim(start, end)
        out.append(Mention(
            doc_id=doc_id, entity_id=entity_key(entity_type, value),
            entity_type=entity_type, raw=raw, value=value,
            start=start, end=end, confidence=round(confidence, 3),
            context=_context(text, start, end)))

    # ── Tier 1: unambiguous structured identifiers ────────────────────────────
    for m in _VEHICLE_RE.finditer(text):
        add("VEHICLE", m.group(0), m.start(), m.end(), 0.94)

    for m in _CRYPTO_RE.finditer(text):
        add("CRYPTO_WALLET", m.group(0), m.start(), m.end(), 0.92)

    for m in _IFSC_RE.finditer(text):
        add("BANK_ACCOUNT", m.group(1), m.start(1), m.end(1), 0.90)

    # Labelled account numbers run *before* IMEI: an explicit "a/c" cue is stronger
    # evidence than a 15-digit length coincidence.
    for m in _ACCOUNT_RE.finditer(text):
        add("BANK_ACCOUNT", m.group(1), m.start(1), m.end(1), 0.88)

    for m in _IMEI_RE.finditer(text):
        # Luhn-valid IMEIs score higher; the pattern alone is already strong.
        conf = 0.96 if _luhn_ok(m.group(1)) else 0.80
        add("IMEI", m.group(1), m.start(1), m.end(1), conf)

    for m in _PHONE_RE.finditer(text):
        add("PHONE", m.group(1), m.start(1), m.end(1), 0.93)

    for m in _EMAIL_RE.finditer(text):
        add("EMAIL", m.group(0), m.start(), m.end(), 0.95)

    for m in _HANDLE_RE.finditer(text):
        add("EMAIL", f"@{m.group(1)}", m.start(), m.end(), 0.70)

    for m in _CURRENCY_RE.finditer(text):
        add("CURRENCY", m.group(0), m.start(), m.end(), 0.90)

    # ── Tier 2: gazetteer terms ───────────────────────────────────────────────
    lowered = text.lower()
    for term_set, kind, conf in ((WEAPON_TERMS, "WEAPON", 0.86),
                                 (CONTRABAND_TERMS, "CONTRABAND", 0.86)):
        # Longest term first, so "country-made pistol" claims its span before the
        # substring "pistol" can be reported as a second, redundant mention.
        for term in sorted(term_set, key=len, reverse=True):
            pat = rf"(?:(?<=\W)|(?<=^)){re.escape(term)}(?:(?=\W)|(?=$))"
            for m in re.finditer(pat, lowered):
                add(kind, text[m.start():m.end()], m.start(), m.end(), conf)

    # ── Tier 3: organisations (suffix-driven, so quite reliable) ───────────────
    for m in _ORG_RE.finditer(text):
        head = m.group(1)
        if head.lower().split()[-1] in PERSON_STOPWORDS and len(head.split()) == 1:
            continue
        add("ORGANISATION", m.group(0), m.start(), m.end(), 0.82)

    # ── Tier 4: cued persons and locations ────────────────────────────────────
    for m in _PERSON_CUED_RE.finditer(text):
        cand = m.group(1)
        if _plausible_person(cand):
            add("PERSON", cand, m.start(1), m.end(1), 0.88)

    # Devanagari / Hindi cued persons
    for m in _HINDI_PERSON_RE.finditer(text):
        cand_raw = m.group(1).strip()
        tokens = cand_raw.split()
        while tokens and tokens[-1].lower() in PERSON_STOPWORDS:
            tokens.pop()
        if tokens:
            cand = " ".join(tokens)
            cand_start = m.start(1)
            cand_end = cand_start + len(cand)
            if _plausible_person(cand):
                add("PERSON", cand, cand_start, cand_end, 0.88)

    for m in _LOCATION_CUED_RE.finditer(text):
        cand = _norm_generic(m.group(1))
        conf = 0.90 if cand.lower() in LOCATION_GAZETTEER else 0.62
        if cand.lower() in LOCATION_GAZETTEER or _plausible_location(cand):
            add("LOCATION", cand, m.start(1), m.end(1), conf)

    # Devanagari / Hindi cued locations
    for m in _HINDI_LOCATION_RE.finditer(text):
        cand_raw = _norm_generic(m.group(1))
        tokens = cand_raw.split()
        while tokens and tokens[-1].lower() in PERSON_STOPWORDS:
            tokens.pop()
        if tokens:
            cand = " ".join(tokens)
            cand_start = m.start(1)
            cand_end = cand_start + len(cand)
            add("LOCATION", cand, cand_start, cand_end, 0.84)

    # Gazetteer locations anywhere in the text (catches "... shifted to Siliguri.").
    for name in sorted(LOCATION_GAZETTEER, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(name)}\b", lowered):
            add("LOCATION", text[m.start():m.end()], m.start(), m.end(), 0.85)

    # ── Tier 5: bare capitalised name sequences (lowest confidence) ────────────
    for m in _PERSON_BARE_RE.finditer(text):
        cand = m.group(1)
        if _plausible_person(cand):
            add("PERSON", cand, m.start(1), m.end(1), 0.58)

    floor = NLP_PARAMS["min_confidence"]
    return [m for m in out if m.confidence >= floor]


def _luhn_ok(digits: str) -> bool:
    """Luhn checksum — a genuine IMEI passes, a random 15-digit run usually doesn't."""
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = ord(ch) - 48
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _is_valid_token(t: str) -> bool:
    return t.isalpha() or bool(re.match(r"^[\u0900-\u097F]+$", t))


def _plausible_person(candidate: str) -> bool:
    """Reject the common false positives: stopwords, places, org heads."""
    cand = _norm_generic(candidate)
    if len(cand) < 3:
        return False
    tokens = cand.lower().split()
    if any(t in PERSON_STOPWORDS for t in tokens):
        return False
    if cand.lower() in LOCATION_GAZETTEER:
        return False
    if any(t in ORG_SUFFIXES for t in tokens):
        return False
    # A person name is 1–3 tokens of letters (Latin or Devanagari).
    return all(_is_valid_token(t) for t in tokens) and 1 <= len(tokens) <= 3


def _plausible_location(candidate: str) -> bool:
    cand = _norm_generic(candidate)
    tokens = cand.lower().split()
    if not tokens or len(cand) < 3:
        return False
    if any(t in PERSON_STOPWORDS for t in tokens if t not in
           {"nagar", "road", "colony", "market", "sector", "block", "chowk", "district"}):
        return False
    return all(t.replace("-", "").isalnum() or bool(re.match(r"^[\u0900-\u097F]+$", t)) for t in tokens)


# ──────────────────────────────────────────────────────────────────────────────
# Corpus-level extraction and co-occurrence linking
# ──────────────────────────────────────────────────────────────────────────────

def extract_corpus(documents: dict[str, str] | pd.DataFrame,
                   *, text_column: str = "text",
                   id_column: str = "doc_id") -> ExtractionResult:
    """Run extraction across many documents and build the co-occurrence graph.

    Args:
        documents: ``{doc_id: text}`` or a DataFrame with id/text columns.
        text_column, id_column: column names when a DataFrame is passed.
    """
    if isinstance(documents, pd.DataFrame):
        if text_column not in documents.columns:
            raise ValueError(f"text column {text_column!r} not found in frame")
        ids = (documents[id_column].astype(str) if id_column in documents.columns
               else pd.Series([f"DOC-{i + 1}" for i in range(len(documents))]))
        docs = dict(zip(ids, documents[text_column].astype(str)))
    else:
        docs = {str(k): str(v) for k, v in documents.items()}

    warnings: list[str] = []
    all_mentions: list[Mention] = []
    doc_rows: list[dict] = []

    for doc_id, text in docs.items():
        mentions = extract_entities(text, doc_id=doc_id)
        all_mentions.extend(mentions)
        doc_rows.append({
            "doc_id": doc_id,
            "n_chars": len(text or ""),
            "n_mentions": len(mentions),
            "n_entities": len({m.entity_id for m in mentions}),
        })

    if not all_mentions:
        empty_cols = ["doc_id", "entity_id", "entity_type", "raw", "value",
                      "start", "end", "confidence", "context"]
        return ExtractionResult(
            entities=pd.DataFrame(columns=["entity_id", "entity_type", "value", "display",
                                           "label", "n_mentions", "n_documents",
                                           "confidence", "documents"]),
            mentions=pd.DataFrame(columns=empty_cols),
            links=pd.DataFrame(columns=["source_id", "target_id", "strength",
                                        "n_documents", "source_type", "target_type"]),
            documents=pd.DataFrame(doc_rows),
            stats={"documents": len(docs), "entities": 0, "mentions": 0, "links": 0},
            warnings=["No entities were extracted — check that the text column holds "
                      "narrative text rather than codes or identifiers."])

    mentions_df = pd.DataFrame([vars(m) for m in all_mentions])

    # ── Aggregate mentions into entities ──────────────────────────────────────
    grouped = mentions_df.groupby(["entity_id", "entity_type", "value"], as_index=False).agg(
        n_mentions=("doc_id", "size"),
        n_documents=("doc_id", "nunique"),
        confidence=("confidence", "max"),
        documents=("doc_id", lambda s: sorted(set(s))),
        raw=("raw", "first"),
    )
    grouped["display"] = [mask_value(t, v) for t, v in
                          zip(grouped["entity_type"], grouped["value"])]
    grouped["label"] = grouped["entity_type"].map(ENTITY_LABELS).fillna(grouped["entity_type"])
    # Corroboration across documents raises confidence, capped below certainty.
    grouped["confidence"] = (grouped["confidence"]
                             + 0.04 * (grouped["n_documents"] - 1)).clip(upper=0.99).round(3)
    entities_df = grouped.sort_values(
        ["n_documents", "n_mentions", "confidence"], ascending=False).reset_index(drop=True)

    # ── Co-occurrence links (parties only — see LINKABLE_ENTITY_TYPES) ────────
    pair_docs: dict[tuple[str, str], set[str]] = {}
    cap = NLP_PARAMS["max_entities_per_doc"]
    skipped = 0
    linkable = mentions_df[mentions_df["entity_type"].isin(LINKABLE_ENTITY_TYPES)]
    for doc_id, grp in linkable.groupby("doc_id"):
        ids = sorted(set(grp["entity_id"]))
        if len(ids) > cap:
            # Same guard as the co-accused clique cap: a bulk dump would otherwise
            # manufacture thousands of meaningless links.
            skipped += 1
            continue
        for a, b in combinations(ids, 2):
            pair_docs.setdefault((a, b), set()).add(doc_id)

    if skipped:
        warnings.append(
            f"{skipped} document(s) mentioned more than {cap} entities and were excluded "
            "from co-occurrence linking to avoid manufacturing a dense clique.")

    types = dict(zip(entities_df["entity_id"], entities_df["entity_type"]))
    base, bonus, ceiling = (NLP_PARAMS["cooccurrence_strength"],
                            NLP_PARAMS["cooccurrence_bonus"],
                            NLP_PARAMS["cooccurrence_max"])
    link_rows = [{
        "source_id": a,
        "target_id": b,
        "strength": round(min(base + bonus * (len(ds) - 1), ceiling), 3),
        "n_documents": len(ds),
        "documents": sorted(ds),
        "source_type": types.get(a, "?"),
        "target_type": types.get(b, "?"),
    } for (a, b), ds in pair_docs.items()]

    links_df = (pd.DataFrame(link_rows).sort_values(["n_documents", "strength"],
                                                    ascending=False).reset_index(drop=True)
                if link_rows else
                pd.DataFrame(columns=["source_id", "target_id", "strength", "n_documents",
                                      "documents", "source_type", "target_type"]))

    stats = {
        "documents": len(docs),
        "entities": len(entities_df),
        "mentions": len(mentions_df),
        "links": len(links_df),
        "by_type": entities_df["entity_type"].value_counts().to_dict(),
        "mean_confidence": round(float(entities_df["confidence"].mean()), 3),
    }
    return ExtractionResult(entities_df, mentions_df, links_df,
                            pd.DataFrame(doc_rows), stats, warnings)


def front_business_candidates(result: ExtractionResult) -> pd.DataFrame:
    """Organisations whose name suffix matches a known front-business archetype.

    Placement agencies, spas, bars and 'travels' recur as fronts in NCRB Women
    Safety casework, so they are surfaced separately rather than buried in the
    organisation list.
    """
    orgs = result.of_type("ORGANISATION")
    if not len(orgs):
        return pd.DataFrame(columns=["entity_id", "value", "n_documents", "matched_suffix"])
    rows = []
    for _, r in orgs.iterrows():
        tokens = str(r["value"]).lower().replace(".", " ").split()
        hit = next((t for t in tokens if t in FRONT_BUSINESS_SUFFIXES), None)
        if hit:
            rows.append({"entity_id": r["entity_id"], "value": r["value"],
                         "n_documents": r["n_documents"], "matched_suffix": hit})
    return (pd.DataFrame(rows).sort_values("n_documents", ascending=False).reset_index(drop=True)
            if rows else
            pd.DataFrame(columns=["entity_id", "value", "n_documents", "matched_suffix"]))
