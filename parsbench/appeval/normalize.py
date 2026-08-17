"""Persian-aware normalization — the always-on layer under every check.

Handles the traps that make English-shaped evals silently fail on Persian:
Arabic-vs-Persian codepoints, three digit scripts, ZWNJ variants,
Jalali vs Gregorian dates, and هزار/میلیون + rial/toman amounts.

Deliberately dependency-free (no hazm import): this layer runs inside every
check, so it stays a handful of table lookups and small regexes.
"""

import re

ZWNJ = "‌"

# Arabic codepoints that render identically to their Persian twins + digit scripts.
_CHAR_MAP = str.maketrans(
    {
        "ي": "ی",  # ي ARABIC YEH
        "ى": "ی",  # ى ALEF MAKSURA
        "ك": "ک",  # ك ARABIC KAF
        "ۀ": "ه",  # ۀ
        "ة": "ه",  # ة
        "٬": "",   # ARABIC THOUSANDS SEPARATOR — ۲٬۵۰۰٬۰۰۰ == 2500000
        "٫": ".",  # ARABIC DECIMAL SEPARATOR — ۲٫۵ == 2.5
        **{chr(0x06F0 + i): str(i) for i in range(10)},  # ۰-۹ Persian
        **{chr(0x0660 + i): str(i) for i in range(10)},  # ٠-٩ Arabic-Indic
    }
)

# thousands commas between digit groups only — «۲,۵» stays untouched
_DIGIT_COMMA = re.compile(r"(?<=\d)[,،](?=\d{3})")


def normalize(text: str) -> str:
    """Unify codepoints/digits, read ZWNJ as a space, drop thousands commas,
    collapse whitespace."""
    text = str(text).translate(_CHAR_MAP).replace(ZWNJ, " ")
    text = _DIGIT_COMMA.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_normalized(haystack: str, needle: str) -> bool:
    """Substring check that survives digit scripts, ZWNJ and spacing variants."""
    return _contains_norm(normalize(haystack), needle)


def _contains_norm(h: str, needle: str) -> bool:
    n = normalize(needle)
    if n in h:
        return True
    # a space in the needle may be a ZWNJ-join or fully joined in the text
    # (می‌روم / می روم / میروم); needles without spaces never merge words.
    if " " not in n:
        return False
    pattern = r"\s?".join(re.escape(part) for part in n.split(" "))
    return re.search(pattern, h) is not None


# --- Numbers & currency ------------------------------------------------------

_SCALES = {"هزار": 1_000, "میلیون": 1_000_000, "میلیارد": 1_000_000_000}
# \w covers Persian letters, so (?!\w) keeps «هزار» from matching inside «هزاران».
_NUM = r"\d+(?:[./]\d+)?"
_SCALED = rf"{_NUM}\s*(?:هزار|میلیون|میلیارد)(?!\w)"
# one amount: scaled terms joined by «و», optionally ending in a bare term
# («۲ میلیون و ۵۰۰ هزار», «۲۵۰ هزار و ۵۰۰»), or a single bare number.
_AMOUNT_BODY = rf"(?:{_SCALED}(?:\s*و\s*{_SCALED})*(?:\s*و\s*{_NUM}(?!\d))?|{_NUM})"
_AMOUNT_EXPR = re.compile(
    rf"(?<![\w.]){_AMOUNT_BODY}(?:\s*(?:تومان|تومن|ریال)(?!\w))?"
)
_TERM_ITER = re.compile(r"(\d+(?:[./]\d+)?)\s*(هزار|میلیون|میلیارد)?")
_CURRENCY = re.compile(r"تومان|تومن|ریال")


def _eval_amount(expr: str) -> tuple[float, str | None, bool]:
    """Evaluate one matched amount expression → (value, unit|None, has_scale).
    The value is in rials whenever a currency word is present."""
    total, has_scale = 0.0, False
    for num, scale in _TERM_ITER.findall(expr):
        value = float(num.replace("/", "."))  # ۲/۵ is the Persian decimal form
        if scale:
            has_scale = True
            value *= _SCALES[scale]
        total += value
    currency = _CURRENCY.search(expr)
    if currency:
        if currency.group(0) != "ریال":
            total *= 10
        return total, "rial", has_scale
    return total, None, has_scale


def _parse_amount(value) -> tuple[float, str | None, bool] | None:
    """Strict parse: the whole string must be a single amount expression.
    Phones, IDs, addresses and other digit-bearing prose return None."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), None, False
    text = normalize(value)
    if not _AMOUNT_EXPR.fullmatch(text):
        return None
    return _eval_amount(text)


def parse_number(value) -> tuple[float, str | None] | None:
    """Parse '۲۵۰ هزار تومان' → (2_500_000.0, 'rial'). Returns (value, unit|None)."""
    parsed = _parse_amount(value)
    return None if parsed is None else (parsed[0], parsed[1])


def numbers_equal(a, b) -> bool:
    pa, pb = _parse_amount(a), _parse_amount(b)
    if pa is None or pb is None:
        return False
    (va, ua, _), (vb, ub, _) = pa, pb
    if ua and ub:
        return va == vb
    # unit missing on one side — accept either rial/toman reading
    if ua or ub:
        return va == vb or va == vb * 10 or vb == va * 10
    return va == vb


def amount_in(haystack: str, needle) -> bool:
    """Does the text state this amount, in any unit/scale/digit-script?
    amount_in('قیمت ۲٬۵۰۰٬۰۰۰ ریال است', '250 هزار تومان') → True."""
    return _amount_in_norm(normalize(haystack), needle)


def _amount_in_norm(h: str, needle) -> bool:
    parsed = _parse_amount(needle)
    if parsed is None:
        return False
    value, unit, has_scale = parsed
    if unit is None and not has_scale:
        return False  # not recognizably money — that's the substring check's job
    for m in _AMOUNT_EXPR.finditer(h):
        found, found_unit, found_scale = _eval_amount(m.group(0))
        if unit and found_unit:
            if found == value:
                return True
        elif found_unit or found_scale:
            # unit missing on one side — accept either rial/toman reading
            if found in (value, value * 10) or value in (found, found * 10):
                return True
        elif found == value:
            # bare number in the text: exact rial-value match only, so order
            # ids and other stray digits can't satisfy a money check
            return True
    return False


# --- Dates (Jalali <-> Gregorian, standard jalaali arithmetic) ---------------


def jalali_to_gregorian(jy: int, jm: int, jd: int) -> tuple[int, int, int]:
    jy += 1595
    days = -355668 + 365 * jy + (jy // 33) * 8 + ((jy % 33) + 3) // 4 + jd
    days += (jm - 1) * 31 if jm < 7 else (jm - 7) * 30 + 186
    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1
    leap = 1 if (gy % 4 == 0 and gy % 100 != 0) or gy % 400 == 0 else 0
    month_days = [31, 28 + leap, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    gm = 1
    for length in month_days:
        if gd <= length:
            break
        gd -= length
        gm += 1
    return gy, gm, gd


def gregorian_to_jalali(gy: int, gm: int, gd: int) -> tuple[int, int, int]:
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    if gy > 1600:
        jy, gy = 979, gy - 1600
    else:
        jy, gy = 0, gy - 621
    gy2 = gy + 1 if gm > 2 else gy
    days = (
        365 * gy
        + (gy2 + 3) // 4
        - (gy2 + 99) // 100
        + (gy2 + 399) // 400
        - 80
        + gd
        + g_d_m[gm - 1]
    )
    jy += 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        return jy, 1 + days // 31, 1 + days % 31
    return jy, 7 + (days - 186) // 30, 1 + (days - 186) % 30


# digit lookarounds keep reference codes like 2026-01-123456 from reading as dates
_DATE_SCAN = re.compile(r"(?<!\d)(\d{3,4})[/-](\d{1,2})[/-](\d{1,2})(?!\d)")
# a date-valued string: one date, optionally with a time tail, nothing else
_DATE_ONLY = re.compile(
    r"(\d{3,4})[/-](\d{1,2})[/-](\d{1,2})"
    r"(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?Z?)?$"
)


def _to_gregorian(y: int, mo: int, d: int) -> tuple[int, int, int] | None:
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return jalali_to_gregorian(y, mo, d) if y < 1600 else (y, mo, d)


def parse_date(value) -> tuple[int, int, int] | None:
    """Parse a date(-time) string to a Gregorian (y, m, d). Year <1600 → Jalali.
    The whole string must be the date — ranges and prose return None."""
    m = _DATE_ONLY.match(normalize(value))
    if not m:
        return None
    return _to_gregorian(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def dates_equal(a, b) -> bool:
    pa, pb = parse_date(a), parse_date(b)
    return pa is not None and pa == pb


def date_in(haystack: str, needle) -> bool:
    """Does the text mention this date, in either calendar?"""
    return _date_in_norm(normalize(haystack), needle)


def _date_in_norm(h: str, needle) -> bool:
    want = parse_date(needle)
    if want is None:
        return False
    for m in _DATE_SCAN.finditer(h):
        got = _to_gregorian(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if got == want:
            return True
    return False


def text_matches(haystack: str, needle, *, normalized: bool = False) -> bool:
    """Equivalence chain for text expectations (contains / not_contains):
    normalized substring, then money-equivalence, then calendar-equivalence.
    Pass normalized=True when the haystack is already normalize()d."""
    h = haystack if normalized else normalize(haystack)
    n = needle if isinstance(needle, str) else str(needle)
    if _contains_norm(h, n):
        return True
    if _amount_in_norm(h, n):
        return True
    return _date_in_norm(h, n)


def values_equal(a, b) -> bool:
    """Equivalence chain for tool arguments: date → number → normalized string."""
    da, db = parse_date(a), parse_date(b)
    if da or db:
        return da == db
    if numbers_equal(a, b):
        return True
    return normalize(a) == normalize(b)
