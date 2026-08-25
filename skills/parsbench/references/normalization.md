# Persian normalization rules

One equivalence chain runs under every text check (`contains=`,
`not_contains=`) and every tool-argument comparison. Always on, nothing to
configure. Use these rules to predict what a golden will and won't match.

## What is equated

**Codepoints and digits.** Arabic twins map to Persian forms (`ي` → `ی`,
`ك` → `ک`, `ة` → `ه`). All three digit scripts unify: `۱۲۳` (Persian),
`١٢٣` (Arabic-Indic), and `123` compare equal. Thousands separators (`٬`,
`,`) drop out: `۲٬۵۰۰٬۰۰۰` == `2500000`. The Arabic decimal separator reads
as a point: `۲٫۵` == `2.5`.

**ZWNJ and spacing.** ZWNJ reads as a space, whitespace collapses, and a
space inside a needle also matches the fully joined form: «می‌روم»,
«می روم», «میروم» all match each other. Needles without spaces never merge
words, so short needles can't false-positive across word boundaries.

**Money.** Amounts parse into rials before comparing, across scale words
(هزار، میلیون، میلیارد) and units. «۲۵۰ هزار تومان», «۲٬۵۰۰٬۰۰۰ ریال», and
«۲/۵ میلیون تومان» all state the same amount, including compound forms like
«۲ میلیون و ۵۰۰ هزار» and both Persian decimal notations (`۲٫۵` and `۲/۵`).
When one side has no currency word, either the rial or the toman reading is
accepted. A bare number in the text only matches at the exact rial value, so
order ids and phone numbers can't satisfy a money check.

**Dates.** Numeric dates parse in either calendar; a year below 1600 reads
as Jalali. A tool argument of `"1405-07-05"` equals `"2026-09-27"`, and
`contains=["1405/07/05"]` matches an answer stating the Gregorian date.
Times after the date (`2026-09-27T08:00`) are tolerated. **Persian month
names («۵ مهر») are NOT parsed by this layer.** The user simulator uses them
as a trap: the bot must convert them to a numeric date before calling a
tool, which this layer then checks. Never write goldens expecting month-name
equivalence in text or tool args.

## Where it applies

- `contains=` / `not_contains=`: each needle tries, in order, normalized
  substring, then money equivalence, then date equivalence.
- Tool arguments: each expected argument compares as date, then number, then
  normalized text.
- Symmetrically: `not_contains=["رزرو شد"]` also catches «رزرو  شد» with odd
  spacing, and a forbidden amount is caught in any unit.

## Using it directly

For custom `check=` functions or the user's own tests:

```python
from parsbench.appeval.normalize import (
    normalize,             # canonical form of a string
    contains_normalized,   # substring check under normalization
    numbers_equal,         # '۲۵۰ هزار تومان' == '2500000 ریال' -> True
    amount_in,             # is this amount stated anywhere in the text?
    dates_equal,           # '1405-07-05' == '2026-09-27' -> True
    date_in,               # is this date mentioned, either calendar?
    values_equal,          # the tool-argument chain: date -> number -> text
)

amount_in("قیمت ۲٬۵۰۰٬۰۰۰ ریال است", "250 هزار تومان")   # True
```

The module is dependency-free (no hazm).
