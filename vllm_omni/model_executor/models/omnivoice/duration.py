

import bisect
import unicodedata
from functools import lru_cache


class RuleDurationEstimator:
    def __init__(self):
        self.weights = {
            "cjk": 3.0,
            "hangul": 2.5,
            "kana": 2.2,
            "ethiopic": 3.0,
            "yi": 3.0,
            "indic": 1.8,
            "thai_lao": 1.5,
            "khmer_myanmar": 1.8,
            "arabic": 1.5,
            "hebrew": 1.5,
            "latin": 1.0,
            "cyrillic": 1.0,
            "greek": 1.0,
            "armenian": 1.0,
            "georgian": 1.0,
            "punctuation": 0.5,
            "space": 0.2,
            "digit": 3.5,
            "mark": 0.0,
            "default": 1.0,
        }

        self.ranges = [
            (0x02AF, "latin"),
            (0x03FF, "greek"),
            (0x052F, "cyrillic"),
            (0x058F, "armenian"),
            (0x05FF, "hebrew"),
            (0x077F, "arabic"),
            (0x089F, "arabic"),
            (0x08FF, "arabic"),
            (0x097F, "indic"),
            (0x09FF, "indic"),
            (0x0A7F, "indic"),
            (0x0AFF, "indic"),
            (0x0B7F, "indic"),
            (0x0BFF, "indic"),
            (0x0C7F, "indic"),
            (0x0CFF, "indic"),
            (0x0D7F, "indic"),
            (0x0DFF, "indic"),
            (0x0EFF, "thai_lao"),
            (0x0FFF, "indic"),
            (0x109F, "khmer_myanmar"),
            (0x10FF, "georgian"),
            (0x11FF, "hangul"),
            (0x137F, "ethiopic"),
            (0x139F, "ethiopic"),
            (0x13FF, "default"),
            (0x167F, "default"),
            (0x169F, "default"),
            (0x16FF, "default"),
            (0x171F, "default"),
            (0x173F, "default"),
            (0x175F, "default"),
            (0x177F, "default"),
            (0x17FF, "khmer_myanmar"),
            (0x18AF, "default"),
            (0x18FF, "default"),
            (0x194F, "indic"),
            (0x19DF, "indic"),
            (0x19FF, "khmer_myanmar"),
            (0x1A1F, "indic"),
            (0x1AAF, "indic"),
            (0x1B7F, "indic"),
            (0x1BBF, "indic"),
            (0x1BFF, "indic"),
            (0x1C4F, "indic"),
            (0x1C7F, "indic"),
            (0x1C8F, "cyrillic"),
            (0x1CBF, "georgian"),
            (0x1CCF, "indic"),
            (0x1CFF, "indic"),
            (0x1D7F, "latin"),
            (0x1DBF, "latin"),
            (0x1DFF, "default"),
            (0x1EFF, "latin"),
            (0x309F, "kana"),
            (0x30FF, "kana"),
            (0x312F, "cjk"),
            (0x318F, "hangul"),
            (0x9FFF, "cjk"),
            (0xA4CF, "yi"),
            (0xA4FF, "default"),
            (0xA63F, "default"),
            (0xA69F, "cyrillic"),
            (0xA6FF, "default"),
            (0xA7FF, "latin"),
            (0xA82F, "indic"),
            (0xA87F, "default"),
            (0xA8DF, "indic"),
            (0xA8FF, "indic"),
            (0xA92F, "indic"),
            (0xA95F, "indic"),
            (0xA97F, "hangul"),
            (0xA9DF, "indic"),
            (0xA9FF, "khmer_myanmar"),
            (0xAA5F, "indic"),
            (0xAA7F, "khmer_myanmar"),
            (0xAADF, "indic"),
            (0xAAFF, "indic"),
            (0xAB2F, "ethiopic"),
            (0xAB6F, "latin"),
            (0xABBF, "default"),
            (0xABFF, "indic"),
            (0xD7AF, "hangul"),
            (0xFAFF, "cjk"),
            (0xFDFF, "arabic"),
            (0xFE6F, "default"),
            (0xFEFF, "arabic"),
            (0xFFEF, "latin"),
        ]
        self.breakpoints = [r[0] for r in self.ranges]

    @lru_cache(maxsize=4096)
    def _get_char_weight(self, char):
        code = ord(char)
        if (65 <= code <= 90) or (97 <= code <= 122):
            return self.weights["latin"]
        if code == 32:
            return self.weights["space"]

        if code == 0x0640:
            return self.weights["mark"]

        category = unicodedata.category(char)

        if category.startswith("M"):
            return self.weights["mark"]

        if category.startswith("P") or category.startswith("S"):
            return self.weights["punctuation"]

        if category.startswith("Z"):
            return self.weights["space"]

        if category.startswith("N"):
            return self.weights["digit"]

        idx = bisect.bisect_left(self.breakpoints, code)
        if idx < len(self.ranges):
            script_type = self.ranges[idx][1]
            return self.weights.get(script_type, self.weights["default"])

        if code > 0x20000:
            return self.weights["cjk"]

        return self.weights["default"]

    def calculate_total_weight(self, text):
        return sum(self._get_char_weight(c) for c in text)

    def estimate_duration(
        self,
        target_text: str,
        ref_text: str,
        ref_duration: float,
        low_threshold: float | None = 50,
        boost_strength: float = 3,
    ) -> float:
        if ref_duration <= 0 or not ref_text:
            return 0.0

        ref_weight = self.calculate_total_weight(ref_text)
        if ref_weight == 0:
            return 0.0

        speed_factor = ref_weight / ref_duration
        target_weight = self.calculate_total_weight(target_text)

        estimated_duration = target_weight / speed_factor
        if low_threshold is not None and estimated_duration < low_threshold:
            alpha = 1.0 / boost_strength
            return low_threshold * (estimated_duration / low_threshold) ** alpha
        else:
            return estimated_duration


if __name__ == "__main__":
    estimator = RuleDurationEstimator()

    ref_txt = "Hello, world."
    ref_dur = 1.5

    test_cases = [
        ("Hindi (With complex marks)", "नमस्ते दुनिया"),
        ("Arabic (With vowels)", "مَرْحَبًا بِالْعَالَم"),
        ("Vietnamese (Lots of diacritics)", "Chào thế giới"),
        ("Chinese", "你好，世界！"),
        ("Mixed Emoji", "Hello 🌍! This is fun 🎉"),
    ]

    print("--- Reference ---")
    print(f"Reference Text: '{ref_txt}'")
    print(f"Reference Duration: {ref_dur}s")
    print("-" * 30)

    for lang, txt in test_cases:
        est_time = estimator.estimate_duration(txt, ref_txt, ref_dur)
        weight = estimator.calculate_total_weight(txt)

        print(f"[{lang}]")
        print(f"Text: {txt}")
        print(f"Total Weight: {weight:.2f}")
        print(f"Estimated Duration: {est_time:.2f} s")
        print("-" * 30)
