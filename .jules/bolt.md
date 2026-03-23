## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2024-05-26 - Optimize Regex Checks and Tight Python Loops
**Learning:** Checking multiple regex patterns in a loop is consistently slower (~30%) than checking a single, combined regex pattern using non-capturing groups `(?:...)`. Furthermore, in tight, frequent loops (like per-character OCR validation), `list` lookups and `.keys()` calls are significantly slower than looking up pre-computed `set`s. String building in tight loops also benefits from `get` with fallbacks and `str.join` over `for` loops.
**Action:** For string validations, use module-level pre-computed sets for O(1) lookups and combined regex patterns to minimize evaluation overhead in Python loops.
