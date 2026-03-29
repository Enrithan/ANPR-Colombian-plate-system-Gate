## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2024-05-26 - Pre-computed Sets and Combined Regex for O(1) Operations
**Learning:** String validation and pattern matching functions run thousands of times per second inside the tight OCR frame processing loops. Iterating over lists or multiple compiled regex patterns causes micro-delays that compound rapidly. Using Python's `in` operator on lists is O(n), whereas on `frozenset` it is O(1). Additionally, executing a single combined regex with logical OR (`|`) is roughly 2x faster than a loop executing multiple distinct patterns. Iterative string concatenation (`+=`) is also a performance trap compared to collecting characters in a list and using `"".join()`.
**Action:** When implementing string validation or formatting inside tight loops, strictly use pre-computed module-level `frozenset` objects for lookups, combined single regex patterns, and `"".join()` for string construction.
