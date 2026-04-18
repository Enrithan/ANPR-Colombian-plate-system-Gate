## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2024-05-26 - String manipulation and dictionary checks in tight loops
**Learning:** In highly iterated utility functions (like validating or mapping OCR character strings), iterative string concatenation and list-based membership checks (`in list` or `.keys()`) introduce significant overhead. Using `frozenset` for O(1) membership checks and aggregating characters into a list before a single `"".join()` call yielded a measurable performance improvement (~1.1x to 1.35x speedup for formatting and validation).
**Action:** When writing or optimizing tight string parsing loops, avoid list-based `in` checks and repeated `+=` string concatenations. Instead, pre-compute `frozenset` collections and build strings with `"".join()`. Avoid checking `.keys()` explicitly on dictionaries.
