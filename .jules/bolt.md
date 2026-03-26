## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2025-02-12 - Optimize String Validation and Formatting in OCR Post-Processing
**Learning:** String validation and formatting inside tight loops (like OCR post-processing) can become a bottleneck when using iterative string concatenation and list-based character checks. Lists check items in O(N) time, and `+=` string building creates entirely new string objects every time.
**Action:** Use pre-computed `frozenset` collections for O(1) character lookups, dictionary `.get()` for safe mapping, and `"".join()` for fast string construction instead of loop-based concatenation.
