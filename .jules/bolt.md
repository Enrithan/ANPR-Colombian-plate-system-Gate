## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2024-05-26 - Pre-computed Frozensets for O(1) Lookups in Tight Loops
**Learning:** `license_complies_format` and `format_license` were evaluating OCR characters by iterating over strings and performing `in` checks on dynamic lists and dictionary `.keys()`. Inside tight inner loops like OCR processing, this repeated construction and `O(N)` lookup scales poorly. Using pre-computed `frozenset` collections and unrolled index checks improved validation speed by ~3.4x, and combining dictionary `.get()` with `"".join()` instead of explicit string concatenation improved formatting speed by ~1.6x.
**Action:** When evaluating characters or small, fixed length strings in tight processing loops, replace dynamic list/dictionary keys evaluation with pre-computed `frozenset` objects for O(1) complexity, and replace explicit string concatenation loops with dictionary `.get()` accompanied by `" ".join()`.
