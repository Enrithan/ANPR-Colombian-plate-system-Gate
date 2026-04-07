## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2024-05-26 - Optimize string validation and formatting in tight loops
**Learning:** In tight OCR post-processing loops (`util.py`), iterating over strings and validating characters against `list.keys()` or lists is highly inefficient. Likewise, iterative string concatenation (`+=`) becomes an O(N^2) bottleneck. Combining regular expression patterns into a single non-capturing group `(?:...|...|...)` is also ~35% faster than looping over multiple compiled regexes.
**Action:** Always use pre-computed `frozenset` collections for O(1) character lookup validations. Construct strings using list builders and `"".join()` with `.get()` mapping instead of loop concatenation. Use combined regex expressions when validating a string against multiple potential patterns.
