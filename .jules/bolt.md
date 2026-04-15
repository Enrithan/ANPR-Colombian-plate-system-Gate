## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2024-05-26 - Eliminate dictionary view object lookups in hot paths
**Learning:** Using `in dict.keys()` in Python creates a dictionary view object and forces a linear O(N) scan on older Python versions, or at best adds unnecessary method call overhead. Doing simply `in dict` takes advantage of direct O(1) hash map membership testing. This micro-optimization matters when called thousands of times inside an OCR string validation loop.
**Action:** Always check dictionary membership using `key in dict`, never `key in dict.keys()`.

## 2024-05-26 - Combine regexes for tight loops
**Learning:** In string validation checks that run very frequently, iterating through a list of multiple compiled regular expressions is measurably slower than running a single combined regex using the OR operator `|` (e.g., `(pattern1|pattern2)`).
**Action:** Always combine mutually exclusive string validation regexes into a single pattern to avoid loop and multi-match overhead in performance-critical code.
