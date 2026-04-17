## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2024-06-25 - Avoid O(N) array checks in tight loops
**Learning:** Checking membership in lists (e.g., `in ['0', '1', ...]`) or string sequences inside tight loops (like OCR post-processing functions) incurs unnecessary O(N) overhead. Additionally, `dict.keys()` creates a dictionary view which adds a small but unnecessary overhead compared to direct `in dict` O(1) membership checks. Iterative string concatenations (`+=`) also incur O(N^2) reallocation overhead in worst-case scenarios.
**Action:** When validating characters or parsing OCR outputs, use module-level pre-computed `frozenset` objects for O(1) validation. Prefer direct `in dict` lookups, and build strings by appending to a list followed by `"".join(chars)`.
