## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2024-05-26 - Optimize String Validation inside Tight Loops
**Learning:** Checking string membership against dynamic lists (`in ['0', '1', ...]`) or dictionary views (`in dict.keys()`) inside high-frequency loops (like OCR post-processing) causes significant CPU overhead due to O(N) linear lookups and constant memory reallocation.
**Action:** Always extract static validation lists into module-level `frozenset` collections for O(1) lookups. Additionally, replace iterative string concatenation (`+=`) with list building (`.append()`) and `"".join()` for O(N) string construction instead of O(N^2).
