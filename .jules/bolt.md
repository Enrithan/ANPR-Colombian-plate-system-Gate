## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2024-05-26 - Optimize string validation in tight loops
**Learning:** Checking string membership against arrays and concatenating strings iteratively in Python loops adds significant overhead in tight loops (e.g. post-processing OCR results for every frame). Pre-computing valid sets into `frozenset` objects for O(1) lookups reduces execution time by roughly 60-70%.
**Action:** Replace iterative string concatenation and list character checks with pre-computed `frozenset` collections, dictionary `.get()` mappings, and list/comprehension `"".join()` when doing high-frequency string validation.
