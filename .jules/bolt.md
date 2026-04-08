## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2024-05-26 - Optimize String Validation and Formatting in Tight Loops
**Learning:** Checking characters against literal lists (like `['0', '1', '2', ...]`) and performing iterative string concatenation (`+=`) inside frequent loops (such as OCR post-processing) introduces significant overhead. Converting these checks to use pre-computed `frozenset` collections reduces validation to O(1) time. Additionally, using Python's `list.append` and `"".join()` alongside dictionary `.get()` mappings dramatically accelerates string construction.
**Action:** Always use module-level `frozenset` objects for repetitive character validation lookups, and prefer `.get()` mapping with `"".join()` over explicit `if/else` checks with iterative string concatenation (`+=`) inside tight post-processing loops.
