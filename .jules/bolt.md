## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2024-05-26 - Optimize string processing in tight loops with pre-computed sets and lists
**Learning:** String processing inside tight loops (like OCR post-processing, bounding box associations) suffers significantly from iterative string concatenation (`+=`) and checking membership using lists or dictionary keys (`in list` or `in dict.keys()`). `in list` is O(N) and string concatenation creates a new string object every time.
**Action:** Use pre-computed `frozenset` objects for O(1) membership checks of allowed characters, use `.get()` to conditionally swap characters safely from mapping dicts, and append characters to a list followed by `"".join(list)` to assemble final strings rather than concatenating them repeatedly.
