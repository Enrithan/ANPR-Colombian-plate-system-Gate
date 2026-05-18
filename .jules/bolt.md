## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2025-03-21 - Precomputing Data Structures & Regular Expressions
**Learning:** Initializing variables, casting datatypes (lists/dicts), parsing `string` constants and iterating inside `for` loops within very fast execution pathways (like plate validation per frame) has an enormous cumulative performance cost (0.22s to 0.04s or 4x difference). Pre-compiling one combined regex (`|`) is similarly 4x faster than looping over a list of independent compiled expressions.
**Action:** Move instantiation of objects, lists, sets, and constants out of tight loops. Use module-level variables with O(1) set-lookups and combine regular expressions where possible to skip Python iteration overhead.

## 2025-03-24 - Pre-compute Static NumPy Arrays in CV Loops
**Learning:** In high-frequency computer vision loops (e.g., OCR inference or perspective transforms), instantiating static NumPy arrays (like spatial coordinates or convolution kernels) introduces severe memory allocation overhead. OpenCV functions like `cv2.getPerspectiveTransform` and `cv2.filter2D` do not mutate their array arguments, so these arrays can be safely reused.
**Action:** Always pre-compute and cache static `np.array` structures as module-level constants to eliminate allocation bottlenecks in hot code paths.

## 2025-03-24 - Chained Replace over Character Iteration
**Learning:** For short strings like license plates (6-8 characters) that require multiple distinct character removals, executing a single chained sequence of `.replace()` calls is faster and incurs less overhead than iterating over a list of characters and performing replacements within a loop.
**Action:** Use chained `.replace()` calls for simple normalization of known, short string formats.
