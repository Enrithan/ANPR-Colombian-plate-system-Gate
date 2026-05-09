## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2025-03-21 - Precomputing Data Structures & Regular Expressions
**Learning:** Initializing variables, casting datatypes (lists/dicts), parsing `string` constants and iterating inside `for` loops within very fast execution pathways (like plate validation per frame) has an enormous cumulative performance cost (0.22s to 0.04s or 4x difference). Pre-compiling one combined regex (`|`) is similarly 4x faster than looping over a list of independent compiled expressions.
**Action:** Move instantiation of objects, lists, sets, and constants out of tight loops. Use module-level variables with O(1) set-lookups and combine regular expressions where possible to skip Python iteration overhead.

## 2025-03-22 - Pre-computing Static Numpy Arrays in OCR Loops
**Learning:** In high-frequency computer vision loops (e.g., OCR inference applied to every frame), repeatedly instantiating static `np.array` objects for transformations like `dst` points or convolution kernels introduces severe allocation overhead. Memory benchmarking showed that instantiating these arrays natively takes roughly an order of magnitude more time than accessing an already instantiated module-level array.
**Action:** Always extract and pre-compute static `np.array` definitions (like `getPerspectiveTransform` boundaries and `filter2D` kernels) as module-level constants to skip object creation time and optimize fast-path logic.

## 2025-03-22 - Chained `.replace` Operations vs For Loops
**Learning:** For applying multiple specific character removals to short strings (like OCR text output), chaining string `.replace(char, '')` calls sequentially is empirically ~30% faster than looping over a list of characters and iteratively applying the `.replace()` inside the loop. The absolute execution time is low, but in OCR pipelines over many frames, the iteration overhead of the `for` loop dominates.
**Action:** Use chained `.replace()` calls instead of iterative character lists for multi-character replacements in short string performance-sensitive paths.
