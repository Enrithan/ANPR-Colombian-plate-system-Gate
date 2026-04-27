## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2025-03-21 - Precomputing Data Structures & Regular Expressions
**Learning:** Initializing variables, casting datatypes (lists/dicts), parsing `string` constants and iterating inside `for` loops within very fast execution pathways (like plate validation per frame) has an enormous cumulative performance cost (0.22s to 0.04s or 4x difference). Pre-compiling one combined regex (`|`) is similarly 4x faster than looping over a list of independent compiled expressions.
**Action:** Move instantiation of objects, lists, sets, and constants out of tight loops. Use module-level variables with O(1) set-lookups and combine regular expressions where possible to skip Python iteration overhead.
## 2026-04-27 - Pre-computing OCR String Sanitization
**Learning:** Iterating over character arrays and calling  multiple times on OCR output strings inside a tight inference loop takes roughly ~0.8s for 1 million iterations. Replacing it with  using a pre-computed mapping is roughly 3x slower (~2.0s) when creating the map on the fly, but roughly 1.5x faster (~0.5s) if the mapping is initialized in the class constructor. Pre-computing translation tables is the idiomatic way for multi-character removal.
**Action:** Use pre-computed  in class constructors and  for multi-character string sanitization, but avoid  if the translation mapping has to be recreated dynamically on every loop.

## 2024-05-26 - Pre-computing OCR String Sanitization
**Learning:** Iterating over character arrays and calling `.replace()` multiple times on OCR output strings inside a tight inference loop is slow. Replacing it with `str.translate` using a mapping is much faster, but *only* if the mapping is initialized outside the tight loop (e.g., in the class constructor). Re-creating the mapping table on the fly makes `translate` 3x slower than chained `replace` calls.
**Action:** Use pre-computed `str.maketrans` in class constructors and `str.translate` for multi-character string sanitization, but avoid `translate` if the mapping table has to be recreated dynamically on every loop.
