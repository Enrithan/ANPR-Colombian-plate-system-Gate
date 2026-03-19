## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Perspective Transformations
**Learning:** Geometric transformations like `cv2.warpPerspective` operate on each channel independently. Converting the image to grayscale *before* applying the warp instead of after speeds up the affine transformation and saves memory bandwidth. In cases where the destination resolution is static, calculate boundaries only once instead of recalculating on every iteration.
**Action:** Always convert image down to single-channel grayscale earlier in the vision pipeline, and remove redundant color conversions later on.
