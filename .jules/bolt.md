## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2025-03-21 - Precomputing Data Structures & Regular Expressions
**Learning:** Initializing variables, casting datatypes (lists/dicts), parsing `string` constants and iterating inside `for` loops within very fast execution pathways (like plate validation per frame) has an enormous cumulative performance cost (0.22s to 0.04s or 4x difference). Pre-compiling one combined regex (`|`) is similarly 4x faster than looping over a list of independent compiled expressions.
**Action:** Move instantiation of objects, lists, sets, and constants out of tight loops. Use module-level variables with O(1) set-lookups and combine regular expressions where possible to skip Python iteration overhead.

## 2025-06-04 - YOLO Detection Result Tensor Access Optimization
**Learning:** In PyTorch/YOLO pipelines, accessing bounding box properties inside a loop (e.g., `box.xyxy[0]` or `box.conf[0]` for each box) causes repeated synchronous GPU-to-CPU data transfers, which severely degrades performance in high-frequency computer vision loops. Additionally, checking class membership against a list is slower than against a set.
**Action:** When extracting data from YOLO detection results, always extract all bounding box data at once using vectorized tensor operations like `results.boxes.data.cpu().numpy()`. Also, use Python sets instead of lists for class membership checks (e.g., `self.vehicle_classes = {2, 3, 5, 7}`) to ensure O(1) lookup times.
