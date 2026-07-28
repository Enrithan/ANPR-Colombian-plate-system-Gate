## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2025-03-21 - Precomputing Data Structures & Regular Expressions
**Learning:** Initializing variables, casting datatypes (lists/dicts), parsing `string` constants and iterating inside `for` loops within very fast execution pathways (like plate validation per frame) has an enormous cumulative performance cost (0.22s to 0.04s or 4x difference). Pre-compiling one combined regex (`|`) is similarly 4x faster than looping over a list of independent compiled expressions.
**Action:** Move instantiation of objects, lists, sets, and constants out of tight loops. Use module-level variables with O(1) set-lookups and combine regular expressions where possible to skip Python iteration overhead.

## 2024-05-26 - Vectorizing Bounding Box Extraction in PyTorch/YOLO Loops
**Learning:** Iterating through `results.boxes` elements sequentially and accessing attributes like `box.xyxy[0]` or `box.conf[0]` creates a synchronous bottleneck because it triggers multiple independent GPU-to-CPU memory transfers.
**Action:** Always extract bounding box data globally at once via `results.boxes.data.cpu().numpy()` and process the extracted native array to eliminate loop overhead and synchronous GPU waits.

## 2024-05-26 - Fast O(1) Class Index Lookups in Hot Paths
**Learning:** Searching an integer list `[2, 3, 5, 7]` via the `in` operator per frame scales at O(N) cost and wastes significant overhead inside inference loops.
**Action:** Use a python `set()` initialization (e.g. `{2, 3, 5, 7}`) for static membership checking to gain O(1) time complexity in real-time detection logic.

## 2024-05-27 - OpenCV Array Allocation in High-Frequency Inference Loops
**Learning:** During continuous frame processing in `cv2.getPerspectiveTransform` and `cv2.filter2D`, repeatedly allocating constant mathematical matrices via `np.array` or `np.zeros` inside the loop has a severe cumulative latency cost. Additionally, `np.empty` is marginally faster than `np.zeros` for dynamically sizing temporary buffers (like `rect`) because it skips the memory-zeroing operation when the array is immediately populated.
**Action:** Always hoist static configuration arrays (like projection `dst` points or morphological `kernels`) to module-level constants to ensure zero-cost allocation per frame. For arrays that must be allocated per frame but are immediately overwritten, prefer `np.empty` over `np.zeros` or `np.ones`.
