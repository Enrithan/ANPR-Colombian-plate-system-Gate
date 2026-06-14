## 2024-05-24 - Speed up EasyOCR inference with allowlist
**Learning:** EasyOCR inference can be significantly accelerated and memory footprint reduced by providing an `allowlist` of expected characters when the domain is known (e.g., license plates only contain `A-Z` and `0-9`). This prunes the search space of the recognizer model.
**Action:** Always restrict OCR character sets using `allowlist` when the expected format is strictly defined.

## 2024-05-25 - Optimize Image Perspective Transformations
**Learning:** Applying geometric transformations like `cv2.warpPerspective` on a 3-channel (BGR) image takes roughly 3x longer than doing it on a single-channel (grayscale) image. Since the OCR pipeline converts to grayscale right after anyway, warping the BGR image was a pure waste of interpolation resources.
**Action:** Always convert to single-channel (grayscale) *before* applying computationally expensive spatial transformations (like warp, rotate, or scale) if the subsequent processing step only requires single-channel data.

## 2025-03-21 - Precomputing Data Structures & Regular Expressions
**Learning:** Initializing variables, casting datatypes (lists/dicts), parsing `string` constants and iterating inside `for` loops within very fast execution pathways (like plate validation per frame) has an enormous cumulative performance cost (0.22s to 0.04s or 4x difference). Pre-compiling one combined regex (`|`) is similarly 4x faster than looping over a list of independent compiled expressions.
**Action:** Move instantiation of objects, lists, sets, and constants out of tight loops. Use module-level variables with O(1) set-lookups and combine regular expressions where possible to skip Python iteration overhead.

## 2025-03-24 - PyTorch/Ultralytics GPU-CPU Sync Bottleneck in YOLO Loops
**Learning:** Extracting tensor values iteratively from `results.boxes` (e.g., `box.xyxy[0]`, `box.conf[0]`) forces repeated synchronous GPU-to-CPU data transfers per bounding box, causing massive CPU stalls in the detection loop. Vectorizing the extraction via `results.boxes.data.cpu().numpy()` pulls all data over the PCIe bus in a single batch, drastically reducing overhead.
**Action:** Never iterate over individual PyTorch tensor elements to extract scalar values. Always extract the entire batch of predictions as a NumPy array first, then iterate over the array in Python. Additionally, explicitly cast the numpy values (like `np.float32`) back to native Python types (`float()`, `int()`) when inserting into lists to avoid downstream serialization bugs (e.g., with `json.dumps()`).
