# Complete Explanation of Every Term in Slides 5, 6, 7, 8

Read this before the presentation. Every term that appears on the slides is explained below in the order it appears.

---

# SLIDE 5: Proposed Methodology in the Paper

---

## "Three-phase framework"

The paper organizes its work into 3 phases:
- **Phase 1 (Build):** Build and train the neural network
- **Phase 2 (Test):** Evaluate it on unseen data from other hospitals
- **Phase 3 (Explain):** Use biology tools to understand WHY the model makes its predictions

Think of it like: Build the car → Test drive it → Open the hood to understand how the engine works.

---

## "4-channel MRI Input (T1, T1ce, T2, FLAIR)"

An MRI machine can produce different types of images of the same brain by changing its settings. Each type highlights different tissues:

**T1-weighted (T1):**
- Shows brain anatomy clearly
- Gray matter appears dark, white matter appears bright
- Good for seeing brain structure
- Think of it as a "structural photograph" of the brain

**T1 contrast-enhanced (T1ce):**
- Same as T1 but the patient is injected with a contrast agent (gadolinium) before scanning
- The contrast agent leaks into areas where the blood-brain barrier is broken (i.e., the tumor)
- Tumor boundaries light up brightly
- Think of it as "highlighting the tumor borders with a marker"

**T2-weighted (T2):**
- Water and fluid appear bright
- Edema (swelling) around the tumor shows up clearly
- Think of it as "showing where the brain is swollen"

**T2-FLAIR (Fluid Attenuated Inversion Recovery):**
- Like T2 but the cerebrospinal fluid (CSF) signal is suppressed (made dark)
- This means edema is still bright but the normal fluid in brain ventricles is dark
- Better for seeing lesions near fluid-filled spaces
- Think of it as "T2 but with background fluid removed for clarity"

**Why 4 channels?** Each modality shows different tissue properties. A tumor that is invisible on T1 might be obvious on FLAIR. By giving the model all 4, it gets the most complete picture. It's like looking at a crime scene with regular light, UV light, infrared, and X-ray — each reveals different evidence.

**In our code:** Each H5 file contains `image` with shape (240, 240, 4) — the 4th dimension is these 4 modalities stacked as channels, similar to how a color image has 3 channels (R, G, B).

---

## "mmFormer Tumor Segmentation"

**What is segmentation?** Identifying exactly which pixels in the MRI belong to the tumor vs normal brain.

**mmFormer** is a specific AI model (Transformer-based) that automatically draws the tumor boundary in 3D MRI. The paper uses it as a preprocessing step — before feeding MRI to their classification model, they first crop out just the tumor region.

**What is a Transformer?** A type of neural network architecture (originally from NLP/language models like GPT) adapted for images. It uses "attention" to focus on relevant parts of the input.

**Why we don't use it:** mmFormer requires 3D volumes and significant GPU memory. Our BraTS dataset already comes with expert-drawn segmentation masks (actually better than mmFormer's automated ones), so we don't need it.

---

## "3D ResNet-10 Shared Backbone"

Let's break every word:

**3D:** The convolutions operate on 3D volumes (height x width x depth), not flat 2D images. A 3D convolution filter slides through the volume in all three spatial directions.

**ResNet:** "Residual Network" — a specific neural network architecture invented by Microsoft Research in 2015 (He et al., won ImageNet competition). Its key innovation is **skip connections** (also called residual connections).

**What is a skip connection?** In a normal network, data flows: Input → Layer1 → Layer2 → Output. In ResNet, it flows: Input → Layer1 → Layer2 → (Output + Input). The "+Input" part is the skip connection. It adds the original input back to the output. This solves the "vanishing gradient problem" — in very deep networks, gradients become too small to train early layers. Skip connections provide a shortcut for gradients to flow backward.

**ResNet-10:** The "10" means the network has 10 layers with learnable weights. There are also ResNet-18, ResNet-34, ResNet-50, etc. Deeper = more capacity but more memory/compute needed. The paper uses the small ResNet-10 because 3D convolutions already consume a lot of memory.

**Shared Backbone:** "Backbone" means the main feature extraction part of the network. "Shared" means all four tasks (IDH, 1p/19q, Grade, Survival) use the SAME backbone. The features it extracts are shared across all tasks. Only the final prediction layers (heads) are task-specific.

**Why shared?** Because IDH, 1p/19q, grade, and survival are all related to the same tumor appearance. A shared backbone learns general tumor features that are useful for all tasks, making it more efficient and better at learning than having 4 separate models.

**In our code:** We use `ResNet18` from `torchvision.models` — a 2D version with 18 layers, pretrained on ImageNet. We adapted the first layer from 3-channel (RGB) to 4-channel (MRI modalities).

---

## "512-dimensional shared features"

After the image passes through the ResNet backbone and global average pooling, it becomes a single vector of 512 numbers. This vector is a compressed representation of the entire image — it encodes tumor shape, texture, intensity patterns, etc. in 512 numbers.

**Global Average Pooling:** The last convolutional layer outputs a feature map of shape (512, 4, 4) — 512 channels, each 4x4 spatial. Global average pooling takes the average of each 4x4 map, collapsing it to (512,) — one number per channel. This removes spatial dimensions and gives a fixed-size vector regardless of input image size.

**Why 512?** It's the default output of ResNet-18's last layer. ResNet-10 also produces 512 features. This number was designed by the ResNet authors and works well empirically.

---

## "Task heads: IDH head, 1p/19q head, Grade head (512→2)"

Each head is a single **fully connected (FC) layer** that takes the 512 shared features and outputs 2 numbers (one for each class: positive/negative).

**Fully Connected (Linear) Layer:** Every input neuron is connected to every output neuron. Mathematically: output = input × weight_matrix + bias. For 512→2: weight matrix is 512×2 = 1,024 parameters + 2 bias = 1,026 parameters per head.

**Why output 2 instead of 1?** Convention in PyTorch. For binary classification, outputting 2 values (one per class) and applying softmax is equivalent to outputting 1 value and applying sigmoid. The 2-output approach works with CrossEntropyLoss directly.

---

## "Gradient Detachment (1p/19q)"

This is the most likely question an evaluator will ask. Here's the full explanation:

**What are gradients?** During training, after the model makes a prediction, we compute how wrong it was (loss). Then we calculate gradients — they tell each weight "how much and in which direction should you change to reduce the loss?" This is called backpropagation.

**What is gradient detachment?** When we write `feat.detach()`, we tell PyTorch: "Use these features for the forward pass (making predictions), but do NOT send any gradients backward through them during backpropagation." The 1p/19q head can still make predictions, but its mistakes don't affect the shared backbone.

**Why detach 1p/19q specifically?**

Biological reason: 1p/19q codeletion is strongly linked to IDH mutation. Almost ALL codeleted tumors are also IDH-mutant. If both IDH and 1p/19q gradients flow through the backbone, they would push the backbone to learn very similar features, creating redundancy and potentially conflicting signals.

Technical reason: The IDH task already teaches the backbone to distinguish IDH-mutant from wildtype. The 1p/19q task needs to make a FINER distinction within IDH-mutant tumors (which of the IDH-mutant are codeleted?). If 1p/19q gradients modified the backbone, they might undo what IDH gradients learned.

**In our code:** `out_1p19q = self.fc_1p19q(feat.detach())` — the `.detach()` call creates a copy of `feat` that is disconnected from the computation graph. The 1p/19q head's weights still update (it learns), but it can't change the backbone.

---

## "Cox Partial Likelihood Loss"

This is the survival prediction loss function. It's from a statistical method called the **Cox Proportional Hazards Model** (invented by Sir David Cox in 1972, one of the most cited statistical papers ever).

**What problem does it solve?** We want the model to learn: "patients who die sooner should get higher risk scores." But some patients are still alive when the study ends (censored) — we don't know when they'll die. Cox handles this elegantly.

**How it works (intuitive):**
1. Line up all patients by survival time
2. For each patient who died: look at everyone who was still alive at that time
3. The model should give the patient who died the HIGHEST risk score among all those still alive
4. If it does, the loss is low. If someone alive has a higher risk score, the loss is high

**Mathematical formula (paper's Equation 5):**
```
loss = -1/N * sum over patients who died of [
    risk_score_i - log(sum of exp(risk_score_j) for all j alive at time_i)
]
```

**Why "partial"?** It only uses the ORDER of events (who dies before whom), not the exact time. This makes it non-parametric — it doesn't assume any specific distribution (like exponential or Weibull) for survival times.

**In our code:** The `NegativeLogLikelihood` class implements this. It sorts patients by time, computes cumulative sums of risk, and averages the loss over patients with events.

---

## "Deep Prognosis Score (DPS)"

The paper's main novelty. DPS is a single number that summarizes a patient's prognosis (how bad their outlook is).

**How it's built:**
```
Step 1: Take the 512 shared features
Step 2: Pass through MLP: 512 → 64 → 16 (with ReLU activation)
        This gives 16 "prognosis features"
Step 3: Get softmax scores from IDH head (2 values), 
        1p/19q head (2 values), Grade head (2 values) = 6 values total
        These are DETACHED (no gradient flow from DPS back to the heads)
Step 4: Concatenate: [16 prognosis features] + [6 task scores] = 22 values
Step 5: Pass through: 22 → 32 → 1 = single DPS value
```

**What does DPS mean?**
- Higher DPS = poorer prognosis (shorter expected survival)
- Lower DPS = better prognosis (longer expected survival)

**Why concatenate task scores?** The molecular status directly affects survival. IDH-mutant patients live longer. Codeleted patients live longer. Low-grade patients live longer. By feeding these predictions into the survival branch, the DPS incorporates molecular information into the prognosis.

**Why are task scores detached?** If they weren't detached, the survival loss would try to change the classification heads to produce scores that minimize survival loss, potentially making them worse at classification. Detachment ensures each task head is only trained by its own loss.

---

## "Softmax"

Softmax converts raw model outputs (logits) into probabilities that sum to 1.

**Example:** If IDH head outputs [2.1, 0.5]:
```
softmax([2.1, 0.5]) = [exp(2.1)/(exp(2.1)+exp(0.5)), exp(0.5)/(exp(2.1)+exp(0.5))]
                     = [8.17/9.82, 1.65/9.82]
                     = [0.83, 0.17]
```
Meaning: 83% probability of class 0 (wildtype), 17% probability of class 1 (mutant).

---

## "ROC curves"

**ROC = Receiver Operating Characteristic.** A plot of True Positive Rate (sensitivity) vs False Positive Rate (1 - specificity) at different classification thresholds.

**How to read it:** A perfect model's ROC curve goes straight up to (0,1) then right to (1,1). A random model's curve is a diagonal line from (0,0) to (1,1).

**AUC (Area Under the Curve):** The area under the ROC curve. Ranges from 0.5 (random) to 1.0 (perfect). It answers: "If I pick one random positive and one random negative patient, what's the probability the model ranks the positive patient higher?"

---

## "Confusion Matrix"

A 2x2 table showing actual vs predicted labels:
```
                Predicted
              Neg    Pos
Actual  Neg   TN     FP
        Pos   FN     TP
```
- **TP (True Positive):** Patient has the mutation AND model says yes ✓
- **TN (True Negative):** Patient doesn't have it AND model says no ✓
- **FP (False Positive):** Patient doesn't have it BUT model says yes ✗ (false alarm)
- **FN (False Negative):** Patient has it BUT model says no ✗ (missed case)

---

## "Calibration curves"

Shows whether the model's predicted probabilities match reality.

**Example:** If the model predicts "70% probability of IDH mutation" for 100 patients, calibration asks: "Did approximately 70 of those 100 actually have IDH mutation?"

A perfectly calibrated model's calibration curve is a diagonal line (predicted probability = observed frequency).

**Why it matters in medicine:** A doctor needs to trust that when the model says "80% chance of high-grade," it really means 80%, not 50%.

---

## "Kaplan-Meier survival curves"

A statistical method to estimate survival probability over time, even when some patients are still alive (censored).

**How to read it:**
- X-axis: time (months)
- Y-axis: probability of still being alive
- The curve starts at 1.0 (100% alive) and drops as patients die
- Each step down represents a death event
- Censored patients (still alive at end of study) are marked with tick marks

**In our implementation:** We split patients into two groups by median DPS (low-DPS vs high-DPS) and plot separate curves. If the model works, the high-DPS group's curve should drop faster (they die sooner).

---

## "Log-rank test"

A statistical test that asks: "Are the survival curves of two groups significantly different, or could the difference be due to random chance?"

- Produces a **p-value**
- p < 0.05 means "the difference is statistically significant" (less than 5% chance it's random)
- p < 0.001 means "highly significant"
- If our low-DPS and high-DPS groups have p < 0.05, it proves the DPS meaningfully separates patients

---

## "Radio-multiomics analysis"

"Radio" = radiological (MRI imaging) features. "Multi-omics" = multiple types of molecular biology data:

- **Genomics:** DNA mutations and copy number variations
- **Transcriptomics:** RNA expression levels (which genes are active)
- **Proteomics:** Protein expression levels
- **Epigenomics:** DNA methylation patterns (chemical modifications that control gene activity)

The paper analyzes these to show that the DPS correlates with known biological mechanisms (cancer pathways, immune response, etc.). We don't do this because it requires matched patient data from TCGA that BraTS doesn't include.

---

## "GSEA, ESTIMATE, ssGSEA"

**GSEA (Gene Set Enrichment Analysis):** Tests whether a predefined set of genes (like "genes involved in cell growth") shows statistically significant differences between two groups (high-DPS vs low-DPS). It answers: "Are cancer-related genes more active in the high-DPS group?"

**ESTIMATE:** A computational method that estimates the fraction of immune cells and stromal (support) cells in a tumor sample from gene expression data. It produces an "immune score" and "stromal score."

**ssGSEA (single-sample GSEA):** A variant of GSEA that assigns enrichment scores to individual patients rather than comparing groups. Used to estimate levels of specific immune cell types in each tumor.

---

# SLIDE 6: Tools, Technologies, and Dataset

---

## "PyTorch"

An open-source deep learning framework by Meta (Facebook) AI Research. It provides:
- **Tensors:** Multi-dimensional arrays that can run on GPU (like NumPy but GPU-accelerated)
- **Autograd:** Automatic differentiation — computes gradients automatically for backpropagation
- **nn.Module:** Building blocks for neural networks (layers, loss functions, etc.)
- **DataLoader:** Efficiently loads batches of data with parallel workers

**Why PyTorch over TensorFlow?** The paper uses PyTorch. Also, PyTorch has more intuitive syntax, better debugging, and is the standard in medical imaging research.

---

## "torchvision"

A companion library to PyTorch that provides:
- **Pretrained models:** ResNet18, ResNet50, VGG, etc. with weights trained on ImageNet
- **Transforms:** Image preprocessing (resize, normalize, augment)
- **Datasets:** Common datasets (ImageNet, CIFAR, etc.)

We use it specifically for `models.resnet18(weights=ResNet18_Weights.DEFAULT)` — loading a ResNet18 that was trained on 1.2 million ImageNet images.

---

## "h5py"

A Python library for reading HDF5 files (.h5). HDF5 (Hierarchical Data Format version 5) is a file format designed for storing large scientific datasets efficiently. Each .h5 file in our dataset stores:
- `image`: the 4-channel MRI slice (240x240x4 float array)
- `mask`: the tumor segmentation mask (240x240x3 binary array)

---

## "scikit-learn"

Python's standard machine learning library. We use it for:
- `roc_auc_score`: Computing AUC
- `roc_curve`: Getting the ROC curve points
- `confusion_matrix`: Building the confusion matrix
- `calibration_curve`: Computing calibration data
- `accuracy_score`, `precision_score`, `recall_score`, `f1_score`: Classification metrics

---

## "lifelines"

A Python library for survival analysis. We use:
- `KaplanMeierFitter`: Fits and plots Kaplan-Meier survival curves
- `logrank_test`: Tests if two survival curves are statistically different

---

## "torch.cuda.amp" (Mixed Precision Training)

**What:** Normally, neural network weights are stored as 32-bit floating point numbers (FP32). Mixed precision uses 16-bit (FP16) for most operations and 32-bit only where necessary.

**Why:** FP16 numbers take half the memory of FP32. This lets us fit larger models/batches on a small GPU. An RTX 3050 with 4GB effectively gets ~6-7GB worth of capacity.

**How:** PyTorch's `autocast()` automatically converts operations to FP16 where safe. `GradScaler` scales gradients to prevent them from becoming too small in FP16 (underflow).

**Does it hurt accuracy?** No — research shows mixed precision training achieves virtually identical results to full precision while being 1.5-2x faster and using ~40% less memory.

---

## "CUDA"

NVIDIA's platform for running computations on their GPUs. When we write `tensor.to('cuda')`, we move data from CPU RAM to GPU memory. GPU has thousands of small cores that can do many operations in parallel — perfect for matrix multiplications in neural networks.

---

## "HGG and LGG"

**HGG = High-Grade Glioma:** WHO Grade III (anaplastic glioma) and Grade IV (glioblastoma/GBM). These are aggressive, fast-growing tumors. Median survival for GBM is ~15 months. MRI shows: ring enhancement, necrosis (dead tissue in center), extensive edema, mass effect.

**LGG = Low-Grade Glioma:** WHO Grade II (diffuse glioma). Slower-growing, better prognosis. Median survival 5-10+ years. MRI shows: minimal or no enhancement, no necrosis, less edema, more diffuse borders.

---

## "NIfTI format"

**NIfTI = Neuroimaging Informatics Technology Initiative.** A file format (.nii or .nii.gz) for storing 3D/4D brain imaging data. It contains:
- A header with metadata (image dimensions, voxel size, orientation)
- The raw 3D volume data as a matrix of numbers

The original BraTS data is in NIfTI. Our Kaggle version was pre-converted to HDF5 slices.

---

## "HDF5 format"

**HDF5 = Hierarchical Data Format 5.** A file format for storing large arrays of numbers efficiently. Each .h5 file can contain multiple named datasets (like a dictionary of arrays). Our files have `image` and `mask` datasets inside each .h5 file.

---

## Dataset cohorts: "TC, IVC, EVC1, EVC2, EVC3"

**TC = Training Cohort:** 1,352 patients from 2 private Chinese hospitals (FAHZU + HNPH). Used to train the model.

**IVC = Internal Validation Cohort:** 579 patients from the same 2 hospitals. Used to tune hyperparameters and check for overfitting. "Internal" because it comes from the same institutions as training data.

**EVC1 = External Validation Cohort 1:** 213 patients from TCGA (The Cancer Genome Atlas). "External" because it comes from DIFFERENT institutions than training data. This tests if the model generalizes to new hospitals with different MRI scanners.

**EVC2:** 410 patients from UCSF (University of California San Francisco).
**EVC3:** 222 patients from EGD (Erasmus Glioma Database, Netherlands).

---

# SLIDE 7: Implementation Details

---

## "Metadata loading: name_mapping.csv, survival_info.csv, meta_data.csv"

Three CSV files in the dataset:

**name_mapping.csv:** Maps patient IDs across BraTS versions and stores the grade label.
```
Grade, BraTS_2020_subject_ID, ...
HGG,   BraTS20_Training_001, ...
LGG,   BraTS20_Training_256, ...
```

**survival_info.csv:** Contains real survival data for 236 patients.
```
Brats20ID,             Age,    Survival_days,  Extent_of_Resection
BraTS20_Training_001,  60.463, 289,           GTR
```
- **Survival_days:** How many days the patient survived after diagnosis
- **Extent_of_Resection:** GTR (Gross Total Resection = all visible tumor removed), STR (Subtotal = partial removal), NA (not available)

**meta_data.csv:** Information about each 2D slice — which volume it belongs to, which slice number, and whether it contains tumor (target=0 or 1).

---

## "Z-score normalization (paper's Equation 1)"

Formula: **X' = (X - mean) / std**

For each MRI channel separately:
1. Calculate the mean pixel value of that channel
2. Calculate the standard deviation
3. Subtract mean, divide by std

**Why?** Different MRI scanners produce different intensity ranges. One scanner's T1 might range from 0-1000, another from 0-5000. Z-score normalization makes them all have mean=0 and std=1, so the model sees consistent inputs regardless of the scanner.

**In our code:**
```python
for c in range(4):
    ch = img_tensor[c]
    mu, sigma = ch.mean(), ch.std()
    if sigma > 0:
        img_tensor[c] = (ch - mu) / sigma
```

---

## "Data augmentation: random horizontal and vertical flips"

During training, we randomly flip images horizontally (left-right) or vertically (top-bottom) with 50% probability.

**Why?** A tumor on the left side of the brain is biologically the same as one on the right side. By flipping, we teach the model that location doesn't matter — only the tumor's appearance matters. This effectively doubles our dataset size and prevents the model from memorizing specific positions.

**What the paper uses additionally:** Rotation, scale transformation, and elastic deformation (warping the image like rubber). We use simpler augmentation because 2D flips are computationally cheap and effective.

---

## "ResNet18 pretrained on ImageNet, first conv adapted from 3-channel RGB to 4-channel MRI"

**ImageNet:** A dataset of 1.2 million natural images (dogs, cars, buildings, etc.) with 1000 categories. Training a model on ImageNet teaches it to detect general visual features: edges, textures, shapes, patterns.

**Transfer Learning:** Using a model trained on one task (ImageNet) as the starting point for another task (MRI classification). Even though MRI images look nothing like dog photos, the low-level features (edges, textures) are universal.

**The adaptation problem:** ImageNet images have 3 channels (R, G, B). Our MRI has 4 channels (T1, T1ce, T2, FLAIR). The first convolutional layer expects 3-channel input.

**Our solution:** 
```python
# Original: Conv2d(3, 64, kernel_size=7) — expects 3 channels
# We need: Conv2d(4, 64, kernel_size=7) — for 4 channels
# Solution: Average the 3 RGB weight channels → 1 channel weight, then copy it 4 times
mean_weight = orig_conv.weight.mean(dim=1, keepdim=True)  # (64,3,7,7) → (64,1,7,7)
new_conv.weight = mean_weight.repeat(1, 4, 1, 1)           # (64,1,7,7) → (64,4,7,7)
```

---

## "Linear(512, 2) — binary classification"

`Linear(512, 2)` means a fully connected layer with:
- 512 input neurons (the shared features)
- 2 output neurons (one per class)

After softmax, the 2 outputs become probabilities: [P(negative), P(positive)]. For IDH: [P(wildtype), P(mutant)].

---

## "Survival branch: 512→64→16, concatenated with 6 task scores, →32→1 (DPS)"

The survival branch is a small neural network (MLP = Multi-Layer Perceptron):

```
512 features ──→ Linear(512,64) ──→ ReLU ──→ Dropout(0.3)
                                              ──→ Linear(64,16) ──→ ReLU
                                                                    = 16 prognosis features

IDH softmax output: [0.3, 0.7]     = 2 values
1p/19q softmax output: [0.9, 0.1]  = 2 values  
Grade softmax output: [0.2, 0.8]   = 2 values
                                    Total = 6 values

Concatenate: [16 prognosis features, 6 task scores] = 22 values
──→ Linear(22, 32) ──→ ReLU ──→ Dropout(0.3) ──→ Linear(32, 1) = DPS
```

**ReLU (Rectified Linear Unit):** An activation function: ReLU(x) = max(0, x). It introduces non-linearity — without it, stacking linear layers would just be one big linear layer and couldn't learn complex patterns.

**Dropout(0.3):** During training, randomly sets 30% of neurons to zero. This prevents overfitting by forcing the network to not rely on any single neuron. During evaluation, dropout is turned off and all neurons are used.

---

## "CrossEntropyLoss"

The standard loss function for classification. It measures how far the model's predicted probabilities are from the true labels.

**Formula:** L = -[y * log(p) + (1-y) * log(1-p)]

Where y is the true label (0 or 1) and p is the predicted probability.

**Example:** If true label = 1 (mutant) and model predicts p=0.9:
L = -[1 * log(0.9)] = -(-0.105) = 0.105 (low loss, good prediction)

If model predicts p=0.1:
L = -[1 * log(0.1)] = -(-2.303) = 2.303 (high loss, bad prediction)

---

## "Weighted CrossEntropyLoss [1.0, 4.0] for 1p/19q"

Same as CrossEntropyLoss but the loss for each class is multiplied by a weight.

**Why?** Only 9.2% of patients are 1p/19q codeleted. Without weighting, the model can get 90.8% accuracy by always predicting "intact" — it never learns to detect codeletion. The weight [1.0, 4.0] means:
- Mistakes on "intact" class cost 1.0
- Mistakes on "codeleted" class cost 4.0

This forces the model to pay 4x more attention to getting codeleted cases right.

---

## "Adam optimizer (lr=0.001, weight_decay=1e-4)"

**Adam (Adaptive Moment Estimation):** An optimizer that updates model weights during training. Unlike basic SGD which uses the same learning rate for all parameters, Adam adapts the learning rate for each parameter individually based on the history of its gradients.

**lr=0.001:** The base learning rate — how big each weight update step is. Too high = unstable, overshoot. Too low = too slow, stuck.

**weight_decay=1e-4:** L2 regularization — slightly shrinks all weights toward zero each step. This prevents weights from growing too large (overfitting). 1e-4 = 0.0001, a mild regularization.

---

## "Polynomial LR decay: lr * (1 - epoch/max_epoch)^0.9"

The learning rate decreases over training:
```
Epoch  1: 0.001 * (1 - 1/35)^0.9  = 0.000974
Epoch 10: 0.001 * (1 - 10/35)^0.9 = 0.000739
Epoch 20: 0.001 * (1 - 20/35)^0.9 = 0.000466
Epoch 30: 0.001 * (1 - 30/35)^0.9 = 0.000174
Epoch 35: 0.001 * (1 - 35/35)^0.9 = 0.000000
```

**Why decay?** Early in training, big steps help explore. Later, small steps help fine-tune. By epoch 35, LR is zero — training naturally stops. The exponent 0.9 makes the decay gradual (not too aggressive).

---

## "Youden's J statistic"

A method to find the best classification threshold. 

**Problem:** The model outputs probabilities (0.0 to 1.0). We need a cutoff: above = positive, below = negative. But what cutoff?

**Youden's J = Sensitivity + Specificity - 1 = TPR - FPR**

We try every possible threshold and pick the one that maximizes J. This finds the threshold that simultaneously maximizes true positives and minimizes false positives.

**Example:** At threshold 0.5, sensitivity=0.9 and specificity=0.6 → J=0.5. At threshold 0.3, sensitivity=0.95 and specificity=0.7 → J=0.65. We'd pick 0.3 as the optimal threshold.

---

# SLIDE 8: Experimental Setup

---

## "Patient-level split (no data leakage)"

**Data leakage:** When information from the test set accidentally enters the training set, making results look artificially good.

**Our risk:** Each patient has 8 slices. If we split at the SLICE level, some slices from the same patient could end up in both train and test. The model would memorize patient-specific features during training and "cheat" during testing.

**Our solution:** Split at the PATIENT level. All 8 slices from a patient go entirely to train OR entirely to test. No patient crosses the boundary. This gives honest, unbiased results.

---

## "Seed = 42 (fixed for reproducibility)"

Neural network training involves randomness (weight initialization, data shuffling, dropout). By setting a fixed seed, we ensure the random number generator produces the same sequence every time.

**Why 42?** Convention (reference to "The Hitchhiker's Guide to the Galaxy"). Any fixed number works. The point is that running the code twice gives identical results.

**In code:**
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

---

## "VRAM (Video RAM)"

The memory on the GPU. Different from system RAM. The model weights, input images, gradients, and optimizer states all must fit in VRAM during training.

**Our breakdown (approximate):**
- Model weights: ~45 MB (11.2M parameters * 4 bytes each, or 2 bytes in FP16)
- Batch of images: 16 * 4 * 128 * 128 * 4 bytes = ~4 MB
- Gradients + activations: ~2-3 GB (the biggest consumer)
- Total: ~3 GB out of 4 GB available

---

## "FP16 vs FP32"

**FP32 (float32):** Standard 32-bit floating point. Can represent ~7 decimal digits of precision. Takes 4 bytes per number.

**FP16 (float16):** Half-precision 16-bit. ~3 decimal digits of precision. Takes 2 bytes per number.

**For neural networks:** Most operations don't need 7 digits of precision. Using FP16 halves memory usage and doubles throughput on modern GPUs (which have dedicated FP16 hardware called Tensor Cores).

---

## "TP, TN, FP, FN"

For a specific example — IDH mutation prediction:

- **TP (True Positive):** Patient IS IDH-mutant, model says "mutant" → Correct ✓
- **TN (True Negative):** Patient is NOT mutant, model says "wildtype" → Correct ✓
- **FP (False Positive):** Patient is NOT mutant, model says "mutant" → Wrong ✗ (unnecessary worry)
- **FN (False Negative):** Patient IS mutant, model says "wildtype" → Wrong ✗ (missed diagnosis, dangerous)

In medicine, FN is usually worse than FP — missing a mutation is more dangerous than a false alarm.

---

## "Precision vs Recall — When the evaluator asks what's the difference"

**Precision** answers: "Of all the patients the model flagged as positive, what fraction actually are positive?"
- High precision = few false alarms
- Formula: TP / (TP + FP)

**Recall (Sensitivity)** answers: "Of all the patients who actually are positive, what fraction did the model catch?"
- High recall = few missed cases
- Formula: TP / (TP + FN)

**Trade-off:** You can get 100% recall by predicting everyone as positive (but precision drops to the prevalence rate). You can get 100% precision by only predicting the most obvious cases as positive (but recall drops because you miss subtle cases).

**F1-Score** is the harmonic mean of precision and recall — it balances both:
- F1 = 2 * (Precision * Recall) / (Precision + Recall)
- High F1 means both precision and recall are high

---

## "AUC — why it's the primary metric"

**Why not accuracy?** If 90% of patients are IDH-wildtype, a model that ALWAYS predicts "wildtype" gets 90% accuracy but is completely useless. Accuracy is misleading for imbalanced datasets.

**AUC is threshold-independent.** It doesn't depend on choosing a cutoff — it evaluates the model's ability to RANK patients correctly (positive patients should have higher predicted probabilities than negative patients). This makes it fair even with imbalanced classes.

**AUC = 0.5** means the model is no better than random coin flipping.
**AUC = 1.0** means perfect separation — every positive patient has a higher score than every negative patient.
**AUC = 0.8** means "if you pick one random positive and one random negative patient, there's an 80% chance the model correctly ranks the positive one higher."
