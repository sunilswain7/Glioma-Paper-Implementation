# Viva Walkthrough Script — say it in your own words

A friendly, in-order flow. Open each file as you go, point at the part in **bold**,
and say roughly what's written. Keep it conversational — you're telling a story, not
reciting. Total: aim for ~8–10 min of walking through, then let her ask.

---

## 0. Opening (30 seconds — set the scene before opening any file)

> "So my project segments brain tumours from MRI scans. A normal model looks at one
> 2-D slice at a time and ignores that a tumour is actually a 3-D object spread across
> many slices. My contribution is a **Cross-Slice Transformer** — it looks at 5
> neighbouring slices together and uses attention to pull context from the neighbours
> into the middle slice, so the tumour boundary stays consistent in 3-D.
>
> The nice part is it's built so that if I set the number of slices to 1, it turns
> back into a plain U-Net — so my baseline and my proposed model are the *same code,
> one setting*. That gives me a really clean comparison.
>
> Let me walk you through the code — it flows in this order: config, then how I load
> the data, then the model, then the loss and metrics, then training."

Then open the `cst/` folder.

---

## 1. `config.py` (30 sec)

> "This is my control panel — every setting lives here in one place, so nothing is
> hard-coded in the model."

Point at:
- **`n_slices = 5`** → "This is the key knob. 5 is my model, 1 is the baseline."
- **`base_ch = 48`** and **`batch_size = 4`** → "These are sized so it fits my 4 GB
  laptop GPU — that constraint shaped a lot of my design choices."
- **`lr = 2e-4`, `seed = 42`** → "Learning rate and a fixed seed so results are
  reproducible."

---

## 2. `dataset.py` (2 min — she'll care about the split)

> "This turns the raw files into training examples. The data is one HDF5 file per
> slice — about 57,000 of them — each with the 4 MRI types and the doctor's tumour
> outline."

Point at, in order:

- **`build_volume_index`** → "First I group all the slice files by patient. One
  patient's full stack of slices is called a *volume* — I have 369 patients."

- **`split_volumes`** ⭐ → "This is important. I split into train and test **by whole
  patient**, never by slice. Because slices from the same patient look almost
  identical — if one went to training and the next to testing, the model would
  basically be tested on something it already saw. That's data leakage, and splitting
  whole patients prevents it."

- **`build_foreground_set`** → "Most slices are empty brain with no tumour, so I only
  train on *foreground* slices — the ones that actually contain tumour. I find them
  instantly from a metadata CSV of pixel counts instead of opening every file."

- **`__getitem__`** ⭐ → "This builds one training example. It's not a single slice —
  it's a stack of 5: the middle slice plus 2 above and 2 below. The model's job is:
  *given these 5, draw the tumour on the middle one.* The neighbours are just context."

- **`_load_slice`** → "And this handles the edges — if the middle slice is at the very
  top or bottom of the brain, the neighbours don't exist, so I just repeat the nearest
  slice instead of crashing."

- **`_augment`** → "For training I randomly flip the images left-right and top-bottom
  — flipping the tumour mask the same way — so from limited data I get more variety
  and the model learns the tumour shape, not its fixed position. Only on training,
  never on the test set."

---

## 3. `model.py` (3–4 min — THE main event, go slow here)

> "This is the network itself. It has three parts: an encoder, my cross-slice
> transformer, and a decoder."

- **`Encoder`** → "The encoder is the standard U-Net down-path. It takes each slice
  and progressively makes it smaller but deeper — 4 channels up to 384 — trading exact
  pixel positions for a rich understanding of what tissue is there. The smallest,
  deepest point is called the bottleneck."

- **`CrossSliceTransformer`** ⭐⭐ → *(this is your novelty — spend the most time here)*
  > "This is my contribution. At the bottleneck, for every location in the image, I
  > take the 5 slices' features and let them **look at each other** using attention.
  > So the middle slice can ask its neighbours 'what do you see at this exact spot?'
  > and blend in their answer. A normal transformer attends across *space* within one
  > image — mine attends across the *slice axis*. That's the novel bit. I also add a
  > small learnable position tag so it knows which slice is the middle and which are
  > neighbours. Then I keep only the enriched middle slice."

  If she wants detail, point at the `permute...reshape` line: "This just reorganizes
  the data so that at each of the 900 locations I get a little sequence of 5 slices for
  the attention to work on. No values change here — it's just arranging the data."

- **`Decoder`** → "The decoder is the U-Net up-path. It takes that enriched bottleneck
  and grows it back to full 240×240, using skip connections from the encoder to
  recover sharp edges, and outputs the final tumour map — 3 channels."

- **`CrossSliceUNet.forward`** ⭐ → "And this ties it together. The important line is
  `use_xslice = n_slices > 1` — if it's 1, the transformer is skipped entirely and
  it's just a plain U-Net. **That single `if` is my whole ablation** — same code, one
  flag switches between baseline and proposed."

---

## 4. `losses.py` (1 min)

> "This is what the model tries to minimize — how wrong its outline is. I combine two
> losses: **Dice**, which handles the fact that the tumour is a tiny part of the image,
> and **BCE**, which gives stable per-pixel gradients. I use sigmoid, not softmax,
> because the tumour regions can overlap — they're not mutually exclusive. My
> experiments showed the combination beats either one alone."

---

## 5. `metrics.py` (1 min)

> "This scores the model using the **Dice coefficient** — basically the overlap
> between my predicted tumour and the real one, from 0 to 1. I report it on the three
> standard BraTS regions: **Whole Tumour, Tumour Core, and Enhancing Tumour** — that's
> the official convention, so my numbers are comparable to published papers."

---

## 6. `train.py` (1–2 min)

> "This runs the training loop. For each batch it makes a prediction, computes the
> loss, and nudges the model to do better — that's one step. A full pass over all
> patients is one epoch; I ran 30 epochs. After each epoch it tests on the held-out
> patients and **saves the model only when it improves**, so I always keep the best
> version."

Point at:
- **the optimizer / scheduler `if` blocks** → "I made the optimizer, learning-rate
  schedule, and loss swappable from the command line — that's how I ran my comparison
  experiments."
- **`GradScaler` / `autocast`** → "This is mixed precision — it's what lets me train
  on just 4 GB of GPU memory."
- **the `save best.pth` line** → "Best model saved here; per-epoch scores go to
  history.json."

---

## 7. `sweep.py` (1 min — leads into results)

> "And this is how I justified my design choices instead of just guessing. It runs the
> whole training many times, changing **one setting at a time** — the number of slices,
> the optimizer, the learning rate, the loss, the scheduler — all on the same fixed
> setup. Then it collects every result into one comparison table. Let me show you what
> that told me."

---

## 8. Showing the results (the important finish — be honest here)

Open `glioma_seg/runs/sweep_results.csv` (or just talk from these numbers).

**First — the design-justification sweep** (quick runs, 16 patients, to compare
choices fairly):

> "Each of these changed one thing. My chosen settings won or tied on almost
> everything:
> - **Number of slices** — this is the key one: 1 slice gave 0.689, 3 slices 0.705,
>   5 slices 0.716. Clear improvement as I add more slice context — that supports my
>   whole idea.
> - **Optimizer** — AdamW was best, and when I tuned SGD fairly it tied, so AdamW is a
>   solid default.
> - **Loss** — Dice+BCE beat either alone.
> - **Learning rate** — 2e-4 was the sweet spot; higher and lower were both worse.
> - **Scheduler** — cosine and constant were basically tied at this short budget, so I
>   don't overclaim cosine; I use it because it helps over longer training."

**Then — the full-length runs** (all 369 patients, 30 epochs, the real headline):

> "Then I ran the two main models fully on all 369 patients:
> - Baseline (1 slice): **0.795** mean Dice
> - My model (5 slices): **0.790** mean Dice
>
> So on the full dataset they're basically tied. And here's the honest and interesting
> part I want to highlight: the cross-slice advantage was **clear when data was
> limited** — with 16 patients my model clearly beat the baseline — but with all 369
> patients the plain U-Net had enough data to catch up. So my finding is that the
> **cross-slice context matters most when labelled data is scarce**, which is exactly
> the real-world situation in many hospitals where annotated scans are rare. I'm
> presenting both results rather than cherry-picking the favourable one."

*(This honesty is a strength — examiners respect it far more than an inflated claim.)*

---

## 9. If she asks something you're unsure about

- Buy a second: "Good question — let me point to where that is in the code," then open
  the file. Talking through the code often gets you to the answer.
- If you genuinely don't know: "I'm not 100% sure, but my understanding is…" — an
  honest attempt beats a confident wrong answer.
- Fall-backs you can always give:
  - **Novelty?** → `model.py` CrossSliceTransformer — attention across slices.
  - **Data leakage?** → `dataset.py` split_volumes — split by whole patient.
  - **4 GB GPU?** → 2.5-D not 3-D, mixed precision, batch size 4.
  - **Why U-Net?** → medical-segmentation standard, sharp edges from skips, fits small
    data and small GPU, and its bottleneck is where my novelty plugs in.

---

## 10. Closing line

> "So in short: it's a U-Net with a cross-slice transformer at the bottleneck that
> brings in 3-D context, built so one flag switches between baseline and my model. I
> validated every design choice with controlled experiments, and my honest finding is
> that the cross-slice idea helps most when labelled data is limited."

**You've got this. Breathe, go slow, point at the code as you talk. Good luck! 🍀**
