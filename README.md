# HeartRisk — Your future isn't one fixed number

An educational, privacy-first **10-year heart-disease risk what-if simulator**
built on a real, trained machine-learning model. The design reference is the
user-shared 5-panel light mockup (serif headline, stepped inputs, 12.4%-style
results panel, risk-over-age chart, what-if simulator, top factors) plus the
dark results card's structure translated into a light editorial palette —
paper/bone background, ink text, one restrained red accent, gridlines,
footnotes and figure captions, like quality data journalism.

One restrained animation only: a subtle ECG-style trace in the hero, with a
soft red glow, looping. It is static when `prefers-reduced-motion` is set,
appears nowhere else, and never touches the model.

> **Educational only — not a diagnosis, not medical advice.** The model is
> a research demonstration on historical cohort data. It is not clinically
> validated for personal use. For personal decisions, talk to a clinician.

## What it does
- Interactive health inputs (age, sex, smoking, cigarettes/day, BP medication,
  prevalent stroke/hypertension, diabetes, cholesterol, systolic/diastolic BP,
  BMI, heart rate, glucose) with training-range warnings.
- Live risk estimate + display bands (Low < 5%, Moderate 5–20%, High > 20% —
  display bands, **not** clinical thresholds), rendered with `aria-live`.
- **Figure 1 — risk over age:** your modelled risk as age varies, other inputs
  held constant, with a same-profile non-smoker curve for comparison.
- **Figure 2 — risk versus one factor:** cigarettes/day, systolic BP,
  cholesterol, BMI, glucose or heart rate. Click/tap the curve to set that
  factor (keyboard users use the sliders).
- **Top factors** driving your estimate, computed by replacing each factor
  with a low-risk reference value and re-running the model.
- **Ranked what-if scenarios** (quit smoking, halve cigarettes, lower systolic
  BP by 10/20, lower cholesterol by 20/40, lower BMI by 2/5), all recomputed
  through the model and ranked by modelled change, plus up to 3 pinned
  scenario comparisons. Diabetes, age, sex and history are fixed context —
  the simulator never offers to "switch diabetes off".
- Example profiles (labelled illustrative, not real people), metrics,
  calibration, methodology, limitations and provenance in the open.

## Model (honest numbers, held-out test set n = 848)
| Model | Accuracy | ROC-AUC | F1 @ 0.5 | Brier |
|---|---|---|---|---|
| Logistic regression — **shipped** | 0.8443 | 0.7021 | 0.0959 | 0.1213 |
| Random forest — compared | 0.8443 | 0.6458 | 0.0704 | 0.1264 |

CV ROC-AUC (5-fold, training): LR 0.732 ± 0.031, RF 0.719 ± 0.029.
The low F1 at 0.5 is real: with 15.2% prevalence the model rarely predicts
the positive class at that threshold (test confusion matrix [[709, 10],
[122, 7]]). That is why the product shows probabilities, not a yes/no label.
Full detail: `METRICS.md`, `docs/methodology.html`.

## Privacy
No account, no browser storage, no analytics, no tracking, no external
requests (no web fonts, no chart library, no CDN). The only fetch is
`model.json` from the same site. Inputs live only in page memory.

## Run it
```bash
python3 -m http.server --directory docs 8000   # then open http://localhost:8000
```
Opening `index.html` directly from disk will block the `model.json` fetch in
most browsers — serve the folder instead.

## Retrain + verify
```bash
python3 src/train.py    # retrains, rewrites docs/model.json + METRICS.md
node src/parity.js      # Python vs browser parity (must be < 1e-4; achieved ~1e-16)
```

## Structure
```
src/train.py          training, comparison, export, parity-case generation
src/parity.js         Node parity runner (browser code vs sklearn)
data/framingham.csv   public Framingham-style dataset (see DATA_PROVENANCE.md)
docs/index.html       simulator (GitHub Pages-ready, relative paths)
docs/app.js           UI, hand-built SVG charts, scenarios
docs/model-core.js    pure model formula (browser + Node parity)
docs/model.json       exported model + metrics + calibration
docs/methodology.html full methodology
docs/style.css        editorial stylesheet (system fonts only)
METRICS.md  PARITY.md  DATA_PROVENANCE.md
```

## Licence caveat for the dataset
The dataset mirror contains no licence file; see `DATA_PROVENANCE.md`
before republishing `data/framingham.csv`. The shipped `docs/model.json`
contains no individual-level data.
