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
>
> **Not enough evidence for clinical decisions.** There is not yet enough
> evidence to trust these individual risk estimates for clinical decisions.
> That requires stronger calibration evidence and evaluation on an
> independent, relevant population — neither of which this project has.
> See EVALUATION.md.

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

## Model (honest numbers — current 60/20/20 evaluation, test n = 848)
Shipped: regularized logistic regression (C=0.01, class_weight='balanced',
tuned by 5-fold CV on train only) + sigmoid (Platt) calibration fitted on
the calibration split only (a=−1.6183, b=0.8737). This supersedes the old
80/20 numbers; do not quote them alongside these.

| Model (sigmoid-calibrated, test) | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 @ 0.5 | Brier | Cal. intercept / slope |
|---|---|---|---|---|---|---|---|---|
| Logistic regression, regularized — **shipped** | 0.8514 | 0.7041 | 0.3021 | 0.6364 | 0.0543 | 0.1000 | 0.1205 | −0.1093 / 0.9416 |
| Random forest — compared | 0.8467 | 0.6777 | 0.2808 | 0.4000 | 0.0155 | 0.0299 | 0.1221 | −0.1601 / 0.9167 |
| HistGradientBoosting — compared | 0.8491 | 0.6809 | 0.2784 | 1.0000 | 0.0078 | 0.0154 | 0.1222 | 0.0072 / 1.0271 |
| LR, unregularized baseline | 0.8491 | 0.7047 | 0.3001 | 0.5714 | 0.0310 | 0.0588 | 0.1202 | −0.0650 / 0.9693 |

CV ROC-AUC (5-fold, training): tuned LR 0.7457 ± (see EVALUATION.md),
RF 0.7378, HistGB 0.7062, baseline 0.7420.
The low recall at 0.5 is real: with 15.2% prevalence the shipped model
catches only 5.4% of positives at that threshold (confusion [[715, 4],
[122, 7]]). That is why the product shows probabilities, not a yes/no label.
Raw tuned-LR probabilities were badly miscalibrated (Brier 0.207) and must
not be used; only the calibrated probabilities are shown. Full detail:
`EVALUATION.md`, `METRICS.md`, `docs/methodology.html`.

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
python3 src/train.py    # retrains, re-evaluates, rewrites docs/model.json + METRICS.md + EVALUATION.md
node src/parity.js      # Python vs browser parity, calibrated model (must be < 1e-4; achieved ~8e-17 on 43 cases)
```

## Structure
```
src/train.py          training, tuning, calibration, comparison, export, parity-case generation
src/evaluate.py       exploratory evaluation script (same design; train.py is the authoritative run)
src/parity.js         Node parity runner (browser code vs sklearn, calibrated)
data/framingham.csv   public Framingham-style dataset (see DATA_PROVENANCE.md)
docs/index.html       simulator (GitHub Pages-ready, relative paths)
docs/app.js           UI, hand-built SVG charts, scenarios
docs/model-core.js    pure model formula (browser + Node parity)
docs/model.json       exported model + metrics + calibration
docs/methodology.html full methodology
docs/style.css        editorial stylesheet (system fonts only)
METRICS.md  EVALUATION.md  PARITY.md  DATA_PROVENANCE.md
```

## Licence caveat for the dataset
The dataset mirror contains no licence file; see `DATA_PROVENANCE.md`
before republishing `data/framingham.csv`. The shipped `docs/model.json`
contains no individual-level data.
