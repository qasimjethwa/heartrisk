/* HeartRisk browser app — all results computed live through docs/model.json.
   No storage, no tracking, no external requests (model.json is same-origin).
   No risk score or scenario improvement is hardcoded anywhere here. */
"use strict";
var MODEL = null, PINNED = [];
var SLIDERS = ["age", "cigs", "sys", "dia", "chol", "hr", "bmi", "glu"];
var PROFILES = {
  a: { male: 1, age: 52, smoking: "current", cigs: 20, bpmeds: 0, stroke: 0, hyp: 1, diabetes: 0, chol: 240, sys: 138, dia: 85, bmi: 27.5, hr: 75, glu: 85 },
  b: { male: 0, age: 48, smoking: "never", cigs: 0, bpmeds: 0, stroke: 0, hyp: 0, diabetes: 0, chol: 200, sys: 118, dia: 76, bmi: 24.0, hr: 70, glu: 80 },
  c: { male: 1, age: 61, smoking: "never", cigs: 0, bpmeds: 1, stroke: 0, hyp: 1, diabetes: 1, chol: 260, sys: 150, dia: 92, bmi: 30.0, hr: 80, glu: 110 }
};
var FACTOR_RANGES = { cigs: [0, 60, 2], sys: [80, 220, 5], chol: [100, 400, 10], bmi: [15, 50, 1], glu: [40, 300, 10], hr: [40, 140, 5] };
var FIELD_OF_FACTOR = { cigs: "cigsPerDay", sys: "sysBP", chol: "totChol", bmi: "BMI", glu: "glucose", hr: "heartRate" };

function $(id) { return document.getElementById(id); }
function fmtPct(p) { return (p * 100).toFixed(1) + "%"; }
function segValue(field) {
  var btn = document.querySelector('[data-field="' + field + '"][aria-pressed="true"]');
  return btn ? btn.getAttribute("data-value") : null;
}
function readInput() {
  var smoking = segValue("smoking");
  var current = smoking === "current" ? 1 : 0;
  return {
    male: Number(segValue("male")), age: Number($("age").value),
    currentSmoker: current, cigsPerDay: current ? Number($("cigs").value) : 0,
    BPMeds: $("bpmeds").checked ? 1 : 0, prevalentStroke: $("stroke").checked ? 1 : 0,
    prevalentHyp: $("hyp").checked ? 1 : 0, diabetes: $("diabetes").checked ? 1 : 0,
    totChol: Number($("chol").value), sysBP: Number($("sys").value),
    diaBP: Number($("dia").value), BMI: Number($("bmi").value),
    heartRate: Number($("hr").value), glucose: Number($("glu").value)
  };
}
function setSeg(field, value) {
  document.querySelectorAll('[data-field="' + field + '"]').forEach(function (b) {
    b.setAttribute("aria-pressed", String(b.getAttribute("data-value") === String(value)));
  });
}
function setControls(p) {
  $("age").value = p.age; $("cigs").value = p.cigs; $("sys").value = p.sys;
  $("dia").value = p.dia; $("chol").value = p.chol; $("hr").value = p.hr;
  $("bmi").value = p.bmi; $("glu").value = p.glu;
  setSeg("male", p.male); setSeg("smoking", p.smoking);
  $("bpmeds").checked = !!p.bpmeds; $("stroke").checked = !!p.stroke;
  $("hyp").checked = !!p.hyp; $("diabetes").checked = !!p.diabetes;
  SLIDERS.forEach(syncOutput);
}
function syncOutput(id) {
  var el = $(id), out = $(id + "-v"); if (!el || !out) return;
  out.textContent = id === "bmi" ? Number(el.value).toFixed(1) : el.value;
}
/* --- SVG chart builder (hand-built, no library) --- */
function lineChart(seriesList, xLabel, yMax, pointX, onClick) {
  var W = 620, H = 260, L = 46, R = 14, T = 14, B = 34, w = W - L - R, h = H - T - B;
  var xMin = seriesList[0].xs[0], xMax = seriesList[0].xs[seriesList[0].xs.length - 1];
  function X(x) { return L + (x - xMin) / (xMax - xMin) * w; }
  function Y(y) { return T + h - (y / yMax) * h; }
  var grid = "";
  [0, 0.25, 0.5, 0.75, 1].forEach(function (f) {
    var y = Y(yMax * f);
    grid += '<line x1="' + L + '" y1="' + y.toFixed(1) + '" x2="' + (L + w) + '" y2="' + y.toFixed(1) + '" stroke="#e8e1d0"/>' +
      '<text x="' + (L - 6) + '" y="' + (y + 4).toFixed(1) + '" font-size="11" fill="#5d574c" text-anchor="end">' + Math.round(yMax * f * 100) + "%</text>";
  });
  for (var i = 0; i <= 4; i++) {
    var xv = xMin + (xMax - xMin) * i / 4;
    grid += '<text x="' + X(xv).toFixed(1) + '" y="' + (H - 10) + '" font-size="11" fill="#5d574c" text-anchor="middle">' + Math.round(xv) + "</text>";
  }
  var paths = seriesList.map(function (s) {
    var d = s.xs.map(function (x, i) { return (i ? "L" : "M") + X(x).toFixed(1) + " " + Y(s.ys[i]).toFixed(1); }).join(" ");
    return '<path d="' + d + '" fill="none" stroke="' + s.color + '" stroke-width="' + (s.width || 2) + '"/>';
  }).join("");
  var pt = "";
  if (pointX !== null && pointX !== undefined) {
    var py = seriesList[0].ys.reduce(function (best, y, i) {
      return Math.abs(seriesList[0].xs[i] - pointX) < Math.abs(seriesList[0].xs[best] - pointX) ? i : best;
    }, 0);
    pt = '<circle cx="' + X(seriesList[0].xs[py]).toFixed(1) + '" cy="' + Y(seriesList[0].ys[py]).toFixed(1) + '" r="5" fill="#b42318" stroke="#fffdf7" stroke-width="2"/>';
  }
  var click = onClick ? ' data-click="1" style="cursor:pointer"' : "";
  return '<svg class="chart" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + xLabel + ' chart"' + click + ">" + grid + paths + pt +
    '<text x="' + (L + w / 2) + '" y="' + (H - 22) + '" font-size="11" fill="#5d574c" text-anchor="middle"></text></svg>';
}
function variedCurve(base, field, xs) {
  return xs.map(function (v) {
    var inp = Object.assign({}, base);
    if (field === "cigsPerDay") { inp.currentSmoker = v > 0 ? 1 : 0; inp.cigsPerDay = v; }
    else inp[field] = v;
    return hrPredict(MODEL, inp);
  });
}

/* --- rendering --- */
function render() {
  if (!MODEL) return;
  var base = readInput(), p = hrPredict(MODEL, base);
  SLIDERS.forEach(syncOutput);
  // warnings: training-range + consistency checks
  var warns = [];
  MODEL.features.forEach(function (f) {
    var r = MODEL.dataset.training_ranges[f];
    if (base[f] < r[0] || base[f] > r[1]) warns.push(MODEL.labels[f] + " is outside the training range.");
  });
  if (base.diaBP >= base.sysBP) warns.push("Diastolic BP should be below systolic BP — check your entries.");
  if (segValue("smoking") !== "current" && Number($("cigs").value) > 0) warns.push("Cigarettes per day is ignored unless smoking status is Current.");
  $("warnings").textContent = warns.join(" ");

  var band = hrBand(MODEL, p);
  $("riskBig").textContent = fmtPct(p);
  var bandEl = $("riskBand");
  bandEl.className = "band " + (band === "low" ? "low" : band === "moderate" ? "mod" : "high");
  bandEl.textContent = band === "low" ? "Low band" : band === "moderate" ? "Moderate band" : "High band";
  $("riskPlain").textContent = "A probability of " + fmtPct(p) + " means the model places about " +
    Math.round(p * 100) + " in 100 similar historical profiles in the event group. It is not a personal prediction, and it is not your fate.";

  // Figure 1 — risk over age
  var ages = []; for (var a = 32; a <= 70; a += 2) ages.push(a);
  var userCurve = ages.map(function (age) { return hrPredict(MODEL, Object.assign({}, base, { age: age })); });
  var nosmoke = ages.map(function (age) {
    return hrPredict(MODEL, Object.assign({}, base, { age: age, currentSmoker: 0, cigsPerDay: 0 }));
  });
  var yMax1 = Math.min(1, Math.max(0.4, Math.ceil(Math.max.apply(null, userCurve.concat(nosmoke)) * 1.25 * 10) / 10));
  $("ageChart").innerHTML = lineChart([
    { xs: ages, ys: userCurve, color: "#b42318", width: 2.5 },
    { xs: ages, ys: nosmoke, color: "#9a9384", width: 1.8 }
  ], "Risk over age", yMax1, base.age) +
  '<p class="tiny" style="margin-top:6px"><span style="color:#b42318">— Your profile</span> &nbsp; <span style="color:#9a9384">— Same profile, non-smoker</span></p>';

  // Figure 2 — risk vs selected factor
  var factor = $("factorSelect").value, fr = FACTOR_RANGES[factor], field = FIELD_OF_FACTOR[factor];
  var xs = []; for (var v = fr[0]; v <= fr[1]; v += fr[2]) xs.push(v);
  var curVal = field === "cigsPerDay" ? (base.currentSmoker ? base.cigsPerDay : 0) : base[field];
  var ys = variedCurve(base, field, xs);
  var yMax2 = Math.min(1, Math.max(0.4, Math.ceil(Math.max.apply(null, ys) * 1.25 * 10) / 10));
  var fc = $("factorChart");
  fc.innerHTML = lineChart([{ xs: xs, ys: ys, color: "#b42318", width: 2.5 }], "Risk vs factor", yMax2, curVal, true);
  fc.querySelector("svg").addEventListener("click", function (ev) {
    var rect = ev.currentTarget.getBoundingClientRect();
    var frac = (ev.clientX - rect.left) / rect.width;
    // viewBox x maps: left pad 46/620, right pad 14/620
    var t = Math.min(1, Math.max(0, (frac * 620 - 46) / (620 - 46 - 14)));
    var val = Math.round((fr[0] + t * (fr[1] - fr[0])) / fr[2]) * fr[2];
    if (factor === "cigs") {
      $("cigs").value = val;
      setSeg("smoking", val > 0 ? "current" : "never");
    } else if (factor === "sys") $("sys").value = val;
    else if (factor === "chol") $("chol").value = val;
    else if (factor === "bmi") $("bmi").value = val;
    else if (factor === "glu") $("glu").value = val;
    else if (factor === "hr") $("hr").value = val;
    render();
  });

  // Top factors: one factor replaced by its reference value, rest held.
  var groups = [
    { name: "Age", fields: ["age"] }, { name: "Smoking (status + cigarettes)", fields: ["currentSmoker", "cigsPerDay"] },
    { name: "Systolic BP", fields: ["sysBP"] }, { name: "Total cholesterol", fields: ["totChol"] },
    { name: "Diastolic BP", fields: ["diaBP"] }, { name: "BMI", fields: ["BMI"] },
    { name: "Glucose", fields: ["glucose"] }, { name: "Heart rate", fields: ["heartRate"] },
    { name: "Sex", fields: ["male"] }, { name: "Diabetes (context)", fields: ["diabetes"] },
    { name: "Prevalent hypertension", fields: ["prevalentHyp"] },
    { name: "BP medication", fields: ["BPMeds"] }, { name: "Prevalent stroke", fields: ["prevalentStroke"] }
  ];
  var facs = groups.map(function (g) {
    var inp = Object.assign({}, base);
    g.fields.forEach(function (f) { inp[f] = MODEL.reference[f]; });
    return { name: g.name, delta: p - hrPredict(MODEL, inp) };
  }).filter(function (f) { return f.delta > 0.001; }).sort(function (a, b) { return b.delta - a.delta; }).slice(0, 5);
  var maxD = facs.length ? facs[0].delta : 0.01;
  $("factors").innerHTML = facs.length ? facs.map(function (f) {
    return '<div class="factor"><div class="frow"><span>' + f.name + '</span><strong>+' + (f.delta * 100).toFixed(1) + " pp</strong></div>" +
      '<div class="bar"><i style="width:' + Math.round(f.delta / maxD * 100) + '%"></i></div></div>';
  }).join("") : '<p class="small">No single factor pushes your estimate above the low-risk reference profile.</p>';

  renderScenarios(base, p);
  renderInsights(base, p);
}
function scenarioDefs(base) {
  var defs = [];
  if (base.currentSmoker) defs.push({ name: "Quit smoking", desc: "Current smoker → non-smoker, cigarettes → 0", apply: { currentSmoker: 0, cigsPerDay: 0 }, ui: { smoking: "never", cigs: 0 } });
  if (base.currentSmoker && base.cigsPerDay >= 4) defs.push({ name: "Halve cigarettes per day", desc: base.cigsPerDay + " → " + Math.floor(base.cigsPerDay / 2) + " per day", apply: { cigsPerDay: Math.floor(base.cigsPerDay / 2) }, ui: { cigs: Math.floor(base.cigsPerDay / 2) } });
  if (base.sysBP > 90) defs.push({ name: "Lower systolic BP by 10", desc: base.sysBP + " → " + (base.sysBP - 10) + " mmHg", apply: { sysBP: base.sysBP - 10 }, ui: { sys: base.sysBP - 10 } });
  if (base.sysBP > 100) defs.push({ name: "Lower systolic BP by 20", desc: base.sysBP + " → " + (base.sysBP - 20) + " mmHg", apply: { sysBP: base.sysBP - 20 }, ui: { sys: base.sysBP - 20 } });
  if (base.totChol > 120) defs.push({ name: "Lower total cholesterol by 20", desc: base.totChol + " → " + (base.totChol - 20) + " mg/dL", apply: { totChol: base.totChol - 20 }, ui: { chol: base.totChol - 20 } });
  if (base.totChol > 140) defs.push({ name: "Lower total cholesterol by 40", desc: base.totChol + " → " + (base.totChol - 40) + " mg/dL", apply: { totChol: base.totChol - 40 }, ui: { chol: base.totChol - 40 } });
  if (base.BMI > 17) defs.push({ name: "Lower BMI by 2", desc: base.BMI.toFixed(1) + " → " + (base.BMI - 2).toFixed(1), apply: { BMI: base.BMI - 2 }, ui: { bmi: base.BMI - 2 } });
  if (base.BMI > 20) defs.push({ name: "Lower BMI by 5", desc: base.BMI.toFixed(1) + " → " + (base.BMI - 5).toFixed(1), apply: { BMI: base.BMI - 5 }, ui: { bmi: base.BMI - 5 } });
  return defs.map(function (d) {
    d.prob = hrPredict(MODEL, Object.assign({}, base, d.apply));
    return d;
  }).sort(function (a, b) { return a.prob - b.prob; });
}
function renderScenarios(base, p) {
  var defs = scenarioDefs(base).map(function (d) { d.delta = d.prob - p; return d; })
    .filter(function (d) { return d.delta < -0.001; }).sort(function (a, b) { return a.delta - b.delta; });
  var el = $("scenarios");
  if (!defs.length) { el.innerHTML = '<p class="small">No listed scenario moves this model estimate by at least 0.1 percentage points from your current inputs.</p>'; }
  else {
    el.innerHTML = defs.map(function (d, i) {
      var idx = PINNED.findIndex(function (x) { return x.name === d.name; });
      return '<div class="scen"><div class="top"><div><strong>' + (i + 1) + ". " + d.name + '</strong><div class="small">' + d.desc + '</div></div>' +
        '<span class="delta">' + fmtPct(d.prob) + " (" + (d.delta * 100).toFixed(1) + " pp)</span></div>" +
        '<button class="btn ghost" type="button" data-apply="' + i + '">Apply to my profile</button> ' +
        '<button class="btn ghost" type="button" data-pin="' + i + '">' + (idx >= 0 ? "Pinned ✓" : "Pin to compare") + "</button></div>";
    }).join("");
    el.querySelectorAll("[data-apply]").forEach(function (b) {
      b.addEventListener("click", function () {
        var d = defs[Number(b.getAttribute("data-apply"))];
        if (d.ui.smoking) setSeg("smoking", d.ui.smoking);
        Object.keys(d.ui).forEach(function (k) { if (k !== "smoking" && $(k)) $(k).value = d.ui[k]; });
        render();
      });
    });
    el.querySelectorAll("[data-pin]").forEach(function (b) {
      b.addEventListener("click", function () {
        var d = defs[Number(b.getAttribute("data-pin"))];
        var idx = PINNED.findIndex(function (x) { return x.name === d.name; });
        if (idx >= 0) PINNED.splice(idx, 1);
        else if (PINNED.length < 3) PINNED.push(d);
        render();
      });
    });
  }
  // pins (recomputed against current baseline for display; pinned deltas are live)
  var pinsEl = $("pins");
  var cards = ['<div class="pin"><strong>Current (baseline)</strong><div>' + fmtPct(p) + "</div></div>"];
  PINNED.forEach(function (d, i) {
    cards.push('<div class="pin"><strong>' + d.name + '</strong><div>' + fmtPct(d.prob) + " (" + (d.delta * 100).toFixed(1) + " pp)</div>" +
      '<button type="button" data-unpin="' + i + '">Remove</button></div>');
  });
  pinsEl.innerHTML = cards.join("");
  pinsEl.querySelectorAll("[data-unpin]").forEach(function (b) {
    b.addEventListener("click", function () { PINNED.splice(Number(b.getAttribute("data-unpin")), 1); render(); });
  });
}
function renderInsights(base, p) {
  var defs = scenarioDefs(base).map(function (d) { d.delta = d.prob - p; return d; })
    .filter(function (d) { return d.delta < -0.001; }).sort(function (a, b) { return a.delta - b.delta; });
  var top = defs.length ? defs[0] : null;
  $("insights").innerHTML = top
    ? "<p><strong>Largest modelled lever:</strong> " + top.name + " — the model estimates " + fmtPct(top.prob) +
      " versus " + fmtPct(p) + " now (" + (top.delta * 100).toFixed(1) + " percentage points). That is a model association, not a guaranteed personal outcome, and it says nothing about how to achieve the change safely.</p>"
    : "<p>No listed scenario moves this model estimate by at least 0.1 percentage points. That does not mean nothing matters — it means this model's inputs, at your current values, are not sensitive to those single-factor changes.</p>";
}
function calibSVG(bins) {
  var W = 320, H = 220, L = 36, B = 26, T = 8, R = 8, iw = W - L - R, ih = H - T - B;
  function X(v) { return L + v * iw; }
  function Y(v) { return T + ih - v * ih; }
  var pts = bins.map(function (b) { return X(b.mean_predicted) + "," + Y(b.observed_rate); }).join(" ");
  return "<svg viewBox='0 0 " + W + " " + H + "' role='img' aria-label='Calibration curve: mean predicted risk against observed rate per decile bin' style='width:100%;max-width:420px;height:auto;display:block'>" +
    "<line x1='" + L + "' y1='" + Y(0) + "' x2='" + (W - R) + "' y2='" + Y(0) + "' stroke='#d8cbb2'/>" +
    "<line x1='" + L + "' y1='" + T + "' x2='" + L + "' y2='" + Y(0) + "' stroke='#d8cbb2'/>" +
    "<line x1='" + L + "' y1='" + Y(0) + "' x2='" + (W - R) + "' y2='" + T + "' stroke='#9aa0a6' stroke-dasharray='4 4'/>" +
    "<polyline points='" + pts + "' fill='none' stroke='#b42318' stroke-width='2'/>" +
    bins.map(function (b) { return "<circle cx='" + X(b.mean_predicted) + "' cy='" + Y(b.observed_rate) + "' r='3.5' fill='#b42318'/>"; }).join("") +
    "<text x='" + (L + iw / 2) + "' y='" + (H - 4) + "' text-anchor='middle' font-size='10' fill='#6b6259'>Predicted risk</text>" +
    "<text x='10' y='" + (T + ih / 2) + "' text-anchor='middle' font-size='10' fill='#6b6259' transform='rotate(-90 10 " + (T + ih / 2) + ")'>Observed rate</text>" +
    "</svg><p class='small'>Dashed line = perfect calibration. Dots = decile bins on the held-out test set (n≈85 each). The top bin is the noisiest; the curve is estimated on one historical cohort, not on an independent population.</p>";
}
function renderMetrics() {
  var M = MODEL.metrics, shipped = MODEL.shipped_test_metrics;
  function row(name, x, cv) {
    return "<tr><td>" + name + "</td><td>" + x.accuracy + "</td><td>" + x.roc_auc + "</td><td>" + x.pr_auc_average_precision + "</td><td>" + x.precision + "</td><td>" + x.recall_sensitivity + "</td><td>" + x.specificity + "</td><td>" + x["f1_at_0.5"] + "</td><td>" + x.brier_score + "</td><td>" + x.log_loss + "</td><td>" + x.calibration_intercept + " / " + x.calibration_slope + "</td><td>" + (cv || "—") + "</td></tr>";
  }
  function candRow(name, key) {
    return row(name, M[key].test.sigmoid, M[key].cv_roc_auc_mean + " ± " + M[key].cv_roc_auc_std);
  }
  $("metrics").innerHTML =
    "<div class='disclaimer' role='note'><strong>Not enough evidence for clinical decisions.</strong> " +
    "These numbers come from one held-out test set on one historical cohort, with calibration fitted on one calibration split. " +
    "Stronger calibration evidence and evaluation on an independent, relevant population are still required before any individual risk estimate here could be trusted for a clinical decision. " +
    "This product is educational only.</div>" +
    "<p><strong>Shipped model (sigmoid-calibrated):</strong> regularized logistic regression, C and class_weight tuned by CV on train only, " +
    "calibration fitted on the calibration split only. Split: " + MODEL.split.design + " (train " + MODEL.split.train_rows + ", calibration " + MODEL.split.calibration_rows + ", test " + MODEL.split.test_rows + ").</p>" +
    "<table><thead><tr><th>Model (test, sigmoid-calibrated)</th><th>Accuracy</th><th>ROC-AUC</th><th>PR-AUC</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1 @ 0.5</th><th>Brier</th><th>Log loss</th><th>Cal. intercept / slope</th><th>CV ROC-AUC</th></tr></thead><tbody>" +
    candRow("Logistic regression, regularized — shipped", "lr_regularized") +
    candRow("Random forest — compared", "random_forest") +
    candRow("HistGradientBoosting — compared", "hist_gb") +
    candRow("Logistic regression, unregularized baseline", "lr_unregularized_baseline") + "</tbody></table>" +
    "<p class='small'>Shipped confusion matrix (rows actual, columns predicted at 0.5): [[" + shipped.confusion_matrix[0].join(", ") + "], [" + shipped.confusion_matrix[1].join(", ") + "]]. " +
    "Only " + shipped.confusion_matrix[1][1] + " of " + (shipped.confusion_matrix[1][0] + shipped.confusion_matrix[1][1]) + " test cases with the outcome were predicted positive at 0.5 — hence the low F1 and recall. " +
    "That is why this UI shows probabilities, never a bare yes/no verdict: a 0.5 cut on a 15% prevalence outcome is a poor summary.</p>" +
    "<p class='small'><strong>Why calibration matters:</strong> if 100 people are given a predicted risk of 20%, a calibrated model means about 20 of them should experience the outcome over 10 years, on average, in an appropriate population. It will not be exactly 20 every time; what matters is agreement across the whole range of risks. " +
    "The tuned model's raw probabilities were badly miscalibrated (uncalibrated test Brier " + M.lr_regularized.test.uncalibrated.brier_score + ", intercept " + M.lr_regularized.test.uncalibrated.calibration_intercept + ") because class_weight='balanced' inflates them. " +
    "Sigmoid calibration fitted on the calibration split only restores them (test Brier " + shipped.brier_score + ", intercept " + shipped.calibration_intercept + ", slope " + shipped.calibration_slope + "). " +
    "Isotonic calibration was tied on cross-fitted Brier but worse on log loss and unstable, so it was not shipped. Full tables: EVALUATION.md.</p>" +
    "<h3 style='margin-top:18px'>Figure — calibration curve (shipped model, test set)</h3>" + calibSVG(MODEL.calibration_bins) +
    "<p class='small'>Calibration bins (mean predicted → observed rate, n): " + MODEL.calibration_bins.map(function (b) { return b.mean_predicted + " → " + b.observed_rate + " (n=" + b.n + ")"; }).join("; ") + ".</p>" +
    "<p class='small'><strong>Selection:</strong> " + MODEL.selection_reason + "</p>";
  $("glossary").innerHTML = MODEL.features.map(function (f) {
    var r = MODEL.dataset.training_ranges[f];
    return "<tr><td>" + f + "</td><td>" + MODEL.labels[f] + "</td><td>" + r[0] + " – " + r[1] + "</td></tr>";
  }).join("");
}
/* --- wiring --- */
document.addEventListener("DOMContentLoaded", function () {
  fetch("model.json").then(function (r) { if (!r.ok) throw new Error("model.json failed"); return r.json(); })
    .then(function (m) {
      MODEL = m; renderMetrics();
      document.querySelectorAll('input[type="range"]').forEach(function (el) {
        el.addEventListener("input", function () { render(); });
      });
      document.querySelectorAll('[data-field]').forEach(function (b) {
        b.addEventListener("click", function () { setSeg(b.getAttribute("data-field"), b.getAttribute("data-value")); render(); });
      });
      ["bpmeds", "stroke", "hyp", "diabetes"].forEach(function (id) {
        $(id).addEventListener("change", render);
      });
      $("factorSelect").addEventListener("change", render);
      document.querySelectorAll("[data-profile]").forEach(function (b) {
        b.addEventListener("click", function () { PINNED = []; setControls(PROFILES[b.getAttribute("data-profile")]); render(); });
      });
      $("heroSample").addEventListener("click", function () { PINNED = []; setControls(PROFILES.a); document.getElementById("simulator").scrollIntoView(); render(); });
      $("resetBtn").addEventListener("click", function () { PINNED = []; setControls(PROFILES.a); render(); });
      setControls(PROFILES.a); render();
    })
    .catch(function (e) {
      $("riskBig").textContent = "Model failed to load";
      $("riskPlain").textContent = "The model file (model.json) could not be loaded. Serve the docs/ folder over HTTP (see README) rather than opening the file directly, if that is the cause.";
    });
});
