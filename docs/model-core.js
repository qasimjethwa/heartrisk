/* HeartRisk model core — pure, dependency-free, no DOM.
   Mirrors src/train.py exactly: median imputation (train medians),
   standard scaling (train mean/scale), logistic regression, sigmoid.
   Used by app.js in the browser and by src/parity.js under Node, so a
   parity failure means the browser really diverges from Python. */
"use strict";
function hrPredict(model, input) {
  var z = model.intercept;
  model.features.forEach(function (f, i) {
    var v = input[f];
    if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) {
      v = model.medians[f];
    }
    var x = (Number(v) - model.scaler_mean[i]) / model.scaler_scale[i];
    z += model.coefficients[i] * x;
  });
  return 1 / (1 + Math.exp(-z));
}
function hrBand(model, p) {
  if (p < model.display_bands.low_max) return "low";
  if (p <= model.display_bands.moderate_max) return "moderate";
  return "high";
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = { hrPredict: hrPredict, hrBand: hrBand };
}
