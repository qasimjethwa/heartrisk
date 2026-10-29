// Node parity runner: loads docs/model-core.js + docs/model.json, recomputes
// the probabilities for src/parity_cases.json, and compares them with the
// sklearn probabilities in src/parity_expected.json.
// Run: node src/parity.js   (from the repository root)
const fs = require("fs");
const path = require("path");
const { hrPredict } = require("../docs/model-core.js");
const root = path.join(__dirname, "..");
const model = JSON.parse(fs.readFileSync(path.join(root, "docs/model.json"), "utf8"));
const cases = JSON.parse(fs.readFileSync(path.join(root, "src/parity_cases.json"), "utf8"));
const expected = JSON.parse(fs.readFileSync(path.join(root, "src/parity_expected.json"), "utf8"));
let maxDiff = 0, worst = null;
cases.forEach((c, i) => {
  const input = {};
  model.features.forEach((f) => { input[f] = c[f] === undefined ? null : c[f]; });
  const js = hrPredict(model, input);
  const diff = Math.abs(js - expected[i].python_probability);
  if (diff > maxDiff) { maxDiff = diff; worst = expected[i].case; }
  if (diff > 1e-4) console.log("FAIL", expected[i].case, "js", js, "py", expected[i].python_probability, "diff", diff);
});
console.log(`cases=${cases.length} max_abs_diff=${maxDiff.toExponential(3)} worst=${worst}`);
console.log(maxDiff < 1e-4 ? "PARITY PASS (<1e-4)" : "PARITY FAIL");
process.exit(maxDiff < 1e-4 ? 0 : 1);
