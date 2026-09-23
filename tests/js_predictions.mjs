// Called from test_export.py: run the browser models over a list of
// feature rows and print the probabilities as JSON.
import { readFileSync } from "node:fs";
import { predict } from "../web/js/model.js";

const [modelPath, calPath, method, rowsPath] = process.argv.slice(2);
const model = JSON.parse(readFileSync(modelPath, "utf8"));
const cal = JSON.parse(readFileSync(calPath, "utf8"))[model.type];
const rows = JSON.parse(readFileSync(rowsPath, "utf8"));
console.log(JSON.stringify(rows.map((r) => predict(model, cal, method, r))));
