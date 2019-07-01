import "./styles.css";

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as Papa from "papaparse";

// data from:
// https://www.kaggle.com/kandij/diabetes-dataset

Papa.parsePromise = function(file) {
  return new Promise(function(complete, error) {
    Papa.parse(file, { complete, error });
  });
};

const run = async () => {
  console.log(new File("./diabetes.csv"));
  // const csv = await Papa.parsePromise(new File("./data/diabetes.csv"));
  // console.log(csv);
};

if (document.readyState !== "loading") {
  run();
} else {
  document.addEventListener("DOMContentLoaded", run);
}
