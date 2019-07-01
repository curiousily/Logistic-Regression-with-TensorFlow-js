import "./styles.css";

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as Papa from "papaparse";
import _ from "lodash";

// data from:
// https://www.kaggle.com/kandij/diabetes-dataset

Papa.parsePromise = function(file) {
  return new Promise(function(complete, error) {
    Papa.parse(file, {
      header: true,
      download: true,
      dynamicTyping: true,
      complete,
      error
    });
  });
};

const oneHot = outcome => Array.from(tf.oneHot([outcome], 2).dataSync());

const prepareData = async testSize => {
  const csv = await Papa.parsePromise(
    "https://raw.githubusercontent.com/curiousily/Logistic-Regression-with-TensorFlow-js/master/src/data/diabetes.csv"
  );

  const data = _.shuffle(csv.data);

  const X = data.map(r =>
    Object.values(r).slice(0, Object.values(r).length - 1)
  );
  const y = data.map(r => oneHot(r.Outcome));

  const [xTrain, xTest] = _.chunk(X, parseInt((1 - testSize) * X.length, 10));
  const [yTrain, yTest] = _.chunk(y, parseInt((1 - testSize) * y.length, 10));

  return [
    tf.tensor2d(xTrain),
    tf.tensor(xTest),
    tf.tensor2d(yTrain),
    tf.tensor(yTest)
  ];
};

const run = async () => {
  const [xTrain, xTest, yTrain, yTest] = await prepareData(0.1);

  console.log(xTrain.shape);
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
      inputShape: [xTrain.shape[1]]
    })
  );
  model.add(tf.layers.dense({ units: 2, activation: "softmax" }));
  const optimizer = tf.train.adam(0.001);
  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });
  const trainLogs = [];
  const lossContainer = document.getElementById("loss-cont");
  const accContainer = document.getElementById("acc-cont");
  console.log("Training...");
  await model.fit(xTrain, yTrain, {
    validationData: [xTest, yTest],
    epochs: 100,
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        trainLogs.push(logs);
        tfvis.show.history(lossContainer, trainLogs, ["loss", "val_loss"]);
        tfvis.show.history(accContainer, trainLogs, ["acc", "val_acc"]);
      }
    }
  });

  const preds = model.predict(tf.slice2d(xTest, 1, 1)).dataSync();
  console.log("Prediction:" + preds);
  console.log("Real:" + yTest.slice(1, 1).dataSync());
};

if (document.readyState !== "loading") {
  run();
} else {
  document.addEventListener("DOMContentLoaded", run);
}
