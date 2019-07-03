import "./styles.css";

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as Papa from "papaparse";
import _ from "lodash";
import * as Plotly from "plotly.js";

// data from:
// https://www.kaggle.com/uciml/pima-indians-diabetes-database/kernels
// https://datahub.io/machine-learning/diabetes#resource-diabetes_arff

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

const prepareData = async () => {
  const csv = await Papa.parsePromise(
    "https://raw.githubusercontent.com/curiousily/Logistic-Regression-with-TensorFlow-js/master/src/data/diabetes.csv"
  );

  return csv.data;
};

const toTensors = (csvData, testSize) => {
  const data = _.shuffle(csvData);

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

const renderOutcomes = data => {
  const outcomes = data.map(r => r.Outcome);

  const [diabetic, healthy] = _.partition(outcomes, o => o === 1);

  const chartData = [
    {
      labels: ["Diabetic", "Healthy"],
      values: [diabetic.length, healthy.length],
      type: "pie",
      opacity: 0.6,
      marker: {
        colors: ["gold", "forestgreen"]
      }
    }
  ];

  Plotly.newPlot("outcome-cont", chartData, {
    title: "Healthy vs Diabetic"
  });
};

const renderHistogram = (container, data, column, config) => {
  const diabetic = data.filter(r => r.Outcome === 1).map(r => r[column]);
  const healthy = data.filter(r => r.Outcome === 0).map(r => r[column]);

  const dTrace = {
    name: "diabetic",
    x: diabetic,
    type: "histogram",
    opacity: 0.6,
    marker: {
      color: "gold"
    }
  };

  const hTrace = {
    name: "healthy",
    x: healthy,
    type: "histogram",
    opacity: 0.4,
    marker: {
      color: "forestgreen"
    }
  };

  Plotly.newPlot(container, [dTrace, hTrace], {
    barmode: "overlay",
    xaxis: {
      title: config.xLabel
    },
    yaxis: { title: "Count" },
    title: config.title
  });
};

const renderScatter = (container, data, columns, config) => {
  const diabetic = data.filter(r => r.Outcome === 1);
  const healthy = data.filter(r => r.Outcome === 0);

  var dTrace = {
    x: diabetic.map(r => r[columns[0]]),
    y: diabetic.map(r => r[columns[1]]),
    mode: "markers",
    type: "scatter",
    name: "Diabetic",
    opacity: 0.4,
    marker: {
      color: "gold"
    }
  };

  var hTrace = {
    x: healthy.map(r => r[columns[0]]),
    y: healthy.map(r => r[columns[1]]),
    mode: "markers",
    type: "scatter",
    name: "Healthy",
    opacity: 0.4,
    marker: {
      color: "forestgreen"
    }
  };

  var chartData = [dTrace, hTrace];

  Plotly.newPlot(container, chartData, {
    title: config.title,
    xaxis: {
      title: config.xLabel
    },
    yaxis: { title: config.yLabel }
  });
};

const run = async () => {
  const data = await prepareData();

  renderOutcomes(data);

  renderHistogram("insulin-cont", data, "Insulin", {
    title: "Insulin levels",
    xLabel: "Insulin 2-hour serum, mu U/ml"
  });

  renderHistogram("glucose-cont", data, "Glucose", {
    title: "Glucose concentration",
    xLabel: "Plasma glucose concentration (2 hour after glucose tolerance test)"
  });

  renderHistogram("age-cont", data, "Age", {
    title: "Age",
    xLabel: "age (years)"
  });

  renderScatter("glucose-age-cont", data, ["Glucose", "Age"], {
    title: "Glucose vs Age",
    xLabel: "Glucose",
    yLabel: "Age"
  });

  renderScatter("skin-bmi-cont", data, ["SkinThickness", "BMI"], {
    title: "Skin thickness vs BMI",
    xLabel: "Skin thickness",
    yLabel: "BMI"
  });

  // const [xTrain, xTest, yTrain, yTest] = toTensors(data, 0.1);

  // console.log(xTrain.shape);
  // const model = tf.sequential();
  // model.add(
  //   tf.layers.dense({
  //     units: 32,
  //     activation: "relu",
  //     inputShape: [xTrain.shape[1]]
  //   })
  // );
  // model.add(tf.layers.dense({ units: 2, activation: "softmax" }));
  // const optimizer = tf.train.adam(0.001);
  // model.compile({
  //   optimizer: optimizer,
  //   loss: "categoricalCrossentropy",
  //   metrics: ["accuracy"]
  // });
  // const trainLogs = [];
  // const lossContainer = document.getElementById("loss-cont");
  // const accContainer = document.getElementById("acc-cont");
  // console.log("Training...");
  // await model.fit(xTrain, yTrain, {
  //   validationData: [xTest, yTest],
  //   epochs: 100,
  //   callbacks: {
  //     onEpochEnd: async (epoch, logs) => {
  //       trainLogs.push(logs);
  //       tfvis.show.history(lossContainer, trainLogs, ["loss", "val_loss"]);
  //       tfvis.show.history(accContainer, trainLogs, ["acc", "val_acc"]);
  //     }
  //   }
  // });

  // const preds = model.predict(tf.slice2d(xTest, 1, 1)).dataSync();
  // console.log("Prediction:" + preds);
  // console.log("Real:" + yTest.slice(1, 1).dataSync());
};

if (document.readyState !== "loading") {
  run();
} else {
  document.addEventListener("DOMContentLoaded", run);
}
