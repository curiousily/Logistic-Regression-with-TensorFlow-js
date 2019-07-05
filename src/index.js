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

const oneHot = outcome => Array.from(tf.oneHot(outcome, 2).dataSync());

const prepareData = async () => {
  const csv = await Papa.parsePromise(
    "https://raw.githubusercontent.com/curiousily/Logistic-Regression-with-TensorFlow-js/master/src/data/diabetes.csv"
  );

  return csv.data;
};

const createDataSets = (data, features, testSize, batchSize) => {
  const X = data.map(r =>
    features.map(f => {
      const val = r[f];
      return val === undefined ? 0 : val;
    })
  );
  const y = data.map(r => {
    const outcome = r.Outcome === undefined ? 0 : r.Outcome;
    return oneHot(outcome);
  });

  const splitIdx = parseInt((1 - testSize) * data.length, 10);

  const ds = tf.data
    .zip({ xs: tf.data.array(X), ys: tf.data.array(y) })
    .shuffle(data.length, 42);

  return [
    ds.take(splitIdx).batch(batchSize),
    ds.skip(splitIdx + 1).batch(batchSize),
    tf.tensor(X.slice(splitIdx)),
    tf.tensor(y.slice(splitIdx))
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

const trainLogisticRegression = async (featureCount, trainDs, validDs) => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 2,
      activation: "softmax",
      inputShape: [featureCount]
    })
  );
  const optimizer = tf.train.adam(0.001);
  model.compile({
    optimizer: optimizer,
    loss: "binaryCrossentropy",
    metrics: ["accuracy"]
  });
  const trainLogs = [];
  const lossContainer = document.getElementById("loss-cont");
  const accContainer = document.getElementById("acc-cont");
  console.log("Training...");
  await model.fitDataset(trainDs, {
    epochs: 100,
    validationData: validDs,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        trainLogs.push(logs);
        tfvis.show.history(lossContainer, trainLogs, ["loss", "val_loss"]);
        tfvis.show.history(accContainer, trainLogs, ["acc", "val_acc"]);
      }
    }
  });

  return model;
};

const trainComplexModel = async (featureCount, trainDs, validDs) => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 12,
      activation: "relu",
      inputShape: [featureCount]
    })
  );
  model.add(
    tf.layers.dense({
      units: 2,
      activation: "softmax"
    })
  );
  const optimizer = tf.train.adam(0.0001);
  model.compile({
    optimizer: optimizer,
    loss: "binaryCrossentropy",
    metrics: ["accuracy"]
  });
  const trainLogs = [];
  const lossContainer = document.getElementById("loss-cont");
  const accContainer = document.getElementById("acc-cont");
  console.log("Training...");
  await model.fitDataset(trainDs, {
    epochs: 100,
    validationData: validDs,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        trainLogs.push(logs);
        tfvis.show.history(lossContainer, trainLogs, ["loss", "val_loss"]);
        tfvis.show.history(accContainer, trainLogs, ["acc", "val_acc"]);
      }
    }
  });

  return model;
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

  const features = ["Glucose"];

  const [trainDs, validDs, xTest, yTest] = createDataSets(
    data,
    features,
    0.1,
    16
  );

  const model = await trainLogisticRegression(
    features.length,
    trainDs,
    validDs
  );

  // const features = ["Glucose", "Age", "Insulin", "BloodPressure"];

  // const [trainDs, validDs, xTest, yTest] = createDataSets(
  //   data,
  //   features,
  //   0.1,
  //   16
  // );

  // const model = await trainComplexModel(features.length, trainDs, validDs);

  const preds = model.predict(xTest).argMax(-1);
  const labels = yTest.argMax(-1);

  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);

  const container = document.getElementById("confusion-matrix");

  tfvis.render.confusionMatrix(container, {
    values: confusionMatrix,
    tickLabels: ["Healthy", "Diabetic"]
  });
};

if (document.readyState !== "loading") {
  run();
} else {
  document.addEventListener("DOMContentLoaded", run);
}
