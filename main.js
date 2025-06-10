
function yFunction(x) {
  return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

function generateXValues(n = 100, min = -2, max = 2) {
  return Array.from({ length: n }, () => Math.random() * (max - min) + min);
}

function calculateY(xVals) {
  return xVals.map(yFunction);
}

function addGaussianNoise(yVals, variance = 0.05) {
  return yVals.map(y => y + tf.randomNormal([1], 0, Math.sqrt(variance)).dataSync()[0]);
}

function splitTrainTest(xVals, yVals) {
  const indices = tf.util.createShuffledIndices(xVals.length);
  const half = Math.floor(xVals.length / 2);
  const xTrain = [], yTrain = [], xTest = [], yTest = [];
  indices.forEach((i, idx) => {
    if (idx < half) {
      xTrain.push(xVals[i]);
      yTrain.push(yVals[i]);
    } else {
      xTest.push(xVals[i]);
      yTest.push(yVals[i]);
    }
  });
  return { xTrain, yTrain, xTest, yTest };
}

function plotData(canvasId, xTrain, yTrain, xTest, yTest, predictions = null) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  const datasets = [];

  if (xTrain.length && yTrain.length) {
    datasets.push({
      label: 'Trainingsdaten',
      data: xTrain.map((x, i) => ({ x, y: yTrain[i] })),
      backgroundColor: 'rgba(0,123,255,0.8)',
      borderColor: 'rgba(0,123,255,0.8)',
      pointRadius: 4,
      pointStyle: 'rectRounded',
      showLine: false
    });
  }

  if (xTest.length && yTest.length) {
    datasets.push({
      label: 'Testdaten',
      data: xTest.map((x, i) => ({ x, y: yTest[i] })),
      backgroundColor: 'rgba(255,105,180,1)',
      borderColor: 'rgba(255,105,180,1)',
      pointStyle: 'circle',
      pointRadius: 5,
      showLine: false
    });
  }

  if (predictions) {
    datasets.push({
      label: 'Vorhersage',
      data: predictions.map(([x, y]) => ({ x, y })),
      borderColor: 'rgba(0,200,0,1)',
      backgroundColor: 'rgba(0,200,0,0.1)',
      borderWidth: 3,
      showLine: true,
      fill: false,
      pointRadius: 0
    });
  }

  new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
          labels: {
            font: { size: 15, family: 'Segoe UI, Arial' }
          }
        },
        tooltip: {
          backgroundColor: 'rgba(30,30,30,0.9)',
          titleColor: '#fff',
          bodyColor: '#fff'
        }
      },
      layout: {
        padding: 20
      },
      scales: {
        x: {
          title: { display: true, text: 'x', font: { size: 16 } },
          grid: { color: 'rgba(200,200,200,0.2)' }
        },
        y: {
          title: { display: true, text: 'y', font: { size: 16 } },
          grid: { color: 'rgba(200,200,200,0.2)' }
        }
      }
    }
  });
}

async function trainModel(xTrain, yTrain, xTest, yTest, label, canvasTrainId, canvasTestId, predictX) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
  model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });

  const xsTrain = tf.tensor2d(xTrain, [xTrain.length, 1]);
  const ysTrain = tf.tensor2d(yTrain, [yTrain.length, 1]);
  const xsTest = tf.tensor2d(xTest, [xTest.length, 1]);
  const ysTest = tf.tensor2d(yTest, [yTest.length, 1]);

  await model.fit(xsTrain, ysTrain, {
    epochs: label === 'Over-Fit' ? 1000 : label === 'Best-Fit' ? 100 : 200,
    batchSize: 32,
    verbose: 0
  });

  const trainLoss = model.evaluate(xsTrain, ysTrain).dataSync()[0];
  const testLoss = model.evaluate(xsTest, ysTest).dataSync()[0];
  console.log(`${label} – Train Loss: ${trainLoss.toFixed(4)}, Test Loss: ${testLoss.toFixed(4)}`);

  const xPred = predictX.sort((a, b) => a - b);
  const yPredTensor = model.predict(tf.tensor2d(xPred, [xPred.length, 1]));
  const yPred = Array.from(yPredTensor.dataSync());
  const predictionPoints = xPred.map((x, i) => [x, yPred[i]]);

  plotData(canvasTrainId, xTrain, yTrain, [], [], predictionPoints);
  plotData(canvasTestId, [], [], xTest, yTest, predictionPoints);

  const containerTrain = document.getElementById(canvasTrainId).parentElement;
  const containerTest = document.getElementById(canvasTestId).parentElement;

 const lossInfoTrain = document.createElement('p');
  lossInfoTrain.textContent = `Train Loss: ${trainLoss.toFixed(4)}`;
  lossInfoTrain.style.marginTop = '0.5rem';
  lossInfoTrain.style.fontWeight = 'bold';
  lossInfoTrain.style.color = '#007bff';
  lossInfoTrain.style.fontFamily = 'Segoe UI, Arial';
  lossInfoTrain.style.textAlign='center';
  containerTrain.appendChild(lossInfoTrain);

  const lossInfoTest = document.createElement('p');
  lossInfoTest.textContent = `Test Loss: ${testLoss.toFixed(4)}`;
  lossInfoTest.style.marginTop = '0.5rem';
  lossInfoTest.style.fontWeight = 'bold';
  lossInfoTest.style.color = '#e83e8c';
  lossInfoTest.style.fontFamily = 'Segoe UI, Arial';
  lossInfoTest.style.textAlign = 'center';
  containerTest.appendChild(lossInfoTest);
}

const xVals = generateXValues();
const yClean = calculateY(xVals);
const yNoisy = addGaussianNoise(yClean);

const cleanSplit = splitTrainTest(xVals, yClean);
const noisySplit = splitTrainTest(xVals, yNoisy);

plotData("plotClean", cleanSplit.xTrain, cleanSplit.yTrain, cleanSplit.xTest, cleanSplit.yTest);
plotData("plotNoisy", noisySplit.xTrain, noisySplit.yTrain, noisySplit.xTest, noisySplit.yTest);

trainModel(cleanSplit.xTrain, cleanSplit.yTrain, cleanSplit.xTest, cleanSplit.yTest,
  'Unverrauscht', 'predUnverrauschtTrain', 'predUnverrauschtTest', xVals);

trainModel(noisySplit.xTrain, noisySplit.yTrain, noisySplit.xTest, noisySplit.yTest,
  'Best-Fit', 'predBestFitTrain', 'predBestFitTest', xVals);

trainModel(noisySplit.xTrain, noisySplit.yTrain, noisySplit.xTest, noisySplit.yTest,
  'Over-Fit', 'predOverFitTrain', 'predOverFitTest', xVals);