import React, { useState } from 'react';
import { Line, LineChart, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import * as tf from '@tensorflow/tfjs';

const SalesForecast = () => {
  const [predictions, setPredictions] = useState([]);
  const [salesData, setSalesData] = useState([]);
  const [model, setModel] = useState(null);
  const [status, setStatus] = useState('');
  const [selectedProduct, setSelectedProduct] = useState('all');
  const [isModelTrained, setIsModelTrained] = useState(false);  

  const targetFileUpload = (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();
  
    reader.onload = (e) => {
      try {
        const text = e.target.result;
        const lines = text.split('\n');
        
        const dataLines = lines.slice(1);
        
        const parsedData = [];
        
        for (let line of dataLines) {
          if (!line.trim()) continue;
          
          const [date, product, quantity] = line.split(',').map(item => item.trim().replace(/"/g, ''));
          
          if (!date?.match(/^\d{4}-\d{2}$/)) continue;
          
          const numQuantity = parseFloat(quantity);
          if (isNaN(numQuantity)) continue;
          
          if (!isNaN(product)) continue;
          
          parsedData.push({
            date: date + '-01', 
            product: product,
            quantity: numQuantity
          });
        }

        if (parsedData.length === 0) {
          throw new Error('No valid data could be parsed from the file');
        }

        setSalesData(parsedData);
        setStatus('Data loaded successfully! You can now train the model.');
        console.log('Parsed data:', parsedData);
      } catch (error) {
        setStatus('Error parsing file: ' + error.message);
      }
    };
  
    reader.readAsText(file);
  };

  const dataProcessing = () => {
    try {
      if (salesData.length === 0) {
        throw new Error('No data available for processing');
      }

      const products = [...new Set(salesData.map(d => d.product))];
      const dates = [...new Set(salesData.map(d => d.date))].sort();

      const encoder = {};
      products.forEach((p, i) => encoder[p] = i / products.length);

      const inputs = salesData.map(d => [
        dates.indexOf(d.date) / dates.length,
        encoder[d.product]
      ]);

      const outputs = salesData.map(d => d.quantity / 1000);

      return {
        inputs: tf.tensor2d(inputs, [inputs.length, 2]),
        outputs: tf.tensor1d(outputs),
        encoder,
        dates,
        products
      };
    } catch (error) {
      console.error('Data processing error:', error);
      setStatus('Error processing data: ' + error.message);
      return null;
    }
  };

  const trainModel = async () => {
    try {
      setStatus('Training started...');
      const processed = dataProcessing();
      if (!processed) return;

      const { inputs, outputs } = processed;

      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [2],  kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
      }));
      model.add(tf.layers.dropout({ rate: 0.2 }));
      model.add(tf.layers.dense({ units: 32, activation: 'relu',  kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
      }));
      model.add(tf.layers.dropout({ rate: 0.2 }));
      model.add(tf.layers.dense({ units: 1 }));

      model.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.adam(0.001)
      });

      await model.fit(inputs, outputs, {
        epochs: 150,
        validationSplit: 0.2,
        batchSize: 32,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            if ((epoch + 1) % 10 === 0) {
              setStatus(`training epoch... ${epoch + 1}/150, loss: ${logs.loss.toFixed(4)}`);
            }
          }
        }
      });

      setModel(model);
      setIsModelTrained(true);  
      predictSales(model);
      setStatus('Training model completed! Predictions of sales generated.');
    } catch (error) {
      setStatus(error.message);
    }
  };

  const predictSales = (trainedModel) => {
    if (!trainedModel) return;
    const processed = dataProcessing();
    if (!processed) return;
    const { dates, encoder, products } = processed;
    const predictions = [];
    const lastDate = new Date(dates[dates.length - 1]);

    products.forEach(product => {
      for (let i = 1; i <= 6; i++) {
        const nextDate = new Date(lastDate);
        nextDate.setMonth(lastDate.getMonth() + i);
        const dateString = nextDate.toISOString().slice(0, 7);

        const input = tf.tensor2d([[
          (dates.length + i) / (dates.length + 6),
          encoder[product]
        ]]);

        const prediction = trainedModel.predict(input);
        const value = prediction.dataSync()[0] * 1000;

        predictions.push({ 
          date: dateString, 
          product, 
          quantity: Math.max(0, Math.round(value)) 
        });
      }
    });

    setPredictions(predictions);
  };

  const getChartData = () => {
    const filteredPredictions = selectedProduct === 'all'
      ? predictions
      : predictions.filter(d => d.product === selectedProduct);

    return filteredPredictions.map(d => ({
      name: d.date,
      predicted: d.quantity,
      product: d.product
    })).sort((a, b) => a.name.localeCompare(b.name));
  };

  const products = [...new Set(salesData.map(d => d.product))];

  return (
    <main className="w-full p-6">
      <div className="max-w-6xl mx-auto">
        <div className="flex flex-col items-center gap-6">
          <h1 className="text-center">My Sales Forecasting Dashboard</h1>
          
          <div className="flex flex-col gap-4 items-center">
            <div className="flex gap-4 items-center">
              <input
                type="file"
                accept=".csv"
                onChange={targetFileUpload}
                className="file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 
                         file:text-sm file:font-semibold file:bg-violet-500/10 
                         file:text-violet-400 hover:file:bg-violet-500/20"
              />
              <button
                onClick={trainModel}
                disabled={salesData.length === 0}
                className={`transition-colors ${
                  salesData.length === 0
                    ? 'opacity-50 cursor-not-allowed'
                    : 'hover:border-violet-500'
                }`}
              >
                Train Model to Generate Forecast
              </button>
            </div>
            <div className="text-sm opacity-60">
              Upload a CSV file
            </div>
          </div>
          
          <div className="text-sm opacity-60">{status}</div>
          
          {salesData.length > 0 && (
            <div className="w-full max-w-md">
              <select
                value={selectedProduct}
                onChange={(e) => setSelectedProduct(e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-transparent border 
                         border-gray-500 focus:border-violet-500 outline-none"
              >
                <option value="all">All Products</option>
                {products.map(product => (
                  <option key={product} value={product}>{product}</option>
                ))}
              </select>
            </div>
          )}
          
          {isModelTrained && predictions.length > 0 && (
            <div className="w-full overflow-x-auto flex justify-center">
              <LineChart 
                width={1000} 
                height={400} 
                data={getChartData()} 
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="name" 
                  stroke="currentColor" 
                  tick={{ fill: 'currentColor' }} 
                />
                <YAxis 
                  stroke="currentColor" 
                  tick={{ fill: 'currentColor' }} 
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1a1a1a',
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: 'rgba(255,255,255,0.87)'
                  }} 
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="predicted" 
                  stroke="#646cff" 
                  name="Predicted Sales" 
                />
              </LineChart>
            </div>
          )}
        </div>
      </div>
    </main>
  );
};

export default SalesForecast;