# OUTPUT
bash```
LSTMModel(
  (lstm): LSTM(1, 64, num_layers=2, batch_first=True, dropout=0.2)
  (fc1): Linear(in_features=64, out_features=32, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=32, out_features=1, bias=True)
)
Epoch [1/30], Train Loss: 0.222681, Test Loss: 0.742928
Epoch [2/30], Train Loss: 0.208642, Test Loss: 0.713822
Epoch [3/30], Train Loss: 0.194985, Test Loss: 0.685280
Epoch [4/30], Train Loss: 0.181894, Test Loss: 0.653384
Epoch [5/30], Train Loss: 0.167828, Test Loss: 0.619749
Epoch [6/30], Train Loss: 0.153496, Test Loss: 0.585081
Epoch [7/30], Train Loss: 0.138975, Test Loss: 0.549691
Epoch [8/30], Train Loss: 0.124978, Test Loss: 0.514010
Epoch [9/30], Train Loss: 0.111706, Test Loss: 0.479404
Epoch [10/30], Train Loss: 0.099339, Test Loss: 0.442178
Epoch [11/30], Train Loss: 0.087170, Test Loss: 0.399210
Epoch [12/30], Train Loss: 0.074234, Test Loss: 0.349226
Epoch [13/30], Train Loss: 0.061098, Test Loss: 0.291426
Epoch [14/30], Train Loss: 0.049026, Test Loss: 0.225839
Epoch [15/30], Train Loss: 0.040502, Test Loss: 0.157192
Epoch [16/30], Train Loss: 0.040167, Test Loss: 0.104471
Epoch [17/30], Train Loss: 0.049791, Test Loss: 0.085690
Epoch [18/30], Train Loss: 0.056364, Test Loss: 0.088405
Epoch [19/30], Train Loss: 0.054657, Test Loss: 0.103606
Epoch [20/30], Train Loss: 0.048540, Test Loss: 0.126137
Epoch [21/30], Train Loss: 0.042494, Test Loss: 0.151534
Epoch [22/30], Train Loss: 0.038815, Test Loss: 0.176129
Epoch [23/30], Train Loss: 0.037620, Test Loss: 0.197503
Epoch [24/30], Train Loss: 0.037633, Test Loss: 0.214473
Epoch [25/30], Train Loss: 0.038647, Test Loss: 0.226730
Epoch [26/30], Train Loss: 0.039617, Test Loss: 0.234416
Epoch [27/30], Train Loss: 0.040472, Test Loss: 0.237859
Epoch [28/30], Train Loss: 0.040724, Test Loss: 0.237429
Epoch [29/30], Train Loss: 0.040632, Test Loss: 0.233517
Epoch [30/30], Train Loss: 0.040001, Test Loss: 0.226507

===== PyTorch LSTM Model Evaluation =====
RMSE: 126.83
MAE: 124.44
MAPE: 51.42%
=========================================





===== Future Forecast =====
            Forecasted_Close
2026-05-20        119.151569
2026-05-21        117.896462
2026-05-22        116.227014
2026-05-25        114.513186
2026-05-26        112.935041
2026-05-27        111.562803
2026-05-28        110.410260
2026-05-29        109.464005
2026-06-01        108.699275
2026-06-02        108.088241
2026-06-03        107.604138
2026-06-04        107.223091
2026-06-05        106.924674
2026-06-08        106.691891
2026-06-09        106.510922
2026-06-10        106.370569
2026-06-11        106.261968
2026-06-12        106.178084
2026-06-15        106.113403
2026-06-16        106.063559
2026-06-17        106.025198
2026-06-18        105.995701
2026-06-19        105.973034
2026-06-22        105.955641
2026-06-23        105.942283
2026-06-24        105.932021
2026-06-25        105.924159
2026-06-26        105.918139
2026-06-29        105.913516
2026-06-30        105.909974
===========================

===== Final Conclusion =====

This project used a PyTorch LSTM deep learning model to analyze and forecast AAPL stock prices.

The model was trained using the previous 60 trading days to predict the next closing price.

Model performance:
- RMSE: 126.83
- MAE: 124.44
- MAPE: 51.42%

The latest actual closing price was $298.97.
The forecasted price after 30 business days is approximately $105.91.

Based on the PyTorch LSTM forecast, the short-term trend appears to be negative / downward.

However, this is only an educational model. Stock prices are influenced by many external factors such as news, earnings reports, interest rates, macroeconomic conditions, and investor sentiment.
Therefore, this result should not be considered financial advice.
```
