Bike sharing client trend forcasting
====
Simple ridge regression model that forecasts usage trends of a bike sharing service. Forcasting can be made on an hourly and daily basis. Forcasting quality is checked against 10% of the data, which has not been used for cross-validation.

### Goals
- build a forecasting model for hourly and daily usages of bike sharing service
- explore data for insights

### Data
For this project a [Bike Sharing Dataset] from UCI was used. Please, see the corresponding description of data quality and format.

### Tools and dependencies
- python `3.4.3`
- requests `2.11.1`
- pandas `0.18.1`
- numpy `1.11.1`
- matplotlib `1.5.2`
- scikit-learn `0.17.1`

### How to run
```
# example usage on linux
$python3.4 data-scientist.py -f ./data/hour.csv
```

### Results and insights
- After checking the scatter plot matrix several findings were made:
    - the data is a non-stationary time series subjected to seasonality and a general increasing trend, with a sharp locally decreasing trend towards the end of 2012
    - there is a noticeable correlation between apparent temperature and both registered and casual users, which seem to contribute equally to both groups
    - naturally, there are more users during good weather days as compared to poor weather ones
- It seems only wise to use time series analysis (TSA) modelling approach and decompose the data into stationary, trend and seasonality and perform analysis and modelling from that point (e.g. using ARIMA model). However, for starters it was decided to use a simple ridge regression model for the non-linear data set:
    - some features were dropped (i.e. `humidity`, `temperature`, etc.) since they were implicitly present and combined in `apparent temperature` feature
    - no dimensionality reduction methods were tested for the sake of simplicity
    - categorical and most time related features were transformed using one-hot-encoding approach due to an explicit bias and misleading information (e.g. a distance measure of 1h and 23h would suggest that the values are quite far apart, however they are merely 2h away)
    -  predicted values were not scaled back, however in a real world usage scenario such operation is obviously required

[Bike Sharing Dataset]: <https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset>