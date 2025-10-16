import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import datetime as dt
import warnings
warnings.filterwarnings("ignore")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def prepare_data(data, horizon=1, features=None):
    df = data.copy()
    df['Future'] = df['Close'].shift(-horizon)
    df.dropna(inplace=True)
    if features is None:
        X = df[['Close']]
    else:
        X = df[features]
    y = df['Future']
    return X, y, df

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge_regression(X_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_lasso_regression(X_train, y_train, alpha=0.01):
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2, y_pred

def plot_results(y_test, y_pred, title="Prediction vs Actual"):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label="Actual", linewidth=2)
    plt.plot(y_pred, label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.show()

def predict_next_day(model, X):
    return model.predict(X.iloc[[-1]])[0]

def generate_recommendation(prediction, current):
    return "Kaufen" if prediction > current else "Verkaufen"



ticker = "AAPL"
start_date = "2023-01-01"
end_date = "2024-08-09"


stock_data = get_stock_data(ticker, start_date, end_date)


stock_data['MA5'] = stock_data['Close'].rolling(window=5).mean()
stock_data['MA10'] = stock_data['Close'].rolling(window=10).mean()
stock_data['Return'] = stock_data['Close'].pct_change()
stock_data['Volatility'] = stock_data['Return'].rolling(window=10).std()
stock_data.dropna(inplace=True)

features = ['Close', 'MA5', 'MA10', 'Return', 'Volatility']


X, y, df = prepare_data(stock_data, horizon=1, features=features)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

n
lr_model = train_linear_regression(X_train, y_train)
ridge_model = train_ridge_regression(X_train, y_train, alpha=0.5)
lasso_model = train_lasso_regression(X_train, y_train, alpha=0.001)


mse_lr, mae_lr, r2_lr, y_pred_lr = evaluate_model(lr_model, X_test, y_test)
mse_ridge, mae_ridge, r2_ridge, y_pred_ridge = evaluate_model(ridge_model, X_test, y_test)
mse_lasso, mae_lasso, r2_lasso, y_pred_lasso = evaluate_model(lasso_model, X_test, y_test)

print("Linear Regression -> MSE:", mse_lr, "MAE:", mae_lr, "R2:", r2_lr)
print("Ridge Regression  -> MSE:", mse_ridge, "MAE:", mae_ridge, "R2:", r2_ridge)
print("Lasso Regression  -> MSE:", mse_lasso, "MAE:", mae_lasso, "R2:", r2_lasso)

plot_results(y_test, y_pred_lr, title="Linear Regression Prediction vs Actual")
plot_results(y_test, y_pred_ridge, title="Ridge Regression Prediction vs Actual")
plot_results(y_test, y_pred_lasso, title="Lasso Regression Prediction vs Actual")


current_price = stock_data['Close'].iloc[-1]

next_day_lr = predict_next_day(lr_model, X)
next_day_ridge = predict_next_day(ridge_model, X)
next_day_lasso = predict_next_day(lasso_model, X)

print("Current Price:", current_price)
print("Linear Regression next day prediction:", next_day_lr)
print("Ridge Regression next day prediction:", next_day_ridge)
print("Lasso Regression next day prediction:", next_day_lasso)

print("Empfehlung (Linear Regression):", generate_recommendation(next_day_lr, current_price))
print("Empfehlung (Ridge Regression):", generate_recommendation(next_day_ridge, current_price))
print("Empfehlung (Lasso Regression):", generate_recommendation(next_day_lasso, current_price))

poly_features = PolynomialFeatures(degree=2, include_bias=False)
poly_pipeline = Pipeline([("poly", poly_features), ("scaler", StandardScaler()), ("lin_reg", LinearRegression())])
poly_pipeline.fit(X_train, y_train)
y_pred_poly = poly_pipeline.predict(X_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("Polynomial Regression -> MSE:", mse_poly, "MAE:", mae_poly, "R2:", r2_poly)
plot_results(y_test, y_pred_poly, title="Polynomial Regression Prediction vs Actual")

next_day_poly = poly_pipeline.predict(X.iloc[[-1]])[0]
print("Polynomial Regression next day prediction:", next_day_poly)
print("Empfehlung (Polynomial Regression):", generate_recommendation(next_day_poly, current_price))


tscv = TimeSeriesSplit(n_splits=5)
errors = []
for train_index, test_index in tscv.split(X):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
    model_cv = LinearRegression()
    model_cv.fit(X_train_cv, y_train_cv)
    y_pred_cv = model_cv.predict(X_test_cv)
    mse_cv = mean_squared_error(y_test_cv, y_pred_cv)
    errors.append(mse_cv)
print("TimeSeries CV Mean MSE:", np.mean(errors))


scaler_nn = StandardScaler()
X_scaled = scaler_nn.fit_transform(X)
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

nn_model = Sequential([
    Dense(64, input_dim=X_train_nn.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
nn_model.fit(X_train_nn, y_train_nn, epochs=50, batch_size=16, verbose=0)

y_pred_nn = nn_model.predict(X_test_nn).flatten()
mse_nn = mean_squared_error(y_test_nn, y_pred_nn)
mae_nn = mean_absolute_error(y_test_nn, y_pred_nn)
r2_nn = r2_score(y_test_nn, y_pred_nn)

print("Neural Network -> MSE:", mse_nn, "MAE:", mae_nn, "R2:", r2_nn)
plot_results(y_test_nn, y_pred_nn, title="Neural Network Prediction vs Actual")

next_day_nn = nn_model.predict(scaler_nn.transform(X.iloc[[-1]]))[0][0]
print("Neural Network next day prediction:", next_day_nn)
print("Empfehlung (Neural Network):", generate_recommendation(next_day_nn, current_price))
