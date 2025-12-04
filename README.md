🏠 California House Price Predictor

1. What is the problem?

Real estate companies need to estimate property values quickly to help sellers set prices. Manually appraising every house is slow and expensive.

The Goal: Build a Machine Learning model that can predict the median price of a house in California based on specific features like location, age of the house, and average income of the neighborhood.

2. What data did I use?

I used the California Housing Dataset (1990 census data), which is a standard benchmark dataset in Machine Learning.

Total Samples: 20,640 houses.

Key Features Used:

MedInc: Median Income in the block group.

HouseAge: Median house age in the block group.

AveRooms: Average number of rooms per household.

3. What was the result?

I used a Linear Regression model because it provides good interpretability for pricing trends.

Model Accuracy (R² Score): ~0.51 (This is a baseline score for this simple model).

Performance: The model captures the general trend that higher income and newer houses correlate with higher prices, though there is variance due to the simplicity of using only 3 features.

4. Visuals

Below is the plot of my model's predictions versus the actual prices. The red dashed line represents a perfect prediction. The closer the blue dots are to the red line, the better the model.

How to Run

Install requirements: pip install pandas scikit-learn matplotlib streamlit

Run the app: streamlit run app.py
