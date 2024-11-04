# CS 506 Final Project: Predicting Housing Prices Using Real Estate Data
- Description: The project aims to predict housing prices based on the size of the house, location, number of rooms, proximity to amenities, and neighborhood characteristics. This will be done using both traditional regression models and more machine learning techniques. Additionally, the project will explore the impact of listing descriptions on prices using sentiment analysis, and segment neighborhoods using clustering methods.
- Goal: Successfully predict the price of a house based on the characteristics mentioned above.
- How to collect data: We plan to collect data from open real estate datasets such as Zillow’s housing dataset or the MLS (Multiple listing services). 
  We will collect: 
  - House price
  - Square Footage
  - Number of bedrooms
  - Year built
  - Neighborhood rating
  - Proximity to services
  - Listing descriptions

To collect this data we will scrape the Zillow Website and/or use an API.
To process the textual data we will use NLP libraries like NLTK or TextBlob.
- Modelling:
  -  We plan to start by modeling the data by fitting a linear model (linear regression) to predict housing prices based on features such as location, size, and number of rooms.
  -  We then will implement a deep learning model using TensorFlow/PyTorch to capture more sophisticated patterns in the data.
  -  We will also use K-Means clustering to segment neighborhoods based on features like house prices and amenities, and then classify new neighborhoods or predict price ranges based on cluster membership.
  - We will also perform sentiment analysis on listing descriptions to determine if the language of the listing affects the price prediction.
- Visualization: We plan to visualize the data using a scatter plot to show the relationship between square footage and price, and a heatmap to illustrate correlations between features. We will add Clustering Visualization to map out neighborhoods and show clusters based on features like price, crime rate, and amenities.
- Test Plan: We plan to use an 80/20 train-test split, where 80% of the data will be used to train the model, and the remaining 20% will be used to test the model's performance. Additionally, we may use cross-validation to further assess the model’s accuracy across different subsets of the data.

https://youtu.be/gFyMkkLTR7I
