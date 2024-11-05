# CS 506 Final Project: Predicting Housing Prices Using Real Estate Data
- Description: The project aims to predict housing prices based on the size of the house, location, number of rooms, proximity to amenities, and neighborhood characteristics. This will be done using both traditional regression models and more machine learning techniques. Additionally, the project will explore the impact of listing descriptions on prices using sentiment analysis, and segment neighborhoods using clustering methods.
- Goal:
  - Develop accurate housing price prediction models using both traditional regression
and advanced machine learning techniques
  - Analyze the impact of listing descriptions on house prices through sentiment analysis
  - Identify and categorize neighborhood segments using clustering methods
  - Create an interpretable and practical tool for real estate price estimation

- Dataset Description:
  - The primary dataset consists of Boston housing listings collected from Redfin and stored in
processed_listings.csv. Key features include:
    - Unique listing identifiers
    - Price points
    - Physical characteristics (bedrooms, bathrooms, square footage)
    - Detailed property descriptions
    - Key property details including lot size, year built, and additional amenities
    - Transportation accessibility scores (walking, transit, and biking)
  - Initially, we planned to obtain the data from Zillow, however, Redfin had fewer protections regarding web scraping; no captcha, and less IP limiting, as well as a better page layout to perform the scraping.

- Data Collection
  - To collect the data, we used Selenium WebDriver to scrape the data from Redfin.
  - We first designed the web scraper (scrape.py), where we implemented pagination handling to collect links from all available pages.
  - We stored these links in links.csv with unique indices.
  - Used Firefox WebDriver with appropriate wait times to ensure reliable data collection.
  - Then we designed runner.py to actually scrape the individual information of each lisitng.

- Data Cleaning
  - Numerical features processing
    - Standardized price values by removing non-numeric characters
    - Converted square footage to numerical format
    - Cleaned bedroom and bathroom counts to pure numeric values
    - Extracted numerical scores from walkscore ratings (e.g., "93/100" → 93)
  - Text data processing
    - Parsed key_details field to extract structured information:
      - Property age/year built
      - Lot size
      - Price per square foot
      - Parking information
      - Additional amenities
    - Cleaned and standardized property descriptions
    - Extracted and separated walk, transit, and bike scores from the walkscore text
- Preliminary Visualization
  - Dataset Overview
    - Total number of properties analyzed: 349
    - Average property price: $1,483,740.88
    - Median property price: $899,000.00
    - Average square footage:1,975.33
  - Property Characteristics
    - Average number of bedrooms: 3.1
    - Average number of bathrooms: 2.1
    - Average price per square foot: $837.90
 - Location Metrics:
    - Average Walk Score: 83.7
    - Average Transit Score: 73.7
    - Average Bike Score: 70.4
  
- <img width="731" alt="Screenshot 2024-11-05 at 08 40 31" src="https://github.com/user-attachments/assets/95aff54c-3d81-4c63-9609-26822b0abfa9">
 
  - A right-skewed distribution indicating more properties in the lower price ranges
  - Several price clusters, suggesting distinct market segments
  - Notable outliers in the luxury market segment


- <img width="754" alt="Screenshot 2024-11-05 at 08 42 02" src="https://github.com/user-attachments/assets/9caac7fa-e0b2-494c-9147-5c00249be620">

  - A strong positive correlation between square footage and price
  - Increasing variance in price as square footage increases

- <img width="513" alt="Screenshot 2024-11-05 at 08 42 33" src="https://github.com/user-attachments/assets/526ecd00-bc51-461e-adcc-7498de161944">
  
  - Strong positive correlation between:
    - Square footage and price
    - Number of bathrooms and price
    - Walk score and transit score
  - Moderate correlations between:
    - Number of bedrooms and square footage
    - Walk score and property price

- Current Implementation and Model Development:
  - Our current implementation focuses on creating a robust data processing and analysis pipeline for the Boston housing market dataset. The core of our work centers on developing sophisticated location-based analysis and comprehensive data visualization tools to understand housing price patterns, laying the groundwork for future predictive modeling.

- Data Processing Implementation:
  - Our preprocessing pipeline handles the transformation of raw listing data into analyzable formats. The code first converts all numeric fields (price, square footage, bedrooms, bathrooms) using pandas' to_numeric function with error handling. To ensure data quality, we remove outliers using the 99th percentile threshold for both price and square footage. This is implemented through simple but effective filtering: df = df[df['price'] < df['price'].quantile(0.99)] and df = df[df['sq_ft'] < df['sq_ft'].quantile(0.99)]. Missing values in key fields are dropped to ensure data consistency, though this is something we plan to handle more sophisticatedly in future iterations.
 
- Location Feature Engineering
  - The heart of our current implementation lies in the location extraction and standardization system. We've developed a two-tier approach through the extract_location_improved function that first attempts to find location information in the key_details field using the "New construction:" tag, and if unsuccessful, searches the property description for matches against known Boston neighborhoods. The function handles variations in neighborhood names through a mapping system that standardizes terms like "Southie" to "South Boston" and "JP" to "Jamaica Plain".
 
- Location-Based Analysis
  - The location analysis component of our implementation focuses on understanding price variations across Boston neighborhoods. We calculate and analyze median prices, property counts, and average square footage for each location using pandas' group by functionality. This provides crucial insights into neighborhood-specific market patterns and helps identify price hotspots across the city. The analysis is performed through our analyze_location_stats function, which prints detailed statistics about property distribution and pricing across different neighborhoods.
 
- Next Steps: Our immediate next steps involve implementing robust predictive modeling capabilities. While our current implementation excels at data processing and visualization, we're expanding into a multi-faceted modeling approach. We'll begin with a linear regression baseline for housing price prediction, using features like location, size, and room count, and progressively incorporate more sophisticated models. Natural language processing will be integrated using NLTK or TextBlob to analyze listing descriptions through sentiment analysis, potentially uncovering how listing language influences pricing.
The modeling strategy will evolve to include deep learning implementations using TensorFlow/PyTorch to capture complex patterns, complemented by K-means clustering to
segment neighborhoods based on prices and amenities. Our visualization framework will be extended to include model performance metrics, prediction analysis, scatter plots showing relationships between square footage and price, and heat maps illustrating feature correlations.
We'll also implement cluster visualization to map neighborhoods based on features like price, crime rate, and amenities.
To ensure model reliability, we'll employ an 80/20 train-test split methodology, with cross-validation to assess model accuracy across different data subsets. The clean, structured dataset our current implementation produces will serve as the foundation for these modeling efforts, setting us up for comprehensive predictive analytics capabilities.

- Video presentation: https://youtu.be/gFyMkkLTR7I
