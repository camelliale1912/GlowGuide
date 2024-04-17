from numpy.core.fromnumeric import prod
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Import the Dataset 
skincare = pd.read_csv("processed_data.csv", encoding='utf-8', index_col=None)

# Header
st.set_page_config(page_title="Skin Care Product Recommendation App", page_icon=":blossom:", layout="wide",)

# Displaying the homepage
st.title("Skin Care Product Recommendation App :sparkles:")

st.write('---') 

# Displaying a local video file
video_file = open("skincare.mp4", "rb").read()
st.video(video_file, start_time = 1) # Displaying the video 

st.write('---') 

st.write(
    """
    ##### **The Skin Care Product Recommendation App is an implementation of a Machine Learning project that provides recommendations for skincare products based on your skin type and concerns. You can input your skin type, concerns, and desired benefits to get the right skincare product recommendations.**
    """)  
st.write('---') 

first,last = st.columns(2)

# Choose a product product type category
# pt = product type
category = first.selectbox(label='Product Category : ', options= skincare['product_type'].unique() )
category_pt = skincare[skincare['product_type'] == category]

# Choose a skin type
# st = skin type
skin_type = last.selectbox(label='Your Skin Type : ', options= ['Normal', 'Dry', 'Oily', 'Combination'] )
category_st_pt = category_pt[category_pt[skin_type] == 1]

# Skin Problems
# prob = st.multiselect(label='Skin Problems : ', options= ['Pores', 'Blemishes', 'Oiliness', 'Dryness', 'Dullness', 'Loss of Firmness and Elasticity', 'Dark Spots', 'Uneven Texture', 'Fine Lines and Wrinkles', 'Acne and Blemishes', 'Dark Circles', 'Puffiness'] )

# Choose notable_effects
# From the products that have been filtered based on product type and skin type (category_st_pt), we will extract the unique values in the notable_effects column.
opsi_ne = category_st_pt['notable_effects'].unique().tolist()
# The unique notable_effects values are then placed into the variable opsi_ne and used for values in the multiselect wrapped in the variable selected_options below.
selected_options = st.multiselect('Notable Effects : ',opsi_ne)
# The results from selected_options are placed into the variable category_ne_st_pt.
category_ne_st_pt = category_st_pt[category_st_pt["notable_effects"].isin(selected_options)]

# Choose product
# Products that have been filtered and are in the variable filtered_df are then refined to extract unique values based on product_name and placed into the variable opsi_pn.
opsi_pn = category_ne_st_pt['product_name'].unique().tolist()
# Create a select box containing the filtered product options above
product = st.selectbox(label='Recommended Products for You', options = sorted(opsi_pn))
# The variable product above will contain a product that will trigger the display of other recommended products

## MODELLING with Content Based Filtering
# Initialization of TfidfVectorizer
tf = TfidfVectorizer()

skincare['notable_effects'].fillna('', inplace=True)

# Performing IDF calculation on the 'notable_effects' data
tf.fit(skincare['notable_effects']) 

# Mapping array from integer index features to feature names
tf.get_feature_names_out()

# Mapping array from feature index integer to feature name
tfidf_matrix = tf.fit_transform(skincare['notable_effects']) 

# Viewing the size of the TF-IDF matrix.
shape = tfidf_matrix.shape

# Converting the TF-IDF vector into matrix form using the todense() function.
tfidf_matrix.todense()

# Creating a DataFrame to view the TF-IDF matrix
# Columns are filled with the desired effects
# Rows are filled with product names
pd.DataFrame(
    tfidf_matrix.todense(), 
    columns=tf.get_feature_names_out(),
    index=skincare.product_name
).sample(shape[1], axis=1).sample(10, axis=0)

# Calculating cosine similarity on the TF-IDF matrix
cosine_sim = cosine_similarity(tfidf_matrix) 

# Creating a DataFrame from the cosine_sim variable with rows and columns representing product names
cosine_sim_df = pd.DataFrame(cosine_sim, index=skincare['product_name'], columns=skincare['product_name'])

# Viewing the similarity matrix for each product name
cosine_sim_df.sample(7, axis=1).sample(10, axis=0)

# Creating a function to obtain recommendations
def skincare_recommendations(product_namee, similarity_data=cosine_sim_df, items=skincare[['product_name', 'brand', 'description']], k=5):

    # Fetching data using argpartition to perform an indirect partition along the given axis
    # The DataFrame is converted to a NumPy array
    # Range(start, stop, step)
    index = similarity_data.loc[:,product_namee].to_numpy().argpartition(range(-1, -k, -1))

    # Fetching data with the highest similarity from the available index
    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    # Dropping the product name so that the searched product name does not appear in the list of recommendations
    closest = closest.drop(product_namee, errors='ignore')
    df = pd.DataFrame(closest).merge(items).head(k)
    return df

# Creating a button to display recommendations
model_run = st.button('Find More Product Recommendations!')
# Getting recommendations
if model_run:
    st.write('Here are Other Similar Product Recommendations as You Desire')
    st.write(skincare_recommendations(product))
