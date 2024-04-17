import streamlit as st
from streamlit_option_menu import option_menu
from numpy.core.fromnumeric import prod
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Import the Dataset 
skincare = pd.read_csv("processed_data.csv", encoding='utf-8', index_col=None)

# Header
st.set_page_config(page_title="GlowGuide", page_icon = "🌷", layout="wide",)

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 1

# Different styles of the app
def streamlit_menu(example=1):
    if example == 1:
        # 1. As sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Skin Care", "Get Recommendation", "Intro Guide"],  # required
                icons=["house", "stars", "book"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. Horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Skin Care", "Get Recommendation", "Intro Guide"],  # required
            icons=["house", "stars", "book"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. Horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Skin Care", "Get Recommendation", "Intro Guide"],  # required
            icons=["house", "stars", "book"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected


selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Skin Care":
    st.markdown('<h1 style="color:lightpink;">Skin Care Product Recommender', unsafe_allow_html=True)
    st.write('---') 

    st.write(
        """
        ##### **Glow Guide is one implementation of Machine Learning that can provide recommendations for skin care products according to your skin type and issues**
        """)
    
    # Displaying a local video file
    video_file = open("skincare.mp4", "rb").read()
    st.video(video_file, start_time = 1) # Displaying the video 
    
    st.write(' ') 
    st.write(' ')
    st.write(
        """
        ##### You will receive skin care product recommendations from various cosmetic brands tailored to your skin needs. 
        ##### There are many different categories of skin care products with 4 different skin types, as well as notable effects that you want to obtain from the products. This recommendation application is just a system that provides recommendations based on the data you input, not a scientific consultation.
        ##### Please select the Get Recommendation page to start receiving recommendations. Or select the Intro Guide page to view tips and tricks about skin care.
        """)
    
    st.write(
        """
        **Good luck trying it out! :)**
        """)
    
    
    st.info('Credit: Created by Tra Le Nguyen Huong')

if selected == "Get Recommendation":
    st.markdown('<h1 style="color:lightpink;">Let\'s Get Recommendation</h1>', unsafe_allow_html=True)
    
    st.write(
        """
        ##### **To get recommendations, please input your skin type, concerns, and desired benefits to receive suitable skin care product recommendations**
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
    
        
if selected == "Intro Guide":
    st.markdown('<h1 style="color:lightpink;">Skincare Essentials:</h1>', unsafe_allow_html=True)

    # Cleansing & Toning
    st.write(
        """
        ### **Cleansing & Toning:**
        """)
    st.write(
        """
        - **Face Wash & Cleansers:** Start your skincare routine with a gentle cleanser to remove dirt and impurities without stripping the skin's natural oils.
        - **Toners:** Balance and prep your skin with a toner to remove any remaining traces of impurities and restore its pH balance.
        """)

    # Moisturizing
    st.write(
        """
        ### **Moisturizing:**
        """)
    st.write(
        """
        - **Moisturizers:** Hydrate and nourish your skin with moisturizers tailored to your skin type—whether it's dry, oily, normal, or combination.
        - **Lip Balms & Treatments:** Keep your lips soft and hydrated with nourishing lip balms and treatments.
        """)

    # Sun Protection
    st.write(
        """
        ### **Sun Protection:**
        """)
    st.write(
        """
        - **Face Sunscreen:** Protect your skin from harmful UV rays and prevent premature aging by applying sunscreen daily.
        """)

    # Treatment & Masks
    st.write(
        """
        ### **Treatment & Masks:**
        """)
    st.write(
        """
        - **Face Serums:** Target specific skin concerns such as dark spots, acne, or aging with potent serums enriched with active ingredients.
        - **Eye Masks:** Revitalize tired eyes and reduce puffiness with hydrating and soothing eye masks.
        """)

    st.markdown('<h1 style="color:lightpink;">Personal Care Essentials:</h1>', unsafe_allow_html=True)

    # Hand & Body Care
    st.write(
        """
        ### **Hand & Body Care:**
        """)
    st.write(
        """
        - **Hand Sanitizer & Hand Soap:** Keep your hands clean and germ-free with hand sanitizers and hand soaps.
        - **Body Lotions & Body Oils:** Moisturize and nourish your skin with luxurious body lotions and oils.
        """)

    # Haircare & Hair Removal
    st.write(
        """
        ### **Haircare & Hair Removal:**
        """)
    st.write(
        """
        - **Hair Supplements:** Support healthy hair growth and strengthen your locks with hair supplements.
        - **Hair Removal:** Remove unwanted hair safely and effectively with hair removal tools and products.
        """)

    # Wellness & Self-Care
    st.write(
        """
        ### **Wellness & Self-Care:**
        """)
    st.write(
        """
        - **Holistic Wellness:** Incorporate holistic wellness products into your routine to support your overall health and well-being.
        - **Teeth Whitening:** Brighten your smile with teeth whitening products that remove stains and reveal a brighter, whiter smile.
        """)

    st.markdown('<h1 style="color:lightpink;">Beauty & Makeup Essentials:</h1>', unsafe_allow_html=True)

    # Makeup Removal
    st.write(
        """
        ### **Makeup Removal:**
        """)
    st.write(
        """
        - **Makeup Removers:** Gently remove makeup and impurities from your skin with effective makeup removers.
        """)

    # Face & Makeup Prep
    st.write(
        """
        ### **Face & Makeup Prep:**
        """)
    st.write(
        """
        - **Face Primer:** Create a smooth canvas for makeup application and prolong the wear of your makeup with face primers.
        - **Foundation, BB & CC Cream:** Achieve a flawless complexion with foundations, BB creams, and CC creams.
        """)

    # Color Cosmetics
    st.write(
        """
        ### **Color Cosmetics:**
        """)
    st.write(
        """
        - **Lipstick & Lip Gloss:** Enhance your lips with a pop of color using lipsticks and lip glosses available in a variety of shades and finishes.
        - **Blush:** Add a healthy flush of color to your cheeks with blushes that come in powder, cream, and liquid formulations.
        """)
