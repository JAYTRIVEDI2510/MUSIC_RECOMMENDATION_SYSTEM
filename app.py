import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle
import requests

with open('data_reduce.pkl', 'rb') as file:
    data_reduce = pickle.load(file)

with open('nn_model.pkl', 'rb') as file:
    nn_model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer= pickle.load(file)





def recom(input_value):
    # Determine the column to filter based on the input
    filter_columns = ["artists", "track_name", "album_name","track_genre"]

    tfidf_matrix = tfidf_vectorizer.fit_transform(data_reduce['features'])
    
    nn_model.fit(tfidf_matrix)

    for filter_column in filter_columns:
        filtered_data = data_reduce[data_reduce[filter_column] == input_value]
        if not filtered_data.empty:
            # Get the index of the first entry for the given input
            input_index = filtered_data.index[0]

            # Use Nearest Neighbors to find similar songs
            _, indices = nn_model.kneighbors(tfidf_matrix[input_index], n_neighbors=20)

            # include the input song itself
            indices = indices.flatten()[:]

            # Create a set to keep track of unique track names
            unique_track_names = set()

            # Print the recommended unique track names
            for index in indices:
                current_track_name = data_reduce.iloc[index]['track_name']
                if current_track_name not in unique_track_names:
                    print(current_track_name)
                    unique_track_names.add(current_track_name)
            recommendations_df = pd.DataFrame(list(unique_track_names), columns=['Recommended Track Names'])
            return recommendations_df

    

def main():
    st.title('Song Recommendation App🎶🎵')
    select_category = st.selectbox("Select an option",("artists", "album_name", "track_genre",'track_name'))
    st.write("You selected category:", select_category)
    selected_item = st.selectbox(f"Select a {select_category}",data_reduce[select_category].unique() ) # Get unique values for the selected category from the dataset
    
    st.write(f"Similar song recommendation are: {selected_item}")
    recommendations_df = recom(selected_item)
    st.table(recommendations_df)


if __name__=="__main__":
    main()
