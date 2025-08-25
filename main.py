import streamlit as st
import pandas as pd
from os import path
import numpy as np
import pickle   #It is used to read .pkl file

# st.title("Hello World")
# st.write("Good to see you again!")
#
# #creating a dataframe
# df_Data = pd.DataFrame({'Column1': [1,2,3,5],
#                         'Column2': ['a','b','c','d']})
# st.write(df_Data) #displaying the dataframe we created.

# st.title("Iris dataset")
# df_iris = pd.read_csv(path.join("Data","iris.csv"))
# #filepath = C:\Users\janik\Downloads\ICT_DSA\Projects\Streamlit_demo\Data\iris.csv
# st.write(df_iris)
#
# st.scatter_chart(df_iris[["sepal_length","sepal_width"]])
#
# df_map = pd.DataFrame(np.array([[9.00067960192081, 76.69450196725744]]),
#                       columns=["lat","lon"])
# st.map(df_map)

# petal_length = st.slider("petal_length")
# petal_width = st.slider("petal_width")
# sepal_length = st.slider("sepal_length")
# sepal_width = st.slider("sepal_width ")

st.title("Iris Species Predictor")
petal_length = st.number_input("enter the petal_length",
                               placeholder = "enter the value between 1.0 and 6.9 ",
                               min_value= 1.0,max_value = 6.9, value = None)
petal_width = st.number_input("enter the petal_width",placeholder = "enter the value between 0.1 and 2.5",min_value = 0.1,max_value = 2.5, value = None)
sepal_length = st.number_input("enter the sepal_length",placeholder = "enter the value between 4.3 and 7.9",min_value = 4.3,max_value = 7.9, value = None)
sepal_width = st.number_input("enter the sepal_width ",placeholder = "enter the value between 2.0 and 4.4",min_value = 2.0,max_value = 4.4, value = None)

#create the dataframe for prediction
df_user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
    columns = ['sepal_length','sepal_width','petal_length','petal_width'])
#using the .pkl file,creating a model named 'iris_predictor'
model_path = path.join("model", "iris_classifier.pkl")
with open(model_path, 'rb') as file:
    iris_predictor = pickle.load(file)
st.write(df_user_input)
dict_species = {0:'setosa',1:'versicolor',2:'virginica'}


if st.button("Predict Species"):
    if((petal_length == None) or (petal_width == None) or (sepal_length == None) or (sepal_width == None)):
        st.write("Please fill all values") #will be executed when any of the values is not entered properly
    else:
        #Prediction can be done here
        predicted_species = iris_predictor.predict(df_user_input)
        #predicted_species[0] will give us the value in the data frame
        #we use that value to find the corresponding species from the
        #directory 'species'
        st.write("The species is",dict_species[predicted_species[0]])


