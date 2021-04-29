import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
import base64
from model_functions import predict
from metric_functions import misclassification_score
from datetime import datetime
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


##### utility functions #####
############ DOWNLOAD THE CSV WITH PREDICTIONS #################
# http://awesome-streamlit.org/
# https://discuss.streamlit.io/t/file-download-workaround-added-to-awesome-streamlit-org/1244

def download_csv(data):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    return href

############# DISPLAY BUTTON TYPE #################\
def display_button(operation_type='Prediction'):
    '''
        This function displays the button based on the operation mode of the model.
    '''
    if operation_type=='Prediction':
        button_name = 'Make Prediction(s)'
    else:
        button_name = 'Start Evaluation'
    # display the button
    button = st.button(button_name)
    return button

#### UPLOAD THE CSV FILE CONTAINING THE DATAPOINTS ####
# https://www.youtube.com/watch?v=Uh_2F6ENjHs&ab_channel=soumilshah1995
def uploader():
    '''
        Function for uploading of the csv file containing the datapoints.
    '''
    upload = st.file_uploader('', type=['csv'])
    return upload

#### SHOW THE CONFUSION, PRECISION AND RECALL MATRICES ON THE STREAMLIT APP ####
def calculate_matrices(y_true, y_pred):
    '''
        This function calculates the elements of the confusion, precision and recall matrices.
    '''
    # confusion matrix
    confusion = confusion_matrix(y_true, y_pred)
    # precision matrix - column sum of confusion matrix
    precision = confusion/confusion.sum(axis=0)
    # recall matrix - row sum of confusion matrix
    recall = (confusion.T/confusion.sum(axis=1)).T

    return (confusion, precision, recall)

def show_matrix(y_true, y_pred, matrix='Confusion'):
    '''
        This function plots the confusion, precision and recall matrices in the streamlit
        webapp.
    '''
    sns.set_style('dark')
    fig, ax = plt.subplots(figsize=(6,6))
    # get the matrices
    confusion, precision, recall = calculate_matrices(y_true, y_pred)
    if matrix == 'Confusion':
        hmap = sns.heatmap(confusion, cbar=False, annot=True, cmap="YlGnBu", fmt='g', ax=ax)
    elif matrix == 'Precision':
        hmap = sns.heatmap(precision, cbar=False, annot=True, cmap="YlGnBu", fmt='.3g', ax=ax)
    elif matrix == 'Recall':
        hmap = sns.heatmap(recall, annot=True, cbar=False, cmap="YlGnBu", fmt='.3g', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    return fig

###############################################################################################################################################
                                        # ACTUAL STREAMLIT PAGE #
###############################################################################################################################################

st.set_page_config(layout='wide') # set layout wide by default

# title
# set header font size in streamlit - https://discuss.streamlit.io/t/change-font-size-in-st-write/7606
st.markdown("""
<style>
.big-font {
    font-size:50px !important;
    font-family: georgia;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Scania APS Fault Prediction</p>', unsafe_allow_html=True)

# image
st.image('https://c4.wallpaperflare.com/wallpaper/678/895/709/2016-crown-r520-scania-wallpaper-preview.jpg',
        caption='Source: https://c4.wallpaperflare.com/wallpaper/678/895/709/2016-crown-r520-scania-wallpaper-preview.jpg',
        width=1000)

# introduction
st.markdown("""
            The _ APS _ (Air Pressure System) of a heavy haul truck is a crucial part of any
            heavy haul truck which needs to be functioning propely at all times to ensure the
            smooth operation of any truck. The APS of a truck provides the required air pressure
            in the brakes of the truck for them to be disengaged. This pressure needs to be consistently
            provided to the brake shoes of all the wheels of the trucks. If any components of the APS 
            has any problem in it, this air pressure gets compromised which leads to the brackes being
            immediately engaged which causes the truck to immediately come to a stop and making it unable to
            move untill the pressure required is again introduced to the brakes. This makes the proper functioning
            of the APS all the more important.<br>
            This is why, a machine learning model has been developed and deployed to predict whether a problem
            in the truck corresponds to the APS or to any other part/mechanism of the truck.
""", unsafe_allow_html=True)

st.markdown('''<hr>''', unsafe_allow_html=True)

# modelling and prediction
# 3 modes - 
#       1. Multiple datapoints - take i/p file loc and o/p file location and save o/p file
#       2. Single datapoint - TODO - maybe take i/p file location and display class label
#       3. Preformance metrics - Take i/p file, predict and calculate performance metrics and display

st.header('''** MODELLING ** ''')

st.markdown('''There are 2 different operation modes.<br>
            1. **Predictions:-** Make predictions for datapoints.
            For this, the user needs to upload a csv file which containing all the datapoints 
            and all the default features. The model will then predict the labels for all the datapoints
            and will write them on a csv file which can be downloaded by the user.<br>
            2. **Performance Evaluation:-** The user will upload a csv file with already present class labels
            and the model after making predictions will calculate and print performance metrics on the screen.
            The metrics that will be printed are as follows:
            
                a. Confusion, Precision and Recall matrices.
                b. Macro-averaged F1 score
                c. Misclassification cost: 
                    Formula:- 10*FP + 500*FN
''', unsafe_allow_html=True)

# select box for mode selection
operation = st.radio('Operation mode:',
    options=('Prediction', 'Peformance evaluation'),
    index=0
)

################ UPLOADING THE REQUIRED DATAFRAME AND GIVE COMMAND FOR MAKING PREDICTIONS ###############
# Appropriate headers and displays based on the operation mode
if operation=='Prediction':
    st.header('Making Predictions')
    st.markdown('''
    Please upload the csv file containing the datapoints for which the prediction is to be made.
    ''')
    st.markdown('NOTE: If there are less than or equal to 10 rows in the csv file, then the predictions will be displayed on the screen along with the features. Else, a csv file link will be generated that the user will have to download to look at the results. The file will contain the class labels for each datapoint.')

    # show the upload area
    upload = uploader()
    # reading the csv and showing the top row
    if upload:
        csv = pd.read_csv(upload, na_values='na')
        st.markdown('Top few rows of the dataframe...')
        st.dataframe(csv.head())
        button = display_button()

elif operation=='Peformance evaluation':
    st.markdown('''
        <h2>Performance Evaluation of the model.</h2>
        ''', unsafe_allow_html=True)
    st.markdown('''
        Please upload the csv file containing the datpoints and the actual class labels for 
        evaluation of the model performance.
    ''')
    
    # showing the upload dialogue
    upload = uploader()
    # reading the csv and showing the csv file
    if upload:
        csv = pd.read_csv(upload, na_values='na')
        Y = csv['class']
        st.markdown('''Top few rows of the dataframe...''')
        st.dataframe(csv.head())

        # showing the button after the upload is completed
        button = display_button('Evaluation')


###################### MAKE PREDICTIONS OR EVALUATE PERFORMANCE BASED ON THE OPERATION MODE ##############
if upload and button: # if any file has been uploaded and the button has been pressed
    if operation=='Prediction':
        # we need to predict and show the results or add to a csv file and the user will download it.
        # making the predictions
        start = datetime.now()
        predictions = predict(csv)
        print('Time required:', datetime.now()-start)
        # now, if there are less than or equal to 10 datapoints, we are going to display the dataframe on the screen
        if predictions.shape[0] <= 10:
            st.dataframe(predictions)
            # href = download_csv(predictions)
            # st.markdown(href, unsafe_allow_html=True)
        else: # if there are more rows, then generate a link to download the csv file.
            href = download_csv(predictions['class'])
            st.markdown(href, unsafe_allow_html=True)
            # st.dataframe(predictions.head(100))
    if operation=='Peformance evaluation':
        st.markdown('''
            <h2> Model Performance metrics:
        ''', unsafe_allow_html=True)
        # we need to make predictions and display the scores
        predictions = predict(csv)['class']
        # misclassification score
        ms = misclassification_score(Y, predictions)
        # f1 score
        f1 = f1_score(Y, predictions, average='macro')
        # matrices
        st.markdown('''
            <h3>Confusion, Precision and Recall matrices...
        ''', unsafe_allow_html=True)
        conf = show_matrix(Y, predictions)
        pre = show_matrix(Y, predictions, 'Precision')
        re = show_matrix(Y, predictions, "Recall")

        # making columns and plotting heatmaps
        col1, col2, col3 = st.beta_columns(3)
        col1.header("Confusion Matrix")
        col1.pyplot(conf)
        col2.header("Precision Matrix")
        col2.pyplot(pre)
        col3.header('Recall Matrix')
        col3.pyplot(re)

        # displaying f1 score and misclassification cost
        col1, col2 = st.beta_columns(2)
        col1.header("Macro - F1 Score")
        col1.write(round(f1, 3))
        col2.header('Misclassification Cost')
        col2.write(ms)