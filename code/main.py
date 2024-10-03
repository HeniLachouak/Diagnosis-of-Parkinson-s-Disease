import pandas as pd
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, BaggingClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
import gradio as gr
from tensorflow.keras.preprocessing import image

# Load dataset
data = pd.read_csv('../data/ReplicatedAcousticFeatures-ParkinsonDataset.csv')

# Split dataset
x = data.drop(['Status'], axis=1)
y = data['Status']

# Define models
model1 = lgb.LGBMClassifier()
model2 = XGBClassifier(eta=0.001, gamma=0, n_estimators=94)
model3 = GradientBoostingClassifier()
model4 = BaggingClassifier(XGBClassifier(eta=0.001, gamma=0, n_estimators=94))

# Voting classifier with weighted ensemble
model = VotingClassifier(estimators=[('lgb', model1), ('xgb', model2), ('gb', model3), ('bc', model4)], 
                         weights=(6, 13, 6, 1), voting='hard')

# Train the model
model.fit(x, y)

# Prediction function
def pred(csv_file):
    dataset = pd.read_csv(csv_file.name, delimiter=',')
    dataset.fillna(0, inplace=True)
    X = dataset.iloc[:, :-1]
    prediction = model.predict(X)

    if prediction == 1:
        return image.load_img('../data/positive_image.jpg'), "This patient has Parkinson's Disease"
    else:
        return image.load_img('../data/negative_image.jpg'), "There is no sign of disease in this patient"

# Gradio interface
with gr.Blocks(css="#img0, #img1 {background:#0B0F19}") as app:
    gr.Markdown(
    """
    # Diagnosis of Parkinson's Disease
    ‏‏‎ ‎
    """)
    with gr.Row() as row:
        with gr.Column():
            img1 = gr.Image('../data/ribbon_image.svg', show_label=False, visible=False)
        with gr.Column():
            img2 = gr.Image('../data/ribbon_image.svg', show_label=False, visible=False)

    csv_input = gr.File(label="Upload CSV")
    result_image = gr.Image(show_label=False)
    diagnosis_output = gr.Textbox(label="Diagnosis")

    # Button to trigger the prediction
    submit_btn = gr.Button("Submit")

    # Prediction event
    submit_btn.click(fn=pred, inputs=csv_input, outputs=[result_image, diagnosis_output])

# Launch app
app.launch()
