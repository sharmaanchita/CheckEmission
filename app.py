import json
import random

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

MODEL_PATH = "xgb1.h5"


def load_model():
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(MODEL_PATH)

    # Print expected input of the model
    print(xgb_model.get_booster().feature_names)

    return xgb_model


model = load_model()
mappings = json.load(open("mappings.json"))

feature_names = [
    "origin_port",
    "pl",
    "customs_procedures",
    "logistic_hub",
    "customer",
    "units",
    "weight",
    "material_handling",
    "weight_class",
    "product_id",
]


# Gradio predict function
def predict(*args):
    df = pd.DataFrame(columns=feature_names)
    origin_port = int(mappings["origin_port"][args[0]])
    pl = int(mappings["3pl"][args[1]])
    customs_procedures = int(mappings["customs_procedures"][args[2]])
    logistic_hub = int(mappings["logistic_hub"][args[3]])
    customer = int(mappings["customer"][args[4]])
    units = args[6]
    weight = args[7]
    material_handling = args[8]
    weight_class = args[9]

    # Print type of each input with a for loop
    for i, arg in enumerate(args):
        print(f"{feature_names[i]}: {type(arg)}")

    # Create a dataframe with the inputs
    df = df.append(
        {
            "units": units,
            "weight": weight,
            "material_handling": material_handling,
            "weight_class": weight_class,
            "customer": customer,
            "origin_port": origin_port,
            "3pl": pl,
            "customs_procedures": customs_procedures,
            "logistic_hub": logistic_hub,
        },
        ignore_index=True,
    )
    print(df.info())

    # Get prediction
    pred = model.predict(df)[0]
    # Get probability
    prob = model.predict_proba(df)[0][pred]

    # Return prediction and probability
    return pred, prob


df = pd.read_csv("data/dataframefinal.csv", sep=",")

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column():
                origin_port = gr.Dropdown(
                    label="Origin Port",
                    choices=list(df.origin_port.unique()),
                )
                pl = gr.Dropdown(
                    label="Third-party logistics company",
                    choices=list(df["3pl"].unique()),
                )
                customs_procedures = gr.Dropdown(
                    label="Customs Procedures",
                    choices=list(df.customs_procedures.unique()),
                )
                logistic_hub = gr.Dropdown(
                    label="Logistic Hub",
                    choices=list(df.logistic_hub.unique()),
                )
            with gr.Column():
                customer = gr.Dropdown(
                    label="Customer",
                    choices=list(df.customer.unique()),
                )
                product_id = gr.Textbox(
                    label="Product ID",
                )
                units = gr.Slider(
                    label="Units",
                    maximum=1000,
                    step=1,
                )
                weight = gr.Slider(
                    label="Weight",
                    maximum=5000,
                    step=1,
                )
                material_handling = gr.Slider(
                    label="Material Handling",
                    maximum=5,
                    step=1,
                )
                weight_class = gr.Slider(
                    label="Weight Class",
                    maximum=5,
                    step=1,
                )

        with gr.Row():
            predict_btn = gr.Button(value="Predict")
            label = gr.Label()
            predict_btn.click(
                predict,
                inputs=[
                    origin_port,
                    pl,
                    customs_procedures,
                    logistic_hub,
                    customer,
                    product_id,
                    units,
                    weight,
                    material_handling,
                    weight_class,
                ],
                outputs=[label],
            )
demo.launch()
