from shiny import App, ui, render
import numpy as np
import pandas as pd

# --- Coefficients and Stats ---
coefs = {
    "const": -1.0573,
    "Overjet(11)": 1.2704,
    "post/ant": -0.7261,
    "U6-6_cusptip": -0.8129,
    "art_pog": -0.3036,
    "U_leeway": -0.3844,
    "class_cat_T2": 0.9187
}

means = {
    "Overjet(11)": 3.176000,
    "post/ant": 0.654154,
    "U6-6_cusptip": 38.553571,
    "art_pog": 85.498626,
    "U_leeway": 2.742571
}

stds = {
    "Overjet(11)": 1.576033,
    "post/ant": 0.035208,
    "U6-6_cusptip": 3.172536,
    "art_pog": 4.377117,
    "U_leeway": 1.263946
}

# --- UI ---
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_numeric("overjet", "Overjet (mm)", 3.0),
        ui.input_numeric("postant", "Post/Ant Ratio", 0.65),
        ui.input_numeric("u66", "U6-6 Width (mm)", 38.5),
        ui.input_numeric("artpog", "Art-Pog (mm)", 85.0),
        ui.input_numeric("uleeway", "Leeway Space (Mx, mm)", 2.7),
        ui.input_select("classcat", "Molar Classification", {"0": "Class I", "1": "Class II"}, selected="1")
    ),
    ui.h4("🔮 Predicted Probability of Class II at T3:"),
    ui.output_text("prob"),

    title="Class II Prediction"  # keep this here — this one *is* a valid named argument
)



# --- Server ---
def server(input, output, session):

    @output
    @render.text
    def prob():
        # Get and scale input values
        raw = {
            "Overjet(11)": input.overjet(),
            "post/ant": input.postant(),
            "U6-6_cusptip": input.u66(),
            "art_pog": input.artpog(),
            "U_leeway": input.uleeway(),
            "class_cat_T2": int(input.classcat())
        }

        scaled = {}
        for key in raw:
            if key == "class_cat_T2":
                scaled[key] = raw[key]
            else:
                scaled[key] = (raw[key] - means[key]) / stds[key]

        # Logistic regression prediction
        logit = coefs["const"] + sum(scaled[k] * coefs[k] for k in scaled)
        prob = 1 / (1 + np.exp(-logit))

        return f"{prob * 100:.2f}%"

    @output
    @render.table
    def scaled():
        raw = {
            "Overjet(11)": input.overjet(),
            "post/ant": input.postant(),
            "U6-6_cusptip": input.u66(),
            "art_pog": input.artpog(),
            "U_leeway": input.uleeway(),
            "class_cat_T2": int(input.classcat())
        }

        scaled = {}
        for key in raw:
            if key == "class_cat_T2":
                scaled[key] = raw[key]
            else:
                scaled[key] = (raw[key] - means[key]) / stds[key]

        df = pd.DataFrame.from_dict(scaled, orient='index', columns=["Scaled Value"])
        df.index.name = "Feature"
        return df.round(3)

# --- App ---



app = App(app_ui, server)

#expose the WSGI server for Gunicorn
server=app.server
