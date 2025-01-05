import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="templates")

# Load the trained model (make sure the model file is named 'model2.pkl')
model2 = pickle.load(open('model2.pkl', 'rb'))  # Ensure the correct model file is used

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    # Default value for visibility (since it's not inputted by the user)
    Visibility = 0.0539

    if request.method == 'POST':
        # Retrieve user input from the form
        Fat = float(request.form['Item_Fat_Content'])
        Item_type = float(request.form['Item_Type'])
        Location = float(request.form['Outlet_Location_Type'])
        Outlet_type = float(request.form['Outlet_Type'])
        Age = float(request.form['Age_Outlet'])
        Price = float(request.form['Item_MRP'])      

        # Predict sales using the model
        prediction = model2.predict([[Fat, Visibility, Item_type, Price, Location, Outlet_type, Age]])
        output = prediction[0]
        output = "{:.2f}".format(output)

        # Render the result on the web page
        if output == "0.00":
            return render_template('index.html', prediction_text="Sale cannot be predicted due to invalid entries.")
        else:
            return render_template('index.html', prediction_text=f"Predicted Sales: {output}")
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
