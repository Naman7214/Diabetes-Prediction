from flask import Flask, request, render_template
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained model
model = load_model('diabetes_model.h5')

# Load the MinMaxScaler for 'BMI'
scaler = MinMaxScaler()

# Load the training data used for scaling
# X_train = pd.read_csv("my_dataframe.csv")
# scaler.fit(X_train[['BMI']])

def preprocess_input(data):
    # Convert binary categorical features from "Yes"/"No" to 1/0
    binary_features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
                        'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk']
    for feature in binary_features:
        data[feature] = 1 if data[feature] == 'Yes' else 0

    data['Sex'] = 1 if data['Sex'].lower() == 'male' else 0

    gen_hlth_mapping = {'excellent': 1, 'very good': 2, 'good': 3, 'fair': 4, 'poor': 5}
    data['GenHlth'] = gen_hlth_mapping.get(data['GenHlth'].lower(), 0)

    numeric_features = ['BMI', 'MentHlth', 'PhysHlth', 'Age']
    for feature in numeric_features:
        data[feature] = pd.to_numeric(data[feature], errors='coerce')

    data_df = pd.DataFrame([data])

    data_df.fillna(0, inplace=True)

    return data_df


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.to_dict()

    input_data = preprocess_input(user_input)
    predictions = model.predict(input_data)

    predictions = (predictions >= 0.5).astype(int)
    return render_template('result.html', prediction=predictions[0][0])

if __name__ == '__main__':
    app.run(debug=True)
