from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the models from disk
model = joblib.load('MachineLearning\Project\sentiment_model.pkl')
vectorizer = joblib.load('MachineLearning\Project\ectorizer.pkl')
label_encoder = joblib.load('MachineLearning\Project\label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the tweet from the POST request
    data = request.json['tweet']
    
    # Preprocess the tweet and predict sentiment
    vectorized_tweet = vectorizer.transform([data])
    prediction = model.predict(vectorized_tweet)
    sentiment = label_encoder.inverse_transform(prediction)[0]
    
    return jsonify(sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
