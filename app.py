from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        user_input = request.form["news"]
        vect_input = vectorizer.transform([user_input])
        result = model.predict(vect_input)[0]
        prediction = "Real News" if result == 1 else "Fake News"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
