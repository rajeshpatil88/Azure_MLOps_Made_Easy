from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "<center><h1> rajeshpatil- app deployed using azure webservice <center><h1>"

if __name__ == '__main__':
    app.run(debug = True)