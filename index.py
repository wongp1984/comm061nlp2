from flask import Flask, render_template, request, redirect, url_for, abort
from datetime import datetime, date

app = Flask(__name__)

########################################
# for logging the user inputs
def LogActivity(input_time, user_input, prediction_time, predict_result):
    current = datetime.now()
    fname = 'action_log' + current.strftime('%Y%m%d')
    with open(fname, 'a') as fp:
        fp.write(f"'{input_time}', '{user_input}', '{prediction_time}', '{predict_result}'\n")
    
########################################

@app.route("/", methods=["GET", "POST"])
def predict_emotion():
    if request.method == "GET":
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

