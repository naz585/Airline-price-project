import os

from flask import Flask, render_template, request
import pandas as pd
from american_airlines_predictor import AmericanAirlinesPricePredictor
from datetime import date

app = Flask(__name__, template_folder='Templates')
port = int(os.environ.get('PORT', 5000))
# Load predictor and train at app start
predictor = AmericanAirlinesPricePredictor()


@app.before_request
def train_model():
    df = pd.read_csv('JFK_ORD_truncated_16.csv')
    predictor.train(df)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:

            search_date = (request.form['search_date'])
            flight_date = (request.form['flight_date'])
            #seats_remaining = int(request.form['seats_remaining'])
            num_connections = int(request.form['num_connections'])
            duration_minutes = int(request.form['duration_minutes'])

            date1 = pd.to_datetime(search_date, format='mixed')
            date2 = pd.to_datetime(flight_date, format='mixed')

            # Calculate the difference
            difference = date2 - date1

            # Get the number of days
            days_between = difference.days

            prediction_info = predictor.predict(
                days_in_advance=days_between,
                #seats_left=seats_remaining,
                num_connections=num_connections,
                duration=duration_minutes
            )

            def get_price_advice(prediction_info):
                if prediction_info[1] == 'Below Average':
                    advice = "The price is **good** compared to the daily average. It's a great time to book!"
                else:
                    advice = ("The price is **above average** for this day. If you wait longer, prices are very likely "
                              "to go even higher, especially for last-minute bookings.")

                # Extra tip for booking timing
                if days_between <= 3:
                    advice += ("You're booking very close to the travel date—prices usually spike as the date "
                               "approaches.")
                elif days_between >= 15:
                    advice += " Booking early usually gets you the best prices."
                elif days_between <= 7:
                    advice += " Consider booking even earlier to potentially save more."

                return advice

            advice = get_price_advice(prediction_info)
            print("vvvvvvvvvvvv")
            print(prediction_info[0])
            return render_template('result.html', prediction=prediction_info, advice=advice)
        except Exception as e:
            print(e)
            return render_template('index.html', error=str(e))
    return render_template('index.html', error=None)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=port)
