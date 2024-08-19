



from flask import Flask, request, jsonify, flash, render_template, redirect, url_for
import io
import pandas as pd
from mlTrain_final import *
from behavior import *
from gemini_pro_Text import *
from flask_cors import CORS



app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/', endpoint='index')
def index():
    return render_template('index.html')



@app.route('/submit', methods=['POST'], endpoint = 'submit')
def submit():
    if 'inputFile' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['inputFile']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        # Extract additional form data
        context = request.form.get('context')
        forecast_period = request.form.get('forecastPeriod')

        # Read the CSV file
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        

        # Ensure the dataframe has the expected structure
        if df.shape[1] != 1:
            return jsonify({"message": "Invalid file format. The file must have one 'value' column."}), 400
        
        
       
        # Rename the column to 'Value'
        df.columns = ['Value']
        print(df)
        #forward fill
        df = df.ffill()


        #-Infer the period---------------------------------------------
        # Determine the frequency of the time series
        inferred_freq = pd.infer_freq(df.index)
        #inferred_freq = 'M'
        if inferred_freq is None:
            return jsonify({"message": "Could not infer frequency of the time series data."}), 400

        print(f"Frequency inferred: {inferred_freq}")   
        #-------------------------------------------------------------

        acf_lag = significant_acf_lag(df)
        lagged_data = create_lagged_features(df, acf_lag)
        mse, mpe = train_partial(lagged_data)
        outsample_forecast = train_all(df, lagged_data, int(forecast_period), inferred_freq, acf_lag)


        #behaviour------------------------------------

        trend_periods = detect_trend_periods(outsample_forecast)

        behaviorRaw = ""
        for period in trend_periods:
            start, end, trend = period
            duration = (end - start).days + 1
        
            temp= f"{trend} trend from {start} to {end}, {duration} days."
            behaviorRaw = behaviorRaw + " " + temp
        #---------------------------------------------

        #Generate explanation -------------------------------
        textResult = explainForecastBehavior(behaviorRaw, context)

        #----------------------------------------------------
        
        #--Returning a response------------------------------------------------------
        # Prepare JSON response
        response_data = {
            "acf_lag": int(acf_lag),
            "mse": float(mse),
            "mpe": float(mpe),
            "outsample_forecast": outsample_forecast.reset_index().to_dict(orient='records'), 
            "text_result": textResult
        }

        return render_template('result.html', result=response_data)
@app.route('/result', endpoint='result')
def result():
    return render_template(result.html)

@app.route('/ask', methods=['POST'], endpoint='ask')
def ask():
    data = request.get_json()
    question = data['question']
    context = data['context']
    text_result = data['text_result']

    answer = answerMessage(question, context, text_result)

    return jsonify({'answer': answer})
  


if __name__ == '__main__':
    app.run(debug=True)