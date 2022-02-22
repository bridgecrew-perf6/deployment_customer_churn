
from flask import Flask, render_template, request 
import joblib

app = Flask(__name__)

#load the model
model = joblib.load('churn_80.pkl')

@app.route('/')
def hello():
    return render_template('landing.html')

@app.route('/data', methods=['post'])
def data():
    SeniorCitizen_Yes = request.form.get('SeniorCitizen_Yes')
    print('SeniorCitizen_Yes', SeniorCitizen_Yes, type(SeniorCitizen_Yes))
    tenure = request.form.get('tenure')
    print('tenure:-',tenure)
    MonthlyCharges = request.form.get('MonthlyCharges')
    print('MonthlyCharges:-',MonthlyCharges)
    TotalCharges = request.form.get('TotalCharges')
    print('TotalCharges:-',TotalCharges)
    gender_Male = request.form.get('gender_Male')
    print('gender_Male:-',gender_Male)
    Partner_Yes = request.form.get('Partner_Yes')
    print('Partner_Yes:-',Partner_Yes)
    Dependents_Yes = request.form.get('Dependents_Yes')
    print('Dependents_Yes:-',Dependents_Yes)
    PhoneService_Yes = request.form.get('PhoneService_Yes')
    print('PhoneService_Yes',PhoneService_Yes)
    MultipleLines_No_phone_service = request.form.get('MultipleLines_No_phone_service')
    print('MultipleLines_No_phone_service',MultipleLines_No_phone_service)
    MultipleLines_Yes = request.form.get('MultipleLines_Yes')
    print('MultipleLines_Yes',MultipleLines_Yes)
    InternetService_Fiber_optic = request.form.get('InternetService_Fiber_optic')
    print('InternetService_Fiber_optic',InternetService_Fiber_optic)
    InternetService_No = request.form.get('InternetService_No')
    print('InternetService_No',InternetService_No)
    OnlineSecurity_No_internet_service = request.form.get('OnlineSecurity_No_internet_service')
    print('OnlineSecurity_No_internet_service',OnlineSecurity_No_internet_service)
    OnlineSecurity_Yes = request.form.get('OnlineSecurity_Yes')
    print('OnlineSecurity_Yes',OnlineSecurity_Yes)
    OnlineBackup_No_internet_service = request.form.get('OnlineBackup_No_internet_service')
    print('OnlineBackup_No_internet_service',OnlineBackup_No_internet_service)
    OnlineBackup_Yes = request.form.get('OnlineBackup_Yes')
    print('OnlineBackup_Yes',OnlineBackup_Yes)
    DeviceProtection_No_internet_service = request.form.get('DeviceProtection_No_internet_service')
    print('DeviceProtection_No_internet_service',DeviceProtection_No_internet_service)
    DeviceProtection_Yes = request.form.get('DeviceProtection_Yes')
    print('DeviceProtection_Yes',DeviceProtection_Yes)
    TechSupport_No_internet_service = request.form.get('TechSupport_No_internet_service')
    print('TechSupport_No_internet_service',TechSupport_No_internet_service)
    TechSupport_Yes = request.form.get('TechSupport_Yes')
    print('TechSupport_Yes',TechSupport_Yes)
    StreamingTV_No_internet_service = request.form.get('StreamingTV_No_internet_service')
    print('StreamingTV_No_internet_service',StreamingTV_No_internet_service)
    StreamingTV_Yes = request.form.get('StreamingTV_Yes')
    print('StreamingTV_Yes',StreamingTV_Yes)
    StreamingMovies_No_internet_service = request.form.get('StreamingMovies_No_internet_service')
    print('StreamingMovies_No_internet_service',StreamingMovies_No_internet_service)
    StreamingMovies_Yes = request.form.get('StreamingMovies_Yes')
    print('StreamingMovies_Yes',StreamingMovies_Yes)
    Contract_One_year = request.form.get('Contract_One_year')
    print('Contract_One_year',Contract_One_year)
    Contract_Two_year = request.form.get('Contract_Two_year')
    print('Contract_Two_year',Contract_Two_year)
    PaperlessBilling_Yes = request.form.get('PaperlessBilling_Yes')
    print('PaperlessBilling_Yes',PaperlessBilling_Yes)
    PaymentMethod_Credit_card_automatic = request.form.get('PaymentMethod_Credit_card_automatic')
    print('PaymentMethod_Credit_card_automatic',PaymentMethod_Credit_card_automatic)
    PaymentMethod_Electronic_check = request.form.get('PaymentMethod_Electronic_check')
    print('PaymentMethod_Electronic_check',PaymentMethod_Electronic_check)
    PaymentMethod_Mailed_check = request.form.get('PaymentMethod_Mailed_check')
    print('PaymentMethod_Mailed_check',PaymentMethod_Mailed_check)

    print('reached here')
    print(type(SeniorCitizen_Yes))
    result = model.predict([[int(SeniorCitizen_Yes),int(tenure),int(MonthlyCharges),int(TotalCharges),int(gender_Male,Partner_Yes),int(Dependents_Yes),int(PhoneService_Yes),int(MultipleLines_No_phone_service),int(MultipleLines_Yes),int(InternetService_Fiber_optic),int(InternetService_No),int(OnlineSecurity_No_internet_service),int(OnlineSecurity_Yes),int(OnlineBackup_No_internet_service),int(OnlineBackup_Yes),int(DeviceProtection_No_internet_service),int(DeviceProtection_Yes),int(TechSupport_No_internet_service),int(TechSupport_Yes),int(StreamingTV_No_internet_service),int(StreamingTV_Yes),int(StreamingMovies_No_internet_service),int(StreamingMovies_Yes),int(Contract_One_year),int(Contract_Two_year),int(PaperlessBilling_Yes),int(PaymentMethod_Credit_card_automatic),int(PaymentMethod_Electronic_check),int(PaymentMethod_Mailed_check)]])


    
    if result[0]==1:
        output = 'Customer will churn'
    else:
        output ='Customer will not churn'
    
    return render_template('result.html', predict = output)

if __name__ == '__main__':
    app.run(debug=True)
    