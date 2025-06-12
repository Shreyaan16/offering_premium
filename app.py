from fastapi import FastAPI 
from fastapi.responses import JSONResponse
from typing import Literal , Annotated
from pydantic import BaseModel , Field , computed_field
import pickle
import pandas as pd

with open('model.pkl' , 'rb') as f:
    model = pickle.load(f)


tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri", "Jaipur"
]


app = FastAPI()

class UserInput(BaseModel):
    age: Annotated[int , Field(... , gt = 0 , lt = 100 , description = "Age of the user")]
    weight: Annotated[float , Field(... , gt = 0 , description = "Weight of the user")]
    height: Annotated[float , Field(... , gt = 0 , description = "Height of the user")]
    income_lpa: Annotated[float , Field(... , description = "Income of the user in lpa")]
    smoker: Annotated[bool , Field(... , description = "Is user a smoker")]
    city: Annotated[str ,  Field(... , description = "City of the user")]
    occupation: Annotated[Literal['retired', 'freelancer', 'student', 'government_job',
                                'business_owner', 'unemployed', 'private_job'], Field(..., description='Occupation of the user')]
    
    @computed_field
    @property
    def city_tier(self) -> int:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        else:
            return 3
        
    @computed_field
    @property
    def bmi(self) -> float:
        return self.weight / ((self.height)**2)
    
    @computed_field
    @property
    def risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        elif (self.smoker and self.bmi > 27) or (not self.smoker and self.bmi > 27):
            return "medium"
        else:
            return "low"
        
    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age >=25 and self.age < 45:
            return "adult"
        elif self.age >= 45 and self.age < 60:
            return "middle_aged"
        else:
            return "senior"
        
@app.post('/predict')
def predict_premium(data : UserInput):
    input_df = pd.DataFrame([{
        'bmi' : data.bmi ,
        'age_group' : data.age_group ,
        'risk' : data.risk , 
        'city_tier' : data.city_tier , 
        'income_lpa' : data.income_lpa , 
        'occupation' : data.occupation
    }])

    pred = model.predict(input_df)[0]

    return JSONResponse(status_code = 200 , content = {'premium' : f'The premium to be offered is {pred}'})