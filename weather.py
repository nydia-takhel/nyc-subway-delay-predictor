import requests


API_KEY = "62714f710ce14a2fbd033f8709815872"
CITY = "New York"

def get_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
    
    response = requests.get(url)
    data = response.json()

    weather_data = {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "visibility": data.get("visibility", 0),
        "weather_main": data["weather"][0]["main"]
    }

    return weather_data


# test run
if __name__ == "__main__":
    print(get_weather())