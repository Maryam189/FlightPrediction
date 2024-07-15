

from flask import Flask, render_template, request


# Read in the datasets
import numpy as np
import pandas as pd
import random
from math import radians, cos, sin, asin, sqrt
from datetime import datetime

# Read in the datasets
# Note: These file paths are placeholders. Replace them with the actual paths of your CSV files.
weather_df = pd.read_csv('weatherDataset.csv')
aircraft_df = pd.read_csv('aircraftDataset.csv')
airports_df = pd.read_csv('airportDataset.csv')

def preprocess_data(df):
    # Replace missing values with mean for numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].mean(), inplace=True)

    # Replace missing values with mode (most frequent value) for categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

# Preprocessing each dataset
weather_df = preprocess_data(weather_df)
aircraft_df = preprocess_data(aircraft_df)
airports_df = preprocess_data(airports_df)



    
def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    return c * r

def get_intermediate_cities(airports_df, start_city, end_city, max_distance=300):
    start_coords = airports_df.loc[airports_df['City'] == start_city, ['Latitude', 'Longitude']].values[0]
    end_coords = airports_df.loc[airports_df['City'] == end_city, ['Latitude', 'Longitude']].values[0]

    intermediate_cities = []
    for index, row in airports_df.iterrows():
        city = row['City']
        if city not in [start_city, end_city]:
            distance_to_start = haversine(start_coords[1], start_coords[0], row['Longitude'], row['Latitude'])
            distance_to_end = haversine(end_coords[1], end_coords[0], row['Longitude'], row['Latitude'])
            if distance_to_start <= max_distance and distance_to_end <= max_distance:
                intermediate_cities.append(city)

    return intermediate_cities

def calculate_wind_effect(weather, start_city, end_city, airport_df):
    default_wind_speed = 10  # Example default value
    default_wind_direction = 180  # Example default direction

    wind_speed = weather['Wind Speed'] if pd.notnull(weather['Wind Speed']) else default_wind_speed
    wind_direction = weather['Wind Direction'] if pd.notnull(weather['Wind Direction']) else default_wind_direction


    # Check if airport data is available for both start and end cities
    if start_city not in airport_df['City'].values or end_city not in airport_df['City'].values:
        print(f"Missing airport data for {start_city} or {end_city}.")
        return float('nan')

    start_coords = airport_df[airport_df['City'] == start_city][['Latitude', 'Longitude']].values[0]
    end_coords = airport_df[airport_df['City'] == end_city][['Latitude', 'Longitude']].values[0]

    # Calculate the direction of the flight
    dy = end_coords[0] - start_coords[0]
    dx = end_coords[1] - start_coords[1]
    flight_direction = np.arctan2(dy, dx) * (180 / np.pi)

    # Correcting negative angles
    flight_direction = flight_direction if flight_direction >= 0 else 360 + flight_direction

    # Calculate the wind's effect on the flight
    wind_angle = (wind_direction - flight_direction + 360) % 360
    wind_effect = wind_speed * np.cos(np.radians(wind_angle))  # Positive for tailwind, negative for headwind

    # Debug print
    print(f"Wind Effect for {start_city} to {end_city}: Wind Speed = {wind_speed}, Wind Direction = {wind_direction}, Flight Direction = {flight_direction}, Wind Effect = {wind_effect}")
    if np.isnan(wind_effect) or wind_effect == float('inf'):
        print(f"Wind effect calculation error for {start_city} to {end_city}.")
        return float('nan')
    
    wind_effect = np.clip(wind_effect, -50, 50)  # Example: cap between -50 and 50

    return wind_effect
    

def calculate_fuel_consumption(aircraft_data, wind_effect):
    # Extract aircraft performance data
    cruise_speed = aircraft_data['Cruise Speed']  # In knots
    fuel_consumption_at_cruise = aircraft_data['Fuel Consumption at Cruise']  # In lbs/nm
    
    # Check for invalid data
    if cruise_speed <= 0 or fuel_consumption_at_cruise <= 0:
        return float('nan')

    # Adjust cruise speed for wind_effect (tailwind increases speed, headwind decreases)
    adjusted_speed = cruise_speed + wind_effect
    
    # Avoid unrealistic speed adjustments
    if adjusted_speed <= 0:
        return float('nan')
    
    # Calculate fuel consumption
    fuel_consumption = fuel_consumption_at_cruise * adjusted_speed
    
    fuel_consumption = max(fuel_consumption, 0)

    return fuel_consumption

def calculate_fitness(route, aircraft_code, weather_df, aircraft_df, airports_df):
    wind_effect_threshold = 20  # Example value, e.g., 20 knots
    wind_penalty_factor = 50   # Example value, e.g., 100 units of penalty

    if len(route) < 3:  # Ensure there are enough cities in the route
        return float('inf')

    total_fuel_consumption = 0
    wind_penalty = 0

    for i in range(len(route) - 1):
        start_city = route[i]
        end_city = route[i + 1]

        # Ensure all required data is available
        if start_city not in weather_df['City'].values or end_city not in airports_df['City'].values:
            return float('inf')

        weather = weather_df[weather_df['City'] == start_city].iloc[0]
        aircraft_data = aircraft_df[aircraft_df['ICAO CODES'] == aircraft_code].iloc[0]
        wind_effect = calculate_wind_effect(weather, start_city, end_city, airports_df)

        # Check for valid wind effect
        if np.isnan(wind_effect):
            return float('inf')

        # Penalize routes with strong headwinds or high crosswinds
        if abs(wind_effect) > wind_effect_threshold:
            wind_penalty += wind_penalty_factor

        fuel_consumption = calculate_fuel_consumption(aircraft_data, wind_effect)

        # Check for valid fuel consumption
        if np.isnan(fuel_consumption):
            return float('inf')

        total_fuel_consumption += fuel_consumption

    # Combine total fuel consumption and wind penalties to calculate the fitness score
    fitness_score = total_fuel_consumption + wind_penalty
    # Normalize the fitness score
    fitness_score = (total_fuel_consumption + wind_penalty) / len(route)

    return fitness_score


def get_intermediate_cities(airports_df, start_city, end_city):
    # Extract latitude and longitude for start and end cities
    start_coords = airports_df.loc[airports_df['City'] == start_city, ['Latitude', 'Longitude']].values[0]
    end_coords = airports_df.loc[airports_df['City'] == end_city, ['Latitude', 'Longitude']].values[0]

    # Function to check if a city is between start and end
    def is_between(lat, lon, start_coords, end_coords):
        return (min(start_coords[0], end_coords[0]) <= lat <= max(start_coords[0], end_coords[0])) and \
               (min(start_coords[1], end_coords[1]) <= lon <= max(start_coords[1], end_coords[1]))

    # Filter cities based on the above condition
    intermediate_cities = airports_df[airports_df.apply(lambda row: is_between(row['Latitude'], row['Longitude'], start_coords, end_coords), axis=1)]

    # Return the list of cities
    return intermediate_cities['City'].tolist()

def create_initial_population(population_size, cities, start_city, end_city):
    population = []
    for _ in range(population_size):
        intermediate_cities = [city for city in cities if city not in [start_city, end_city]]
        random_route = random.sample(intermediate_cities, len(intermediate_cities))
        route = [start_city] + random_route + [end_city]
        population.append(route)
    return population

def select_parents(population, fitness_scores, num_parents):
    parents = []
    for _ in range(num_parents):
        best_fitness_idx = np.argmin(fitness_scores)
        parents.append(population[best_fitness_idx])
        fitness_scores[best_fitness_idx] = np.inf  # Set high value to not select it again
    return parents

def crossover(parent1, parent2, start_city, end_city):
    child = [start_city]  # Start with the start city
    
    # Choose a segment from the middle of one parent
    start_idx = int(random.random() * (len(parent1) - 2)) + 1
    end_idx = int(random.random() * (len(parent1) - 2)) + 1
    start_gene, end_gene = min(start_idx, end_idx), max(start_idx, end_idx)
    middle_segment = parent1[start_gene:end_gene]
    
    # Fill the rest with genes from the other parent, excluding the start and end cities
    child += [city for city in parent2 if city not in middle_segment and city not in [start_city, end_city]]
    child.append(end_city)  # End with the end city
    
    return child

def mutate(route, mutation_rate, start_city, end_city):
    for i in range(1, len(route) - 1):  # Avoid mutating the start and end cities
        if random.random() < mutation_rate:
            swap_with = int(random.random() * (len(route) - 2)) + 1
            
            # Swap cities
            route[i], route[swap_with] = route[swap_with], route[i]
    return route
def create_next_generation(population, fitness_scores, num_parents, mutation_rate):
    # Sort the population with the best fitness at the top
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0])]
    # Select the top num_parents routes
    parents = sorted_population[:num_parents]
    # Generate children from parents (crossover and mutation)
    children = []
    for _ in range(len(population) - num_parents):
        parent1, parent2 = random.sample(parents, 2)
        child = crossover(parent1, parent2)  # You need to define this function
        child = mutate(child, mutation_rate)  # You need to define this function
        children.append(child)
    # Create the new generation
    next_generation = parents + children
    return next_generation

def genetic_algorithm(start_city, end_city, aircraft_code, population_size, num_generations, mutation_rate):
    intermediate_cities = get_intermediate_cities(airports_df, start_city, end_city)
    population = create_initial_population(population_size, intermediate_cities, start_city, end_city)

    for generation in range(num_generations):
        fitness_scores = [calculate_fitness(route, aircraft_code, weather_df, aircraft_df, airports_df) for route in population]
        
        # Select parents for the next generation
        parents = select_parents(population, fitness_scores, int(population_size / 2))

        # Create children through crossover and mutation
        children = []
        for _ in range(len(population) - len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2, start_city, end_city)
            child = mutate(child, mutation_rate, start_city, end_city)
            children.append(child)

        # Combine parents and children to create the next generation
        population = parents + children

        # Find and print the best route in the current generation
        best_route_idx = np.argmin(fitness_scores)
        best_fitness = fitness_scores[best_route_idx]
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    # After all generations, return the best route found
    best_route_idx = np.argmin(fitness_scores)
    return population[best_route_idx]


def display_route_details(best_route, aircraft_code, weather_df, aircraft_df, airports_df):
    for i in range(len(best_route) - 1):
        start_city = best_route[i]
        end_city = best_route[i + 1]
        
        weather = weather_df[weather_df['City'] == start_city].iloc[0]
        aircraft_data = aircraft_df[aircraft_df['ICAO CODES'] == aircraft_code].iloc[0]

        wind_effect = calculate_wind_effect(weather, start_city, end_city, airports_df)
        fuel_consumption = calculate_fuel_consumption(aircraft_data, wind_effect)

        print(f"Leg: {start_city} to {end_city}, Wind Effect: {wind_effect}, Fuel Consumption: {fuel_consumption}")

from datetime import datetime

def validate_date(date, dataset):
    try:
        # Convert user input date to datetime object
        date = datetime.strptime(date, '%d-%m-%Y')
        print("User Input Date:", date)  # Debug print

        # If the dataset 'Date' column is not already in datetime format, convert it
        if not pd.api.types.is_datetime64_any_dtype(dataset['Date']):
            dataset['Date'] = pd.to_datetime(dataset['Date'])

        min_date = dataset['Date'].min()
        max_date = dataset['Date'].max()

        print("Dataset Date Range:", min_date, "to", max_date)  # Debug print
        
        if min_date <= date <= max_date:
            return True
        else:
            print(f"Entered date is not within the valid range ({min_date.strftime('%d-%m-%Y')} to {max_date.strftime('%d-%m-%Y')})")
            return False
    except ValueError as e:
        print("Invalid date format. Please use DD-MM-YYYY format.")
        print("Error:", e)  # Debug print
        return False





app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # Check if the entered date is valid for the weather dataset
    # Exit the program if the date is not valid

    start_city = request.form['start_city']
    end_city = request.form['end_city']
    date = request.form['date']
    aircraft_code = 'MULT'  # Example ICAO code
    # Rest of your code (from the genetic_algorithm function onwards)

    if not validate_date(date, weather_df):
        print("Program terminated due to invalid date.")
        exit()   # 
    population_size = 50
    num_generations = 100
    mutation_rate = 0.01
    tournament_size = 5 
    best_route = genetic_algorithm(start_city, end_city, aircraft_code, population_size, num_generations, mutation_rate)
    print(f"Best Route: {best_route}")

    # Displaying detailed route information
    # display_route_details(best_route, aircraft_code, weather_df, aircraft_df, airports_df)

    return render_template('result.html', best_route=best_route)

if __name__ == '__main__':
    app.run(debug=True)



# Displaying detailed route information
#display_route_details(best_route, aircraft_code, weather_df, aircraft_df, airports_df)


