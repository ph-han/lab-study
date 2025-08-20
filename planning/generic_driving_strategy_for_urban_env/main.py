import json
from Car import Car
import sim
from driving_planning import generate_random_vehicles, planning


if __name__ == "__main__":
    # is_collision_vehicles(None, None)

    with open('urban_env.json', 'r') as json_file:
        event_json_data = json.load(json_file)

    # Keep only static events from the file
    static_events = {k: v for k, v in event_json_data.items() if v.get('type') == 'static'}

    # Generate random dynamic vehicles
    random_vehicles = generate_random_vehicles(len(static_events), 3)

    # Combine static and dynamic events
    event_json_data = static_events
    event_json_data.update(random_vehicles)

    # Create a list of NPC cars from the generated events
    npcs = []
    if event_json_data:
        for event_key, event_value in event_json_data.items():
            if event_value and event_value.get('type') == 'dynamic':
                npcs.append((Car(event_value["begin_distance"], 0, 0), event_key))

    ego = Car(0, 0, 0)

    sim.simulation(ego, npcs, event_json_data)
