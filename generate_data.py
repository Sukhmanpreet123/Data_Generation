import opensim as osm
import numpy as np
import pandas as pd

def run_single_simulation(mass_val, initial_speed):
    # 1. Build a simple Pendulum Model
    model = osm.Model()
    
    # Add a Body
    pendulum = osm.Body("pendulum", mass_val, osm.Vec3(0, -0.5, 0), osm.Inertia(1,1,1,0,0,0))
    model.addBody(pendulum)
    
    # Add a Pin Joint
    joint = osm.PinJoint("joint", model.getGround(), osm.Vec3(0,0,0), osm.Vec3(0,0,0), 
                         pendulum, osm.Vec3(0,0.5,0), osm.Vec3(0,0,0))
    model.addJoint(joint)
    
    # 2. Initialize the system
    state = model.initSystem()
    
    # Set the starting position (90 degrees) and the random Initial Speed
    joint.getCoordinate().setValue(state, 1.57) 
    joint.getCoordinate().setSpeedValue(state, initial_speed)
    
    manager = osm.Manager(model)
    state.setTime(0)
    
    # 3. Run simulation for 1.0 second
    manager.initialize(state)
    state = manager.integrate(1.0)
    
    # 4. Return Final Position (Target)
    return joint.getCoordinate().getValue(state)

# Loop to generate 1000 samples
results = []
print("Starting 1000 simulations...")

for i in range(1000):
    m = np.random.uniform(0.5, 5.0)     # Varying Mass
    v = np.random.uniform(-10.0, 10.0)  # Varying Initial Speed
    
    final_pos = run_single_simulation(m, v)
    results.append({'mass': m, 'initial_speed': v, 'final_position': final_pos})
    
    if (i+1) % 100 == 0:
        print(f"Progress: {i+1}/1000 complete.")

# Save the dataset
df = pd.DataFrame(results)
df.to_csv('opensim_data.csv', index=False)
print("Success! Dataset saved as 'opensim_data.csv'")