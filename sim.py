import numpy as np
import matplotlib.pyplot as plt
import time

# Plane class to represent each plane in the simulation
class Plane:
    def __init__(self, position, heading, speed=5.0):
        self.position = np.array(position, dtype=np.float32)
        self.prev_position = np.array(position, dtype=np.float32)
        self.heading = heading
        self.speed = speed
        self.active = True

    def step(self):
        if not self.active:
            return
        self.prev_position = self.position.copy()
        dx = self.speed * np.cos(self.heading) # speed in x direction
        dy = self.speed * np.sin(self.heading) # speed in y direction
        self.position += np.array([dx, dy], dtype=np.float32) # update position based on speed and heading

# Plane simulation environment
class PlaneSim:
    def __init__(self, n_planes=5):
        self.size = 100 # size of the simulation area
        self.current_step = 0 # current step of the simulation
        self.max_steps = 50 # maximum steps per episode
        self.collision_radius = 10 # radius for collision detection
        self.n_planes = n_planes # number of planes
        self.planes = []
        self.landing_status = {} # dictionary to track landing status of planes and their landing zones
        self.collision = False # flag to indicate if a collision has occurred
        self.done = False # flag to indicate if the simulation is done

        # Define runways: position and heading --> tuple with (center position, heading in radians)
        self.runways = [
            (np.array([100, 125]), 0),        # runway facing to the right
            (np.array([150, 75]), np.pi / 4), # another facing diagonally up-right
        ]
        self.runway_length = 50 # length of each runway

        # Define heli pads: position and radius
        self.helipad_pos = [
            np.array([10, 10])
        ]
        self.helipad_radius = 10 # radius of the heli pad

        # Landing triangle parameters
        self.landing_triangle_length = 60
        self.landing_triangle_width = 60
        self.triangle_offset = 30

        # RL Parameters
        self.state_size = 3 * n_planes
        self.action_space = 5 # actions: [none, up, down, left, right]

        # Initialize planes
        np.random.seed(5) # make random choices reproducible
        self.reset()

    # Discretize the state space
    def discretize_state(self, state, bins=10):
        # Initialize discretized state space to normalize and bin each state
        discretized = []
        
        # Plane position (x,y) normalized to [0,1]
        discretized.append(np.digitize(state[0]/self.size, np.linspace(0, 1, bins)) - 1)
        discretized.append(np.digitize(state[1]/self.size, np.linspace(0, 1, bins)) - 1)
        
        # Distance to target normalized to [0,1]
        max_dist = np.sqrt(2 * self.size**2) # maximum possible distance
        discretized.append(np.digitize(state[2]/max_dist, np.linspace(0, 1, bins)) - 1)
        
        # Relative heading is already in [-pi, pi]
        discretized.append(np.digitize(state[3], np.linspace(-np.pi, np.pi, bins)) - 1)
        
        return tuple(discretized)

    # Get the current state (right now unused, instead get_simple_state is used)
    def get_state(self):
        state = []
        for plane in self.planes:
            if plane.active:
                dx = plane.speed * np.cos(plane.heading)
                dy = plane.speed * np.sin(plane.heading)
                state.extend([plane.position[0], plane.position[1], dx, dy])

                # # Add relative position to nearest runway (for planes so not used for now)
                # nearest = min(self.runways, key=lambda r: np.linalg.norm(plane.position - r[0]))
                # rel = nearest[0] - plane.position
                # state.extend(rel.tolist())

                # Add relative position to nearest helipad (for helicopters)
                nearest_helipad = min(self.helipad_pos, key=lambda hp: np.linalg.norm(plane.position - hp))
                rel_helipad = nearest_helipad - plane.position
                state.extend(rel_helipad.tolist())
            else:
                state.extend([0, 0, 0, 0, 0, 0])  # zero everything for inactive
        return np.array(state)

    # Simpler state space representation
    def get_simple_state(self):
        state = []
        for plane in self.planes:
            if plane.active:
                # Choose correct landing zone type
                landing_zones = self.helipad_pos

                # Nearest landing zone position
                target_pos = min(landing_zones, key=lambda p: np.linalg.norm(plane.position - p))

                # Distance to nearest landing zone
                dist = np.linalg.norm(target_pos - plane.position)

                # Bearing from plane to nearest landing zone
                bearing_to_target = np.arctan2(
                    target_pos[1] - plane.position[1],
                    target_pos[0] - plane.position[0]
                )

                # Relative heading of plane w.r.t. nearest landing zone)
                rel_heading = bearing_to_target - plane.heading
                rel_heading = (rel_heading + np.pi) % (2 * np.pi) - np.pi # (clamp to [-pi, pi]

                # Add to state: plane position (x and y), distance, and relative heading
                state.extend([plane.position[0], plane.position[1], dist, rel_heading])
            else:
                # Inactive plane --> all states 0
                state.extend([0, 0, 0, 0])
        return np.array(state, dtype=np.float32)

    # Get the intruder state (relative positions and headings of all other planes) --> not used not, instead get_simple_state is used
    def get_intruder_state(self):
        state = []
        active_planes = [p for p in self.planes if p.active]

        for i, plane in enumerate(self.planes):
            if plane.active:
                count = 0
                for j, other in enumerate(self.planes):
                    if i == j or not other.active:
                        continue

                    vec = other.position - plane.position
                    dist = np.linalg.norm(vec)
                    angle = np.arctan2(vec[1], vec[0])
                    rel_angle = (angle - plane.heading + np.pi) % (2*np.pi) - np.pi
                    heading_diff = (other.heading - plane.heading + np.pi) % (2*np.pi) - np.pi

                    state.extend([dist, rel_angle, heading_diff])
                    count += 1

                # pad if fewer intruders
                while count < self.n_planes - 1:
                    state.extend([0, 0, 0])
                    count += 1
            else:
                # inactive plane
                state.extend([0, 0, 0] * (self.n_planes - 1))

        return np.array(state)

    # Get the reward (not used right now, instead get_simple_reward is used)
    def get_reward(self):
        # Different reward components for logging
        components = {
            'landing': 0,
            'collision': 0,
            'edge': 0,
            'progress': 0
        }
        
        # # Landing rewards (for planes) (not used for now)
        # for i, plane in enumerate(self.planes):
        #     if self.landing_status[i] and self.landing_status[i]["status"] == "redirecting":
        #         components['landing'] += 0.3 # Partial reward for entering landing triangle
        #     if not plane.active and i in self.landing_status:
        #         components['landing'] += 1.0  # Full reward for landing successfully

        # Landing rewards (for helicopters)
        for helipad_pos in self.helipad_pos:
            for plane in self.planes:
                if plane.active and self.crossed_helipad(plane, helipad_pos):
                    components['landing'] += 1.0 # full reward for landing successfully

        # # Tiny progress reward (distance to nearest runway) (for planes so not used for now)
        # for plane in self.planes:
        #     if plane.active:
        #         min_dist = min(np.linalg.norm(plane.position - r[0]) for r in self.runways)
        #         components['progress'] += 0.01 * (self.size - min_dist) / self.size

        # Tiny progress reward (distance to nearest helipad) (for helicopters)
        for plane in self.planes:
            if plane.active:
                min_dist = min(np.linalg.norm(plane.position - helipad) for helipad in self.helipad_pos)
                components['progress'] += (0.5 * (self.size - min_dist) / self.size) / self.max_steps
        
        # Penalties for collisions and edge hits
        for i in range(len(self.planes)):
            for j in range(i+1, len(self.planes)):
                if self.planes[i].active and self.planes[j].active and self.check_collision(self.planes[i], self.planes[j]):
                    components['collision'] -= 1 # severe penalty for collision
                    
            if self.edge_hit_occurred:
                components['edge'] -= 0.5 / self.max_steps # small penalty for hitting the edge

        total = sum(components.values())
        return total, components # also return breakdown of components for logging

    # Simpler reward function
    def get_simple_reward(self, action):
        reward = 0
        time_penalty = -0.01 # encourages faster landings
        
        # 1. Landing reward (time-discounted, i.e. more reward if landed earlier)
        for helipad_pos in self.helipad_pos:
            for plane in self.planes:
                if plane.active and self.crossed_helipad(plane, helipad_pos):
                    time_factor = 1.0 - (self.current_step / self.max_steps)
                    reward += 100.0 * (0.5 + 0.5*time_factor) # 100-50 points based on when they land
                    plane.active = False # deactivate plane after landing

        # 2. Distance-based reward (more reward as planes get closer to landing zone)
        active_count = sum(1 for p in self.planes if p.active)
        for plane in self.planes:
            if plane.active:
                min_dist = min(np.linalg.norm(plane.position - hp) for hp in self.helipad_pos) # distance to nearest helipad
                reward += (1 - (min_dist / self.size)) * (0.5 / max(1, active_count)) # scaled with active plane count

        # 3. Penalize getting too close to edges
        # edge_threshold = 0.1 * self.size
        # for plane in self.planes:
        #     if plane.active:
        #         x, y = plane.position
        #         near_edge = (x < edge_threshold or x > self.size - edge_threshold or
        #                     y < edge_threshold or y > self.size - edge_threshold)
        #         if near_edge:
        #             reward -= 0.2

        # # 4. Collision penalty
        # if self.collision:
        #     reward -= 10.0

        # # 5. Small penalty for unnecessary turns
        # if action != 0:
        #     reward -= 0.05
            
        return reward + time_penalty

    # Initialize the simulation with random planes
    def reset(self):
        self.collision = False
        self.done = False
        self.planes = []
        self.current_step = 0
        
        min_spawn_distance = 0.2 * self.size # minimum distance between any two planes at spawn
        spawn_positions = []

        for _ in range(self.n_planes):
            max_attempts = 100 # prevent infinite loops
            for _ in range(max_attempts):
                side = np.random.choice(['top', 'left', 'right', 'bottom'])
                angle_variation = np.pi / 6 # 30 degrees variation in incoming heading

                # Randomly place planes on one of the four sides
                if side == 'top':
                    pos = [np.random.uniform(0.1*self.size, 0.9*self.size), self.size]
                    base_heading = -np.pi / 2 # mostly downward
                elif side == 'left':
                    pos = [0, np.random.uniform(0.1*self.size, 0.9*self.size)]
                    base_heading = 0 # mostly right
                elif side == 'right':
                    pos = [self.size, np.random.uniform(0.1*self.size, 0.9*self.size)]
                    base_heading = np.pi # mostly left
                else:  # bottom
                    pos = [np.random.uniform(0.1*self.size, 0.9*self.size), 0]
                    base_heading = np.pi / 2 # mostly upward

                # Check distance to existing spawn positions
                if all(np.linalg.norm(np.array(pos) - np.array(existing)) >= min_spawn_distance for existing in spawn_positions):
                    spawn_positions.append(pos)
                    heading = base_heading + np.random.uniform(-angle_variation, angle_variation)
                    self.planes.append(Plane(pos, heading))
                    break
            else:
                raise ValueError("Failed to place plane without collision after 100 attempts.")

    # Check if a PLANE has crossed the landing triangle base (not used for now)
    def crossed_triangle_base(self, plane, runway_pos, runway_heading):
        # Get the direction of the runway
        forward = np.array([np.cos(runway_heading), np.sin(runway_heading)])
        forward /= np.linalg.norm(forward) # normalize

        # Compute the triangle base points
        perp = np.array([-forward[1], forward[0]]) # perpendicular vector to runway (i.e. the base direction)
        half_w = self.landing_triangle_width / 2
        tip_shifted = runway_pos + forward * self.triangle_offset # shift the tip back along the runway
        base_center = tip_shifted - forward * self.landing_triangle_length
        base_left = base_center + perp * half_w # left base point
        base_right = base_center - perp * half_w # right base point

        # Quickly rule out distant planes
        midpoint = (base_left + base_right) / 2
        max_distance = self.landing_triangle_length * 1.5
        if np.linalg.norm(plane.position - midpoint) > max_distance:
            return False # has not crossed the triangle base

        return self.segments_intersect(plane.prev_position, plane.position, base_left, base_right)
    
    # Check if a HELICOPTER has crossed a helipad
    def crossed_helipad(self, plane, helipad_pos):
        return np.linalg.norm(plane.position - helipad_pos) < self.helipad_radius # returns True if the plane is within the helipad radius

    # Check if two line segments intersect (i.e. whether the plane's path has crossed the runway base) (not used for now)
    def segments_intersect(self, p1, p2, q1, q2):
        # Function that checks if the points a -> b -> c form a counter-clockwise turn
        def ccw(a, b, c):
            return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2) # returns True only if the segments p1-p2 and q1-q2 intersect

    # Check if two planes are colliding
    def check_collision(self, p1, p2):
        return np.linalg.norm(p1.position - p2.position) < self.collision_radius

    # Redirect plane if it goes out of bounds
    def redirect_plane(self, plane):
        x, y = plane.position
        margin = 0  # Strict boundary
        bounced = False
        
        # Left/right boundaries (x-axis)
        if x <= margin:
            plane.heading = np.pi - plane.heading # reflect off left edge
            plane.position[0] = margin # clamp position
            bounced = True
        elif x >= self.size - margin:
            plane.heading = np.pi - plane.heading # reflect off right edge
            plane.position[0] = self.size - margin # clamp position
            bounced = True

        # Top/bottom boundaries (y-axis)
        if y <= margin:
            plane.heading = -plane.heading # reflect off bottom
            plane.position[1] = margin # clamp position
            bounced = True
        elif y >= self.size - margin:
            plane.heading = -plane.heading # reflect off top
            plane.position[1] = self.size - margin # clamp position
            bounced = True

        # Keep heading in range [-pi, pi]
        if bounced:
            plane.heading = (plane.heading + np.pi) % (2 * np.pi) - np.pi
    
        return bounced
    
    # Not used for now
    def find_intruder_pairs(self, threshold=60):
        intruder_pairs = []
        for i, p1 in enumerate(self.planes):
            if not p1.active:
                continue
            for j in range(i + 1, len(self.planes)):
                p2 = self.planes[j]
                if not p2.active:
                    continue
                dist = np.linalg.norm(p1.position - p2.position)
                if dist < threshold:
                    intruder_pairs.append((i, j))
        return intruder_pairs

    # Step the simulation (basically update plane positions and check for collisions/landings)
    def step(self):
        # Reset edge hit status (for reward)
        self.edge_hit_occurred = False
        self.current_step += 1

        # Step each plane
        for plane in self.planes:
            plane.step()

        # Check for redirecting planes at edges
        for plane in self.planes:
            if not plane.active:
                continue

            # Edge redirection
            if self.redirect_plane(plane):
                # print(f"Redirected at edge: new heading {np.degrees(plane.heading):.1f} degrees")
                self.edge_hit_occurred = True
                pass

        # Check for landings
        for i, plane in enumerate(self.planes):
            if not plane.active:
                continue
            
            # # Check if plane is in a landing zone (and being redirected to the triangle tip) (not used for now)
            # if self.landing_status[i] and self.landing_status[i]["status"] == "redirecting":
            #     tip = self.landing_status[i]["tip"] # final landing location (tip of triangle)
            #     if np.linalg.norm(plane.position - tip) < 5: # close enough to triangle tip consider landed
            #         print(f"Plane {i} landed at {tip}")
            #         plane.active = False # deactivate plane
            #         continue

            # for runway_center, heading in self.runways:
            #     if self.crossed_triangle_base(plane, runway_center, heading): # plane has crossed the triangle base, hence we redirect it to the triangle tip
            #         forward = np.array([np.cos(heading), np.sin(heading)]) # direction of the runway
            #         tip_shifted = runway_center + forward * self.triangle_offset # shifted triangle tip

            #         print(f"Plane {i} entered landing triangle at {plane.position}")
            #         direction = tip_shifted - plane.position
            #         plane.heading = np.arctan2(direction[1], direction[0]) # set heading towards triangle tip
            #         self.landing_status[i] = {
            #             "status": "redirecting", # set status to redirecting (to triangle tip)
            #             "tip": tip_shifted # store the (shifted) triangle tip as the final landing location
            #         }
            #         break
            
            # Check if helicopter has landed on a helipad
            for helipad_pos in self.helipad_pos:
                if self.crossed_helipad(plane, helipad_pos):
                    print(f"Helicopter {i} landed on helipad at {helipad_pos}")

        # Check if all planes are inactive (i.e. all have landed or crashed)
        self.done = all(not plane.active for plane in self.planes)

        # Collision check
        for i in range(len(self.planes)):
            for j in range(i + 1, len(self.planes)):
                p1, p2 = self.planes[i], self.planes[j]
                if p1.active and p2.active and self.check_collision(p1, p2):
                    print(f"Collision between planes at {p1.position} and {p2.position}")
                    # Deactivate both planes
                    p1.active = False
                    p2.active = False
                    self.collision = True # set collision flag
                    self.done = True # set done flag to True if any plane collides

    # Render the simulation
    def render(self, ax=None):
        # Plot settings
        if ax is None:
            ax = plt.gca()
    
        ax.clear()
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')

        # Draw each runway
        for pos, heading in self.runways:
            length = self.runway_length
            dx = length * np.cos(heading)
            dy = length * np.sin(heading)
            x, y = pos
            ax.plot([x, x + dx], [y, y + dy], color='gray', linewidth=10, zorder=1)

        # Draw planes as arrows with collision circles
        for plane in self.planes:
            if plane.active:
                x, y = plane.position
                dx = 20 * np.cos(plane.heading)
                dy = 20 * np.sin(plane.heading)
                ax.arrow(x, y, dx, dy, head_width=10, color='blue')

                # Draw the collision circle
                collision_circle = plt.Circle(
                    (x, y), 
                    self.collision_radius / 2, 
                    color='red', 
                    fill=False, 
                    linestyle='-', 
                    linewidth=1,
                    zorder=4
                )
                ax.add_patch(collision_circle)

        # # Draw landing triangles (not used for now)
        # for runway_center, heading in self.runways:
        #     # Get runway direction
        #     forward = np.array([np.cos(heading), np.sin(heading)])
        #     forward /= np.linalg.norm(forward) # normalize

        #     # Compute the triangle base points
        #     perp = np.array([-forward[1], forward[0]]) # perpendicular vector to runway (i.e. the base direction)
        #     half_w = self.landing_triangle_width / 2
        #     tip_shifted = runway_center + forward * self.triangle_offset # shift the tip back along the runway
        #     base_center = tip_shifted - forward * self.landing_triangle_length
        #     base_left = base_center + perp * half_w # left base point
        #     base_right = base_center - perp * half_w # right base point

        #     # Draw triangle area
        #     triangle_points = [base_left, tip_shifted, base_right]  # base -> tip -> base
        #     triangle = plt.Polygon(triangle_points, closed=True, color='green', alpha=0.2, zorder=2)
        #     ax.add_patch(triangle)

        #     # Highlight triangle base in yellow
        #     ax.plot([base_left[0], base_right[0]], [base_left[1], base_right[1]],
        #             color='yellow', linewidth=2, zorder=3)
        #     # Draw dashed line from triangle tip to base center
        #     ax.plot([base_center[0], tip_shifted[0]], [base_center[1], tip_shifted[1]], color='black', linestyle='dotted')

        # Draw helipads
        for helipad in self.helipad_pos:
            helipad_circle = plt.Circle(helipad, self.helipad_radius, color='orange', alpha=0.3, zorder=2)
            ax.add_patch(helipad_circle)

        plt.pause(0.01) # pause to update the plot

# Run the simulation
if __name__ == "__main__":
    sim = PlaneSim(n_planes=5)
    plt.ion() # interactive mode on
    for step in range(300): # run for 300 steps
        sim.step() # step the simulation
        sim.render() # render the current state in the plot
        time.sleep(0.05) # small delay for visualization
    plt.ioff() # interactive mode off
