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
        self.max_steps = int(0.5 * self.size) # maximum steps per episode
        self.collision_radius = 10 # radius for collision detection
        self.n_planes = n_planes # number of planes
        self.planes = []
        self.plane_speed = 5
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
            np.array([0.8 * self.size, 0.5 * self.size]),
            np.array([0.2 * self.size, 0.2 * self.size])
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
        
        # Relative heading to target is already in [-pi, pi]
        discretized.append(np.digitize(state[3], np.linspace(-np.pi, np.pi, bins)) - 1)

        # Distance to closest other plane normalized to [0,1]
        discretized.append(np.digitize(state[4]/max_dist, np.linspace(0, 1, bins)) - 1)

        # Relative heading to closest other plane is already in [-pi, pi]
        discretized.append(np.digitize(state[5], np.linspace(-np.pi, np.pi, bins)) - 1)

        return tuple(discretized)

    # State space representation
    def get_state(self, plane_idx):
        state = []
        plane = self.planes[plane_idx]

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

            # Relative heading of plane w.r.t. nearest landing zone
            rel_heading = bearing_to_target - plane.heading
            rel_heading = (rel_heading + np.pi) % (2 * np.pi) - np.pi # clamp to [-pi, pi]

            # Nearest other plane (for collision avoidance)
            other_dists = []
            for j, other in enumerate(self.planes):
                if plane_idx != j and other.active:
                    other_dists.append(np.linalg.norm(plane.position - other.position))
            nearest_dist = min(other_dists) if other_dists else self.size * 2 # 2 * environment size if no other planes

            # Nearest other plane info (for collision avoidance)
            nearest_dist = self.size * 2 # large initial value
            rel_bearing_other = 0.0      # default (0) if no other plane

            # Loop over other planes
            for other_idx, other_plane in enumerate(self.planes):
                if other_idx != plane_idx and other_plane.active:
                    vec = other_plane.position - plane.position # vector to other plane
                    dist = np.linalg.norm(vec)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        
                        # Relative bearing (angle to other plane, relative to heading of current plane)
                        rel_bearing_other = (np.arctan2(vec[1], vec[0]) - plane.heading + np.pi) % (2 * np.pi) - np.pi  # clamp to [-pi, pi]

            # Add to state: plane position (x and y), distance to landing zone, relative heading to landing zone, distance to nearest other plane, and relative bearing to nearest other plane
            state = [plane.position[0], plane.position[1], dist, rel_heading, nearest_dist, rel_bearing_other]
        else:
            # Inactive plane --> all states 0
            state = [0, 0, 0, 0, 0, 0]
        return np.array(state, dtype=np.float32)

    # Get the state (for DQN algorithm)
    def get_dqn_state(self, plane_idx):
        plane = self.planes[plane_idx]
        if not plane.active:
            return np.zeros(6) # should match number of states
        
        # Landing zone info
        target_pos = min(self.helipad_pos, key=lambda p: np.linalg.norm(plane.position - p))
        dist_to_target = np.linalg.norm(plane.position - target_pos)
        bearing_to_target = np.arctan2(target_pos[1] - plane.position[1], target_pos[0] - plane.position[0])
        rel_heading_target = (bearing_to_target - plane.heading + np.pi) % (2 * np.pi) - np.pi

        # Nearest plane info
        nearest_dist = self.size * 2
        rel_bearing_other = 0.0
        for other_idx, other_plane in enumerate(self.planes):
            if other_idx != plane_idx and other_plane.active:
                vec = other_plane.position - plane.position
                dist = np.linalg.norm(vec)
                if dist < nearest_dist:
                    nearest_dist = dist
                    rel_bearing_other = (np.arctan2(vec[1], vec[0]) - plane.heading + np.pi) % (2 * np.pi) - np.pi # clamp to [-pi, pi]

        return np.array([
            plane.position[0] / self.size, # normalized x
            plane.position[1] / self.size, # normalized y
            dist_to_target / self.size,    # normalized distance to landing zone
            rel_heading_target / np.pi,    # normalized heading to landing zone
            nearest_dist / self.size,      # normalized distance to nearest other plane
            rel_bearing_other / np.pi      # normalized rel bearing to nearest other plane
        ], dtype=np.float32)

    # Reward function
    def get_reward(self, plane_idx):
        reward = 0
        # 1. Time penalty (encourages faster landings)
        time_penalty = -0.01

        plane = self.planes[plane_idx]
        
        # 2. Landing reward (time-discounted, i.e. more reward if landed earlier)
        for helipad_pos in self.helipad_pos:
            if plane.active and self.crossed_helipad(plane, helipad_pos):
                time_factor = 1.0 - (self.current_step / self.max_steps)
                reward += 100.0 * (0.5 + 0.5*time_factor) # 100-50 points based on when they land
                plane.active = False # deactivate plane after landing

        # 3. Collision avoidance penalty (penalize proximity to other planes)
        if plane.active:
            min_other_dist = self.size * 2 # initialize with a large value
            for other_idx, other_plane in enumerate(self.planes):
                if other_idx != plane_idx and other_plane.active:
                    dist = np.linalg.norm(plane.position - other_plane.position)
                    if dist < min_other_dist:
                        min_other_dist = dist

            # Apply penalty if another plane is within 25% of the environment size
            if min_other_dist < 0.25 * self.size:
                proximity_penalty = -10.0 * (1 - (min_other_dist / (0.25 * self.size))) # scales from 0 to -10
                reward += proximity_penalty

        # 4. Collision penalty
        if self.collision:
            reward -= 10.0

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
                side = np.random.choice(['top', 'left', 'right', 'bottom']) # only top and left now (for testing collision avoidance)
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
                    self.planes.append(Plane(pos, heading, self.plane_speed))
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
                dx = 0.1 * self.size * np.cos(plane.heading) # scale length of arrows with environment size
                dy = 0.1 * self.size * np.sin(plane.heading) # scale length of arrows with environment size
                ax.arrow(x, y, dx, dy, head_width=0.05 * self.size, color='blue') # scale head width with environment size

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
