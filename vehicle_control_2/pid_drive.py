#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MetaDrive Steering Control with PID Controller
This script implements PID steering control for a vehicle in MetaDrive environment
and visualizes the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from config import CONFIG
from utils import generate_gif, get_heading_diff
from metadrive.envs.metadrive_env import MetaDriveEnv


class PIDController:
    """
    A class that implements a Proportional-Integral-Derivative controller.
    """
    def __init__(self, kp=1.0, ki=0.1, kd=0.05):
        """
        Initialize the PID controller with gain parameters.
        
        Args:
            kp (float): Proportional gain
            ki (float): Integral gain
            kd (float): Derivative gain
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
    
    def compute(self, setpoint, measurement, dt):
        """
        Compute the control signal based on the error.
        
        Args:
            setpoint (float): Desired value
            measurement (float): Current measured value
            dt (float): Time step in seconds
            
        Returns:
            float: Control signal
        """
        # Calculate error
        error = setpoint - measurement
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup (limit the integral to prevent excessive accumulation)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)  # Anti-windup
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = self.kd * (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        
        # Total control signal
        control = p_term + i_term + d_term
        
        # Clip the control signal to the expected range for steering [-1, 1]
        control = np.clip(control, -1.0, 1.0)
        
        return control


def main():
    """
    Main function that runs the MetaDrive simulation with PID steering control.
    """
    # Create output directory for plots
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the environment
    env = MetaDriveEnv(CONFIG)
    
    # Initialize PID controller for steering
    # These gains may need tuning based on the specific scenario
    controller = PIDController(kp=0.8, ki=0.1, kd=0.2)
    
    # Fixed throttle value as specified
    FIXED_THROTTLE = 0.05
    
    # Data collection for visualization
    frames = []
    heading_errors = []
    steering_commands = []
    reference_headings = []
    actual_headings = []
    time_steps = []
    
    # Simulation parameters
    dt = 0.1  # Assumed time step for the environment
    max_steps = 2000
    
    # Reset the environment
    obs, info = env.reset()
    
    # Main simulation loop
    for i in tqdm(range(max_steps), desc="PID Steering Control", unit="steps"):
        # Calculate heading error (wrapped to [-pi, pi])
        heading_error, ref_line_heading, current_heading = get_heading_diff(env)
        
        # Compute steering command using PID controller
        steering = controller.compute(0, heading_error, dt)  # setpoint is 0 as we want zero error
        
        # Create action with computed steering and fixed throttle
        action = np.array([steering, FIXED_THROTTLE])
        
        # Apply action to the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render and record frame
        frame = env.render(mode="topdown", screen_record=True, window=False)
        frames.append(frame)
        
        # Collect data for visualization
        heading_errors.append(heading_error)
        steering_commands.append(steering)
        reference_headings.append(ref_line_heading)
        actual_headings.append(current_heading)
        time_steps.append(i * dt)
        
        # Reset if episode is terminated
        if terminated or truncated:
            print(f"Episode finished at step {i}")
            break
            obs, info = env.reset()
            controller.integral = 0.0  # Reset integral term
            controller.prev_error = 0.0  # Reset previous error
            
    
    # Close the environment
    env.close()
    
    print(f"Test drive successful! -- Number of time steps: {len(frames)}")
    print("Rendering gif...", end="\t")
    generate_gif(frames)
    print("Done!")
    
    # Plot heading error and steering commands
    plot_results(time_steps, heading_errors, steering_commands, 
                 reference_headings, actual_headings, output_dir)


def plot_results(time_steps, heading_errors, steering_commands, 
                reference_headings, actual_headings, output_dir):
    """
    Plot the results of the simulation.
    
    Args:
        time_steps (list): Time steps
        heading_errors (list): Heading errors at each time step
        steering_commands (list): Steering commands at each time step
        reference_headings (list): Reference headings at each time step
        actual_headings (list): Actual headings at each time step
        output_dir (str): Directory to save the plots
    """
    # Convert to numpy arrays for easier manipulation
    time_steps = np.array(time_steps)
    heading_errors = np.array(heading_errors)
    steering_commands = np.array(steering_commands)
    reference_headings = np.array(reference_headings)
    actual_headings = np.array(actual_headings)
    
    # Plot 1: Heading error over time
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, heading_errors, 'r-', label='Heading Error')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Heading Error (rad)')
    plt.title('Heading Error over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'heading_error.png'))
    
    # Plot 2: Steering commands over time
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, steering_commands, 'b-', label='Steering Command')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Command')
    plt.title('Steering Commands over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'steering_commands.png'))
    
    # Plot 3: Reference vs Actual Heading
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, reference_headings, 'g-', label='Reference Heading')
    plt.plot(time_steps, actual_headings, 'b-', label='Actual Heading')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Heading (rad)')
    plt.title('Reference vs Actual Heading')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'heading_comparison.png'))
    
    # Plot 4: Combined plot
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, reference_headings, 'g-', label='Reference Heading')
    plt.plot(time_steps, actual_headings, 'b-', label='Actual Heading')
    plt.grid(True)
    plt.ylabel('Heading (rad)')
    plt.title('Reference vs Actual Heading')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, heading_errors, 'r-', label='Heading Error')
    plt.grid(True)
    plt.ylabel('Error (rad)')
    plt.title('Heading Error')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, steering_commands, 'b-', label='Steering Command')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Command')
    plt.title('PID Control Output')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_plots.png'))


if __name__ == "__main__":
    main()