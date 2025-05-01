#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Steering Control Project with PID Controller
This script simulates a basic steering control system with PID controller
and visualizes the system response over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

class SteeringSystem:
    """
    A class that represents a simple steering system with dynamics.
    """
    def __init__(self, inertia=1.0, damping=0.5):
        """
        Initialize the steering system with physical parameters.
        
        Args:
            inertia (float): Moment of inertia of the steering system
            damping (float): Damping coefficient
        """
        self.inertia = inertia
        self.damping = damping
        self.angle = 0.0        # Current steering angle
        self.angular_vel = 0.0  # Current angular velocity
    
    def update(self, torque, dt):
        """
        Update the steering system state given an input torque.
        
        Args:
            torque (float): Input torque applied to the steering system
            dt (float): Time step in seconds
            
        Returns:
            float: Current steering angle
        """
        # Simple physics model: Torque = Inertia * angular_acceleration + damping * angular_velocity
        angular_acc = (torque - self.damping * self.angular_vel) / self.inertia
        
        # Update angular velocity using acceleration
        self.angular_vel += angular_acc * dt
        
        # Update steering angle using velocity
        self.angle += self.angular_vel * dt
        
        # Add some constraints to the steering angle (typical for vehicles)
        self.angle = np.clip(self.angle, -np.pi/4, np.pi/4)  # Limit to +/- 45 degrees
        
        return self.angle


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
        self.integral = np.clip(self.integral, -10.0, 10.0)  # Anti-windup
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = self.kd * (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        
        # Total control signal
        control = p_term + i_term + d_term
        
        return control


def visualize_steering_wheel(angle, fig, ax):
    """
    Visualize a simple steering wheel with the given angle.
    
    Args:
        angle (float): Current steering angle in radians
        fig: matplotlib figure
        ax: matplotlib axis
    """
    ax.clear()
    
    # Set the limits and aspect ratio
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    
    # Draw steering wheel
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    
    # Draw spokes
    num_spokes = 3
    for i in range(num_spokes):
        spoke_angle = angle + i * (2 * np.pi / num_spokes)
        x = np.cos(spoke_angle)
        y = np.sin(spoke_angle)
        ax.plot([0, x], [0, y], 'k-', linewidth=2)
    
    # Draw a dot indicating the reference direction
    ax.plot([0, 0], [0, 1.2], 'ro', markersize=5)
    
    ax.set_title(f'Steering Wheel (Angle: {np.degrees(angle):.1f}°)')
    ax.axis('off')


def main():
    """
    Main function that runs the simulation, applies PID control, and visualizes the results.
    """
    # Simulation parameters
    dt = 0.01  # Time step (seconds)
    total_time = 10.0  # Total simulation time (seconds)
    
    # Initialize the steering system and controller
    steering = SteeringSystem(inertia=2.0, damping=1.0)
    controller = PIDController(kp=20.0, ki=5.0, kd=7.0)
    
    # Prepare arrays for storing simulation data
    num_steps = int(total_time / dt)
    time_points = np.linspace(0, total_time, num_steps)
    setpoints = np.zeros(num_steps)
    actual_angles = np.zeros(num_steps)
    control_signals = np.zeros(num_steps)
    
    # Create step inputs at different times
    setpoints[int(0.5/dt):int(3.5/dt)] = np.radians(30)  # 30 degrees at t=0.5s
    setpoints[int(3.5/dt):int(7.0/dt)] = np.radians(-20)  # -20 degrees at t=3.5s
    setpoints[int(7.0/dt):] = np.radians(10)  # 10 degrees at t=7.0s
    
    # Run the simulation
    print("Running simulation...")
    for i in range(num_steps):
        # Get setpoint for current time step
        setpoint = setpoints[i]
        
        # Compute control signal using PID controller
        torque = controller.compute(setpoint, steering.angle, dt)
        
        # Apply control to the steering system
        actual_angle = steering.update(torque, dt)
        
        # Store data
        actual_angles[i] = actual_angle
        control_signals[i] = torque
    
    # Visualize time-state results
    plt.figure(figsize=(12, 8))
    
    # Plot steering angle
    plt.subplot(2, 1, 1)
    plt.plot(time_points, np.degrees(setpoints), 'r--', label='Setpoint (degrees)')
    plt.plot(time_points, np.degrees(actual_angles), 'b-', label='Actual Angle (degrees)')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Angle (degrees)')
    plt.legend()
    plt.title('Steering Angle Response with PID Control')
    
    # Plot control signal
    plt.subplot(2, 1, 2)
    plt.plot(time_points, control_signals, 'g-', label='Control Signal (torque)')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Signal (Nm)')
    plt.legend()
    plt.title('Control Signal (Torque)')
    
    plt.tight_layout()
    plt.savefig('steering_response.png')
    
    # Create animation of the steering wheel
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(frame):
        angle = actual_angles[frame]
        visualize_steering_wheel(angle, fig, ax)
        return ax,
    
    # Create animation with a subset of frames to make it smoother
    # anim_frames = min(200, num_steps)  # Limit frames for smoother animation
    # frame_indices = np.linspace(0, num_steps-1, anim_frames, dtype=int)
    
    # anim = FuncAnimation(fig, update, frames=frame_indices, interval=50, blit=True)
    # plt.tight_layout()
    
    # Save animation (optional)
    # anim.save('steering_animation.gif', writer='pillow', fps=20)
    
    # plt.show()


if __name__ == "__main__":
    main()