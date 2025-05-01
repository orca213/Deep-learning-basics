#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Behavioral Cloning from PID Controller in MetaDrive Environment
This script collects optimal driving data using a PID controller,
then trains a neural network to imitate the driving behavior.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pickle

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
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)  # Anti-windup
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = self.kd * (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        
        # Total control signal
        control = p_term + i_term + d_term
        
        # Clip the control signal
        control = np.clip(control, -1.0, 1.0)
        
        return control


class BehavioralCloningModel(nn.Module):
    """
    Neural network model for behavioral cloning.
    """
    def __init__(self, input_size=259, hidden_size=128, output_size=2):
        """
        Initialize the model with specified layer sizes.
        
        Args:
            input_size (int): Size of input observation (lidar data)
            hidden_size (int): Size of hidden layers
            output_size (int): Size of output action (steering and throttle_brake)
        """
        super(BehavioralCloningModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            
            nn.Linear(hidden_size // 2, output_size),
            nn.Tanh()  # Tanh activation to bound outputs between -1 and 1
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input observation
            
        Returns:
            torch.Tensor: Predicted action
        """
        return self.network(x)


class DrivingDataset(Dataset):
    """
    Dataset for training the behavioral cloning model.
    """
    def __init__(self, observations, actions):
        """
        Initialize the dataset.
        
        Args:
            observations (numpy.ndarray): Array of observations
            actions (numpy.ndarray): Array of actions
        """
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
    
    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.observations)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (observation, action)
        """
        return self.observations[idx], self.actions[idx]


def create_directories():
    """
    Create necessary directories for saving data and models.
    """
    directories = ["data", "models", "plots"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}/")


def collect_pid_driving_data(num_episodes=10, max_steps_per_episode=500):
    """
    Collect optimal driving data using PID controller.
    
    Args:
        num_episodes (int): Number of episodes to collect data from
        max_steps_per_episode (int): Maximum steps per episode
        
    Returns:
        tuple: (observations, actions)
    """
    # Initialize environment
    env = MetaDriveEnv(CONFIG)
    
    # Initialize PID controller for steering
    pid_controller = PIDController(kp=0.8, ki=0.1, kd=0.2)
    
    # Fixed throttle value
    FIXED_THROTTLE = 0.05
    
    # Data collection variables
    all_observations = []
    all_actions = []
    dt = 0.1  # Assumed time step
    
    print(f"Collecting optimal driving data using PID controller for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Reset environment
        obs, info = env.reset()
        
        # Reset PID controller
        pid_controller.integral = 0.0
        pid_controller.prev_error = 0.0
        
        episode_frames = []
        episode_observations = []
        episode_actions = []
        
        # Run episode
        for step in tqdm(range(max_steps_per_episode), 
                        desc=f"Episode {episode+1}/{num_episodes}", 
                        unit="steps"):
            
            # Calculate heading error
            heading_error, ref_line_heading, current_heading = get_heading_diff(env)
            
            # Compute steering command using PID controller
            steering = pid_controller.compute(0, heading_error, dt)
            
            # Create action with computed steering and fixed throttle
            action = np.array([steering, FIXED_THROTTLE])
            
            # Store observation and action
            episode_observations.append(obs.copy())
            episode_actions.append(action.copy())
            
            # Apply action to environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render frame for visualization
            if episode == 0:  # Save frames only for the first episode to save space
                frame = env.render(mode="topdown", screen_record=True, window=False)
                episode_frames.append(frame)
            
            # End episode if terminated
            if terminated or truncated:
                print(f"Episode {episode+1} terminated after {step+1} steps")
                break
        
        # Add episode data to collection
        all_observations.extend(episode_observations)
        all_actions.extend(episode_actions)
        
        # Generate GIF for first episode
        if episode == 0 and episode_frames:
            print("Generating GIF of PID driving...")
            generate_gif(episode_frames, "data", "pid_driving.gif")
    
    # Close environment
    env.close()
    
    # Convert to numpy arrays
    all_observations = np.array(all_observations, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)
    
    print(f"Data collection complete: {len(all_observations)} samples collected")
    print(f"Observations shape: {all_observations.shape}")
    print(f"Actions shape: {all_actions.shape}")
    
    # Save collected data
    with open("data/driving_data.pkl", "wb") as f:
        pickle.dump({
            "observations": all_observations,
            "actions": all_actions
        }, f)
    
    print("Saved data to data/driving_data.pkl")
    
    return all_observations, all_actions


def train_behavioral_cloning(observations, actions, epochs=100, batch_size=64, lr=0.001):
    """
    Train the behavioral cloning model.
    
    Args:
        observations (numpy.ndarray): Array of observations
        actions (numpy.ndarray): Array of actions
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate
        
    Returns:
        BehavioralCloningModel: Trained model
    """
    # Create dataset
    dataset = DrivingDataset(observations, actions)
    
    # Split into training and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = BehavioralCloningModel(input_size=observations.shape[1])
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # Training history
    train_losses = []
    val_losses = []
    
    print(f"Training behavioral cloning model for {epochs} epochs...")
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for obs_batch, act_batch in train_loader:
            # Move data to device
            obs_batch, act_batch = obs_batch.to(device), act_batch.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(obs_batch)
            
            # Calculate loss
            loss = criterion(outputs, act_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                # Move data to device
                obs_batch, act_batch = obs_batch.to(device), act_batch.to(device)
                
                # Forward pass
                outputs = model(obs_batch)
                
                # Calculate loss
                loss = criterion(outputs, act_batch)
                
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "models/behavioral_cloning_model.pth")
    print("Model saved to models/behavioral_cloning_model.pth")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Behavioral Cloning Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/training_history.png")
    
    return model


def evaluate_model(model, num_episodes=1, max_steps_per_episode=1000):
    """
    Evaluate the trained model in the environment.
    
    Args:
        model (BehavioralCloningModel): Trained model
        num_episodes (int): Number of episodes to evaluate
        max_steps_per_episode (int): Maximum steps per episode
    """
    # Initialize environment
    env = MetaDriveEnv(CONFIG)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Evaluating behavioral cloning model for {num_episodes} episodes...")
    
    # Evaluation metrics
    episode_rewards = []
    episode_steps = []
    completion_rates = []
    
    for episode in range(num_episodes):
        # Reset environment
        obs, info = env.reset()
        
        episode_frames = []
        total_reward = 0
        
        # Run episode
        for step in tqdm(range(max_steps_per_episode), 
                        desc=f"Evaluation Episode {episode+1}/{num_episodes}", 
                        unit="steps"):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Get action from model
            with torch.no_grad():
                action = model(obs_tensor).cpu().numpy()[0]
            
            # Apply action to environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Accumulate reward
            total_reward += reward
            
            # Render frame
            frame = env.render(mode="topdown", screen_record=True, window=False)
            episode_frames.append(frame)
            
            # End episode if terminated
            if terminated or truncated:
                print(f"Episode {episode+1} terminated after {step+1} steps")
                break
        
        # Generate GIF
        print(f"Generating GIF for evaluation episode {episode+1}...")
        generate_gif(episode_frames, "plots", f"eval_episode_{episode+1}.gif")
        
        # Store metrics
        episode_rewards.append(total_reward)
        episode_steps.append(step + 1)
        
        # Calculate completion rate if available in info
        if hasattr(env, 'vehicle') and hasattr(env.vehicle, 'navigation'):
            try:
                road_completion = min(1.0, env.vehicle.navigation.route_completion)
                completion_rates.append(road_completion)
                print(f"Route completion: {road_completion:.2f}")
            except:
                pass
    
    # Close environment
    env.close()
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_steps):.2f} steps")
    if completion_rates:
        print(f"Average completion rate: {np.mean(completion_rates):.2f}")
    
    # Plot evaluation results
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, num_episodes+1), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Evaluation Results - Total Reward per Episode')
    plt.grid(True)
    plt.savefig("plots/evaluation_rewards.png")


def compare_pid_and_bc(num_steps=1000):
    """
    Compare PID controller and Behavioral Cloning model side by side.
    
    Args:
        num_steps (int): Number of steps to run for comparison
    """
    # Check if model exists
    model_path = "models/behavioral_cloning_model.pth"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    # Initialize only one environment
    env = MetaDriveEnv(CONFIG)
    
    # Initialize PID controller
    pid_controller = PIDController(kp=0.8, ki=0.1, kd=0.2)
    
    # Load BC model
    model = BehavioralCloningModel(input_size=259)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Fixed parameters
    FIXED_THROTTLE = 0.05
    dt = 0.1
    
    # Data for comparison
    frames_pid = []
    frames_bc = []
    steering_values_pid = []
    steering_values_bc = []
    
    print("Running comparison between PID controller and Behavioral Cloning model...")
    
    # Reset PID controller
    pid_controller.integral = 0.0
    pid_controller.prev_error = 0.0
    
    # First run: PID controller
    obs, _ = env.reset()
    print("Running PID controller...")
    for step in tqdm(range(num_steps), desc="PID Controller", unit="steps"):
        # Get heading error
        heading_error, ref_line_heading, current_heading = get_heading_diff(env)
        
        # Compute PID control
        steering_pid = pid_controller.compute(0, heading_error, dt)
        action_pid = np.array([steering_pid, FIXED_THROTTLE])
        
        # Store values
        steering_values_pid.append(steering_pid)
        
        # Apply action
        obs, _, terminated, truncated, _ = env.step(action_pid)
        
        # Render frame
        frame = env.render(mode="topdown", screen_record=True, window=False)
        frames_pid.append(frame)
        
        # Break if environment terminates
        if terminated or truncated:
            print(f"PID controller episode terminated after {step+1} steps")
            break
    
    # Reset PID controller
    pid_controller.integral = 0.0
    pid_controller.prev_error = 0.0
    
    # Second run: Behavioral Cloning model
    obs, _ = env.reset()
    print("Running Behavioral Cloning model...")
    for step in tqdm(range(num_steps), desc="Behavioral Cloning", unit="steps"):
        # Get BC action
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action_bc = model(obs_tensor).cpu().numpy()[0]
        
        # Store values
        steering_values_bc.append(action_bc[0])
        
        # Apply action
        obs, _, terminated, truncated, _ = env.step(action_bc)
        
        # Render frame
        frame = env.render(mode="topdown", screen_record=True, window=False)
        frames_bc.append(frame)
        
        # Break if environment terminates
        if terminated or truncated:
            print(f"Behavioral Cloning episode terminated after {step+1} steps")
            break
    
    # Close environment
    env.close()
    
    # Make sure both arrays have the same length for plotting
    min_length = min(len(steering_values_pid), len(steering_values_bc))
    steering_values_pid = steering_values_pid[:min_length]
    steering_values_bc = steering_values_bc[:min_length]
    
    # Generate GIFs
    print("Generating comparison GIFs...")
    generate_gif(frames_pid, "plots", "pid_driving_comparison.gif")
    generate_gif(frames_bc, "plots", "bc_driving_comparison.gif")
    
    # Plot steering comparison
    time_steps = np.arange(min_length) * dt
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, steering_values_pid, 'b-', label='PID Controller')
    plt.plot(time_steps, steering_values_bc, 'r-', label='Behavioral Cloning')
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Command')
    plt.title('Comparison of Steering Commands: PID vs Behavioral Cloning')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/steering_comparison.png")
    
    print("Comparison complete. Results saved to plots/ directory.")


def main():
    """
    Main function to run the behavioral cloning project.
    """
    # Create necessary directories
    create_directories()
    
    # Check if data already exists
    data_path = "data/driving_data.pkl"
    if os.path.exists(data_path):
        print(f"Loading existing data from {data_path}")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        observations = data["observations"]
        actions = data["actions"]
    else:
        # Collect optimal driving data using PID controller
        observations, actions = collect_pid_driving_data(num_episodes=10)
    
    # Train behavioral cloning model
    model = train_behavioral_cloning(observations, actions, epochs=100)
    
    # Evaluate the trained model
    evaluate_model(model, num_episodes=3)
    
    # Compare PID controller and Behavioral Cloning model
    compare_pid_and_bc()


if __name__ == "__main__":
    main()