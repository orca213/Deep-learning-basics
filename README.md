# Deep Learning Basics

This project introduces fundamental concepts of deep learning along with practical code implementations.

## Installation

### 1. Clone the Repository

Run the following commands to clone the repository and navigate to the project directory:

```bash
git clone https://github.com/orca213/Deep-learning-basics.git
cd Deep-learning-basics
```

### 2. Environments

Choose one of the following methods to set up your local practice environment.

<details>
    <summary><b>A. Docker</b></summary>

### A.1 Install Docker Desktop

Download and install Docker Desktop from [this link](https://www.docker.com/).

### A.2 Build the Docker Container

Set up an Ubuntu-based environment for hands-on practice by running:

```bash
bash docker/run_docker.sh
```

</details>

<details>
    <summary><b>B. Vessl AI</b></summary>

### B.1 Generate SSH key

Generate local SSH key. Press enter twice after executing the following.

```bash
ssh-keygen -t rsa -C "vessl-ai"
```

### B.2 Register public key to Vessl-AI

Copy the output and paste it in Vessl-AI > Profile > Account settings > General > SSH public keys > + Add key

```bash
cat ~/.ssh/id_rsa.pub
```

</details>

## Project Contents

This project covers the following topics:

- **Python Basics**
- **Reinforcement Learning**
- **Convolutional Neural Networks (CNN)**

Additionally, it includes some interesting side projects:

- **Selenium**
