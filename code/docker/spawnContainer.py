import docker

# API Refrence: https://docker-py.readthedocs.io/en/stable/containers.html

# Create a Docker client
client = docker.from_env()


def spawnContainer(image, container_name, command, environment=[], ports={}, volumes=[], ipc_mode="host"):
    # Define container options (optional)
    container_options = {
        "detach": True,    # Run the container in the background
        "tty": True,       # Allocate a pseudo-TTY
        "name": container_name,
        "environment": environment,
        "ports": ports,
        "volumes": volumes,
        "command": command,
        "ipc_mode": ipc_mode
    }

    # Create and start the container
    try:
        container = client.containers.run(image, **container_options)
        print(f"Container {container.name} is running.")
    except docker.errors.ImageNotFound:
        print(f"Image {image} not found. Make sure it exists.")
    except docker.errors.APIError as e:
        print(f"Error starting container: {e}")
