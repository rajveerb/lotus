# Anaconda environment

[How to generate `conda_requirements.env` file?](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#exporting-the-environment-yml-file)

1. Activate the environment to export: 

    `conda activate myenv`
    
    Note: Replace `myenv` with the name of the environment.

2. Export your active environment to a new file:
    
    `conda env export > conda_requirements.env`

[How to create a conda enviroment using above file?](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

1. First change the environment name in `conda_requirements.env`, if you want to call it something specific

2. Create the environment from the `conda_requirements.env` file:
    
    `conda env create -f conda_requirements.env`

3. Activate the new environment: 
    
    `conda activate <your_env_name>`

4. Verify that the new environment was installed correctly:
    
    `conda env list` 