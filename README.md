## Getting started with Hyperparameter Optimisation

Welcome to the Getting started with Hyperparameter Optimisation project.

__Please start by creating a copy of this project in your personal projects space.__
To avoid accidentaly modifying the master project, please create a copy in your personal projects space. You can do this by clicking the Copy button in the top-right corner of the screen. 

![Copy button](images/copy.png)

This will open a Copy dialog, promting you to enter a new project name. You can leave this field blank and keep the project's original name. Since the copy will reside in your personal namespace, its name will not clash with the original project. 
Now click the __Copy__ button. Domino creates a personal copy of this project and automatically switches you to this new copy.

### Project description

This project contains all the hands-on work for the Getting started with Hyperparameter Optimisation workshop. All the assets are available in the [Files](browse) section section of the project. 

### Setting up your JupyterLab workspace

A Domino workspace is an interactive session where you can conduct research, analyze data, train models, and more. Use workspaces to work in the development environment of your choice, like [Jupyter](https://jupyter.org/) notebooks, [RStudio](https://rstudio.com/), [VS Code](https://code.visualstudio.com/), and many other customizable environments.

When creating a new Workspace for this workshop, please keep the following in mind:

* Make sure you select JupyterLab as your IDE
* Make sure the select compute environment is left to the default value (Hyperopt Workspace)
* Make sure your hardware tier is set to "Small".

To create a new Workspace for your project, go to Workspaces -> Create New Workspace, or simply press the button below:

[![Run Notebook](images/create_workspace.png)](/workspace/:ownerName/:projectName?showWorkspaceLauncher=True)

### Setting up your Ray workspace

We'll use a distributed Ray cluster for the second part of this workshop. You could attach a cluster to your existing workspace or simply create a second one for your project. Whichever way you decide to go, the settings in the Environment & Hardware section should be identical to what we used initially. 
The only change you need to make is in the Compute Cluster section, which should look like this:

* Make sure that "Attach Compute Cluster" is set to Ray
* Make sure that "Min workers" is set to 1
* Make sure that "Cluster Compute Environment" is set to "Hyperopt Ray"

![Ray workspace](images/ray.png)

### Compute environments

This project uses two compute environments --- Hyperopt Workspace abd Hyperopt Ray. This section contains the respective Dockerfiles.

```
# Hyperopt Workspace
FROM quay.io/domino/compute-environment-images:ubuntu20-py3.9-r4.2-domino5.2-gpu

USER ubuntu
RUN pip install --upgrade pip

RUN pip install torch --user torchsummary==1.5.1 torchvision==0.13.1

RUN pip install hyperopt==0.2.7

RUN pip install --user ray[all]==1.12.0
```

The Pluggable Workspace Tools configuration for this environment is as follows:

```
jupyter:
  title: "Jupyter (Python, R, Julia)"
  iconUrl: "/assets/images/workspace-logos/Jupyter.svg"
  start: [ "/var/opt/workspaces/jupyter/start" ]
  httpProxy:
    port: 8888
    rewrite: false
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    requireSubdomain: false
  supportedFileExtensions: [ ".ipynb" ]
jupyterlab:
  title: "JupyterLab"
  iconUrl: "/assets/images/workspace-logos/jupyterlab.svg"
  start: [  /var/opt/workspaces/Jupyterlab/start.sh ]
  httpProxy:
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    port: 8888
    rewrite: false
    requireSubdomain: false
```

The Ray cluster uses the following compute environment:

```
# Hyperopt Ray
FROM quay.io/domino/cluster-environment-images:ray1.12.0-py3.9.5-gpu

RUN pip install torchsummary==1.5.1 torchvision==0.13.1

RUN pip install hyperopt==0.2.7

USER root
RUN \
  groupadd -g 12574 ubuntu && \
  useradd -u 12574 -g 12574 -m -N -s /bin/bash ubuntu


RUN chmod a+rw /home/ray
```

Also note, that when creating the Hyperopt Ray environment, Supported Cluster Settings should be set to Ray.
