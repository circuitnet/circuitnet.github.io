## ISEDA 2023 Tutorial

1. Connect to VPN.
2. Open terminal.

    For Windows: press win+r, enter cmd to open terminal.

    For MAC: open terminal.

1. Connect to remote server.

    (1) Enter: `ssh -p 22 tutorialxx@10.100.2.16 –L 127.0.0.1:8888:127.0.0.1:80xx `

    where tutorialxx is your account name, and xx is the number in your account name.

    e.g. ssh -p 22 tutorial01@10.100.2.16 –L 127.0.0.1:8888:127.0.0.1:8001 

    (2) Enter password, which is the same as your account name.

4. Setup environment.

    Type following commands and press enter to activate conda environment, copy tutorial scripts into your work directory, and run Jupyter notebook.

    `source /home/circuitnet/bashrc`

    `cp -r /home/circuitnet/circuitnet-tutorial ./`

    `jupyter notebook --config /home/circuitnet/jupyter_notebook_config.py --port 80xx `

   where 80xx is the same as the one in step 3.

5. Run Jupyter notebook.
   
    (1) Open Internet browser and enter `127.0.0.1:8888` as the url address.

    (2) Enter `iseda` as the password to enter Jupyter notebook.

6. Try feature extraction and model inference for routability prediction.
   
    Open circuitnet_tutorial.ipynb, and run code blocks one by one.
