# setup docker apt repo
sudo apt-get install -y lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# install docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo apt-get clean

# setup jupyter app launcher
mkdir -p ~/.local/share/jupyter/jupyter_app_launcher/
ln -s /project/.devx/jp_app_launcher.yaml ~/.local/share/jupyter/jupyter_app_launcher/jp_app_launcher.yaml

# allow sudo without password
echo "$(whoami) ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/workbench-persist > /dev/null

# load a secrets file, if one exists
bash -c 'cat << "EOF" >> ~/.bashrc

# load secrets
set -a
source /project/secrets.env
set +a
EOF'

# include home dir bin
bash -c 'cat << EOF >> ~/.bashrc

# add bin dirs
export PATH=\$PATH:~/.local/bin/:~/bin
EOF'
