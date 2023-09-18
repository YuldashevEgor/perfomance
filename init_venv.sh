python3.8 -m venv "$(pwd)/venv"
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools
pip install "git+https://github.com/ai-forever/Kandinsky-2.git"