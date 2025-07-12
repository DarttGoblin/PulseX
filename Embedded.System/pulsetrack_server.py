from flask import Flask, request, jsonify
import json

app = Flask(__name__)

STATE_FILE = "device_state.json"

# Initialize state file
def init_state():
    with open(STATE_FILE, "w") as f:
        json.dump({"device_enabled": False, "metadata": {}}, f)

@app.route('/submit_metadata', methods=['POST'])
def submit_metadata():
    data = request.json
    # store metadata & enable device
    with open(STATE_FILE, "w") as f:
        json.dump({"device_enabled": True, "metadata": data}, f)
    return jsonify({"status": "success", "device_enabled": True})

@app.route('/status', methods=['GET'])
def status():
    with open(STATE_FILE, "r") as f:
        state = json.load(f)
    return jsonify(state)

if __name__ == '__main__':
    init_state()
    app.run(host="0.0.0.0", port=5000)
