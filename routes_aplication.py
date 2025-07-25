import subprocess
import os
from flask import Flask, jsonify, request
from threading import Lock
from app import app
process_lock = Lock()


@app.route('/run_script', methods=['POST'])
def run_script():
    if process_lock.acquire(blocking=False):
        try:
            # Determine the path to the Python executable in the virtual environment
            venv_python_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv', 'Scripts', 'python.exe')
            if not os.path.exists(venv_python_path):
                # Fallback for non-Windows or if venv is structured differently
                venv_python_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv', 'bin', 'python')

            if not os.path.exists(venv_python_path):
                return jsonify({'status': 'error', 'message': 'Virtual environment Python executable not found.'}), 500

            # Start the process and save its PID to a file for later termination
            main_app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'application', 'main_aplication.py')
            process = subprocess.Popen([venv_python_path, main_app_path], cwd=os.path.dirname(os.path.abspath(__file__)))
            pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main_aplication.pid')
            with open(pid_file, 'w') as f:
                f.write(str(process.pid))

            return jsonify({'status': 'success', 'message': 'main_aplication.py started.'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
        finally:
            process_lock.release()
    else:
        return jsonify({'status': 'error', 'message': 'main_aplication.py is already running.'}), 409

@app.route('/stop_script', methods=['POST'])
def stop_script():
    import signal
    pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main_aplication.pid')
    if not os.path.exists(pid_file):
        return jsonify({'status': 'error', 'message': 'No running main_aplication.py process found.'}), 404

    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        # Try to terminate the process
        os.kill(pid, signal.SIGTERM)
        # Optionally, wait and force kill if still alive
        import time
        time.sleep(1)
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass  # Process already terminated

        os.remove(pid_file)
        return jsonify({'status': 'success', 'message': 'main_aplication.py stopped.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to stop main_aplication.py: {str(e)}'}), 500