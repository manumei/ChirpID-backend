import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import create_app

app = create_app()

if __name__ == "__main__":
    print("Starting ChirpID Backend Server...")
    debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    port = int(os.getenv('PORT', 5001))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
