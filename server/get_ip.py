#!/usr/bin/env python3
"""
Script to get the local IP address for the React Native app configuration.
"""
import socket

def get_local_ip():
    """Get the local IP address that can be used by other devices on the network"""
    try:
        # Connect to a remote address to determine the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        print(f"Error getting IP: {e}")
        return "localhost"

if __name__ == "__main__":
    ip = get_local_ip()
    print(f"Local IP Address: {ip}")
    print(f"Your React Native app should use: EXPO_PUBLIC_API_URL=http://{ip}:5000")
    print("\nUpdate your .env file in the React Native project with this URL.")
