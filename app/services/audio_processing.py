import random

def process_audio(file_path):
    # Mock bird identification data for testing
    # In a real implementation, this would use a ML model to analyze the audio
    
    mock_birds = [
        {
            "species": "Rufous Hornero",
            "scientificName": "Furnarius rufus",
            "confidence": 0.89
        },
        {
            "species": "Toco Toucan",
            "scientificName": "Ramphastos toco", 
            "confidence": 0.92
        },
        {
            "species": "Andean Condor",
            "scientificName": "Vultur gryphus",
            "confidence": 0.78
        },
        {
            "species": "Southern Lapwing",
            "scientificName": "Vanellus chilensis",
            "confidence": 0.85
        },
        {
            "species": "Great Kiskadee",
            "scientificName": "Pitangus sulphuratus",
            "confidence": 0.91
        }
    ]
    
    # Return a random bird for testing
    selected_bird = random.choice(mock_birds)
    
    return selected_bird
