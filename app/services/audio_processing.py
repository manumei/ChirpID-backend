import random

def process_audio(file_path):
    # Mock bird identification data for testing
    # In a real implementation, this would use a ML model to analyze the audio
    
    mock_birds = [
        {
            "species": "American Robin",
            "scientificName": "Turdus migratorius",
            "confidence": 0.87
        },
        {
            "species": "House Sparrow",
            "scientificName": "Passer domesticus", 
            "confidence": 0.92
        },
        {
            "species": "Northern Cardinal",
            "scientificName": "Cardinalis cardinalis",
            "confidence": 0.78
        },
        {
            "species": "Blue Jay",
            "scientificName": "Cyanocitta cristata",
            "confidence": 0.85
        },
        {
            "species": "Song Sparrow",
            "scientificName": "Melospiza melodia",
            "confidence": 0.91
        }
    ]
    
    # Return a random bird for testing
    selected_bird = random.choice(mock_birds)
    
    return selected_bird
