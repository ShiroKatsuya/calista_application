from external_data_acc.youtube_search import download_youtube_audio

def main():
    # Try the problematic video
    print("\nTesting problematic video:")
    result = download_youtube_audio("https://www.youtube.com/watch?v=aDEDfbN5gIU")
    print(f"Result: {result}")
    
    # Try a known working video
    print("\nTesting working video:")
    result = download_youtube_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print(f"Result: {result}")

if __name__ == "__main__":
    main() 