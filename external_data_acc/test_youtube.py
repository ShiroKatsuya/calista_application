from pytubefix import YouTube
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def test_video(url):
    try:
        print(f"\nTesting URL: {url}")
        
        yt = YouTube(url)
        print(f"Title: {yt.title}")
        
        print("Attempting to get streams...")
        streams = yt.streams.filter(progressive=True, file_extension='mp4')
        print(f"Available streams: {len(streams)}")
        
        stream = streams.order_by('resolution').desc().first()
        if stream:
            print(f"Found stream: {stream}")
            print("Download successful!")
            # Uncomment to actually download:
            # stream.download(filename="test_download.mp4")
        else:
            print("No suitable stream found")
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test with the working video first
    test_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    
    # Now test with the originally problematic video
    test_video("https://www.youtube.com/watch?v=aDEDfbN5gIU") 