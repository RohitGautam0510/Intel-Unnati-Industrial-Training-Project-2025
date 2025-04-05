import cv2
import time

def record_video(duration=10):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('store_video.mp4', fourcc, 20.0, (640,480))
    
    start_time = time.time()
    print('Recording... (10 seconds)')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Recording...', frame)
            
            # Break after duration seconds or if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) >= duration:
                break
        else:
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('Recording complete: store_video.mp4')
