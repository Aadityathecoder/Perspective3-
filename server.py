import cv2
import numpy as np 
from flask import Flask, Response, request, jsonify

CAME_INDEX = 0
app = Flask(__name__)

# Camera is opened one time only
camera = cv2.VideoCapture(CAME_INDEX)

if not camera.isOpened():
    raise RuntimeError("Camera wont load until fixed something is fixed.")
    
state = {
    "up": False,
    "down": False,
    "left": False,
    "right": False,
    "command": "stop"
}

def interested_area(image, points):
    height = image.shape[0]
    width = image.shape[1]
    
    mask = np.zeros((height, width))
    mask = mask.astype("uint8")
    cv2.fillPoly(mask, points, 255)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    return masked_image
    
def draw_lines(image, lines, color=(0, 255, 0), thickness=5):
    height = image.shape[0]
    width = image.shape[1]
    
    result = np.zeros((height, width, 3))
    result = result.astype("uint8")
    
    if lines is None:
        return result
    for line in lines:
        if line is None:
            continue
        if len(line) != 4:
            continue
        
        x1, y1, x2, y2 = line
        cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        
    if len(lines) == 2 and lines[0] is not None and lines[1] is not None:
        left = lines[0]
        right = lines[1]
        
        #aadi's midpt creation
        mid_x1 = (left[0] + right[0]) //2
        mid_y1 = (left[1] + right[1]) //2
        mid_x2 = (left[2] + right[2]) //2
        mid_y2 = (left[3] + right[3]) //2 
        
        cv2.line(result, (mid_x1, mid_y1), (mid_x2, mid_y2), (0, 255, 255), thickness)
    return result
    
def create_coordinates(image, parameters, y1, y2):
    if parameters is None:
        return None
    slope, interc = parameters
    
    if slope == 0:
        return None
        
    x1 = int((y1-interc)/slope)
    x2 = int((y2-interc)/slope)
    return np.array([x1, y1, x2, y2])
    
def slopes_average_intercept(lines, image):
    left = []
    right = []
    
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        
        if x2-x1 == 0:
            continue
        slope = (y2-y1)/(x2-x1)
        interc = y1-slope*x1 
        
        if abs(slope) < 0.5:
            continue
        if slope < 0:
            left.append((slope, interc))
        else:
            right.append((slope, interc))
            
    y1 = image.shape[0]
    y2 = int(y1*0.6)
    result = []
    
    if len(right) > 0:
        right_avg = np.average(right, axis=0)
        right_line = create_coordinates(image, right_avg, y1, y2)
        if right_line is not None:
            result.append(right_line)
    if len(left) > 0:
        left_avg = np.average(left, axis=0)
        left_line = create_coordinates(image, left_avg, y1, y2)
        if left_line is not None:
            result.append(left_line)
    if len(result) == 0:
        return None
    return np.array(result)

def detect_lanes(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    
    points = np.array([[
        (int(width * 0.10), height),
        (int(width * 0.45), int(height * 0.60)),
        (int(width * 0.55), int(height * 0.60)),
        (int(width * 0.90), height)
    ]])
    points = points.astype(int)
    edges = interested_area(edges, points)
    
    lines = cv2.HoughLinesP(edges, 2, np.pi/180, 50, minLineLength=40, maxLineGap=100)
    averaged = slopes_average_intercept(lines, frame)
    line_image = draw_lines(frame, averaged)
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    if averaged is not None and len(averaged) == 2:
        left = averaged[0]
        right = averaged[1]
        xs = [left[0], left[2], right[0], right[2]]
        ys = [left[1], left[3], right[1], right[3]]
        pad = 20
        x1 = max(0, min(xs)-pad)
        y1 = max(0, min(ys)-pad)
        x2 = min(width, max(xs)+pad)
        y2 = min(height, max(ys)+pad)
        
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return result

def lane_generation():
    while True:
        got_frame, frame = camera.read()
        if not got_frame:
            continue
        frame = detect_lanes(frame)
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        
        frame_bytes = jpg.tobytes()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        
def raw_feed_generation():
    while True:
        ok, frame = camera.read()
        if not ok:
            continue
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"

@app.route("/video_feed_lanes")
def vid_feed_lanes():
    return Response(lane_generation(),  
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_raw")
def vid_feed_raw():
    return Response(raw_feed_generation(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    with open("gui.html", "r") as f:
        return f.read()

@app.route("/control", methods=["POST"])
def control():
    global state
    data = request.json
    
    if "up" in data:
        state["up"] = data["up"]
    if "down" in data:
        state["down"] = data["down"]
    if "left" in data:
        state["left"] = data["left"]
    if "right" in data:
        state["right"] = data["right"]
    if "command" in data:
        state["command"] = data["command"]
    
    return jsonify({"status": "ok", "state": state})

@app.route("/state", methods=["GET"])
def get_state():
    return jsonify(state)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)

