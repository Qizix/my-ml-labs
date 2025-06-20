import os
import argparse
import numpy as np
import random
import time
import cv2
import mediapipe as mp


def create_digital_glitch(face_region):
    """Create subtle digital glitch effects"""
    h, w = face_region.shape[:2]
    glitched = face_region.copy()
    
    # RGB channel displacement for chromatic aberration
    if len(face_region.shape) == 3:
        # Subtle red channel shift
        red_shift = np.roll(glitched[:, :, 2], random.randint(-3, 3), axis=1)
        glitched[:, :, 2] = red_shift
        
        # Subtle blue channel shift in opposite direction
        blue_shift = np.roll(glitched[:, :, 0], random.randint(-2, 2), axis=1)
        glitched[:, :, 0] = blue_shift
    
    # Add occasional horizontal line glitches
    if random.random() < 0.6:  # 60% chance
        for _ in range(random.randint(1, 3)):
            line_y = random.randint(0, h-2)
            line_height = random.randint(1, 2)
            shift_amount = random.randint(-8, 8)
            
            if line_y + line_height < h:
                line_segment = glitched[line_y:line_y + line_height, :].copy()
                line_segment = np.roll(line_segment, shift_amount, axis=1)
                glitched[line_y:line_y + line_height, :] = line_segment
    
    return glitched

def create_advanced_pixelation(face_region):
    """Create dynamic pixelation with varying block sizes"""
    h, w = face_region.shape[:2]
    pixelated = face_region.copy()
    
    # Create different pixelation zones
    block_sizes = [6, 8, 10, 12]
    
    # Divide face into regions and apply different pixelation
    for i in range(2):
        for j in range(2):
            y_start = i * h // 2
            y_end = (i + 1) * h // 2
            x_start = j * w // 2
            x_end = (j + 1) * w // 2
            
            region = face_region[y_start:y_end, x_start:x_end]
            if region.size > 0:
                block_size = random.choice(block_sizes)
                
                # Pixelate region
                small_h, small_w = max(1, region.shape[0] // block_size), max(1, region.shape[1] // block_size)
                small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                pixelated_region = cv2.resize(small, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                pixelated[y_start:y_end, x_start:x_end] = pixelated_region
    
    return pixelated

def add_cyberpunk_filter(face_region):
    """Apply cyberpunk-style color grading and effects"""
    filtered = face_region.copy().astype(np.float32)
    
    # Enhanced contrast and brightness
    filtered = cv2.convertScaleAbs(filtered, alpha=1.2, beta=10)
    
    # Cyberpunk color grading - cooler tones with cyan/blue tint
    filtered = filtered.astype(np.float32)
    
    # Reduce red, enhance blue and green slightly
    filtered[:, :, 0] *= 1.1  # Blue
    filtered[:, :, 1] *= 1.05  # Green  
    filtered[:, :, 2] *= 0.9   # Red
    
    # Add subtle cyan tint
    filtered[:, :, 0] = np.minimum(filtered[:, :, 0] + 15, 255)  # More blue
    filtered[:, :, 1] = np.minimum(filtered[:, :, 1] + 8, 255)   # Slight green
    
    return np.clip(filtered, 0, 255).astype(np.uint8)

def add_scan_lines_effect(face_region):
    """Add subtle scan lines for digital display effect"""
    h, w = face_region.shape[:2]
    scan_lined = face_region.copy().astype(np.float32)
    
    # Add horizontal scan lines every few pixels
    for y in range(0, h, 4):
        if y < h:
            scan_lined[y, :] *= 0.85
    
    # Add very subtle vertical lines occasionally
    if random.random() < 0.3:
        for x in range(0, w, 8):
            if x < w:
                scan_lined[:, x] *= 0.95
    
    return np.clip(scan_lined, 0, 255).astype(np.uint8)

def add_digital_noise(face_region):
    """Add subtle digital noise/grain"""
    if random.random() < 0.7:  # 70% chance
        noise_intensity = random.randint(5, 15)
        noise = np.random.normal(0, noise_intensity, face_region.shape).astype(np.int16)
        
        noisy = face_region.astype(np.int16) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    return face_region

def create_edge_enhancement(face_region):
    """Enhance edges for that digital sharpness"""
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Convert edges back to 3-channel
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Blend with original (very subtle)
    enhanced = cv2.addWeighted(face_region, 0.9, edges_colored, 0.1, 0)
    
    return enhanced

def process_img(img, face_detection):
    H, W, _ = img.shape
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)
            
            # Extract face region
            face_region = img[y1:y1 + h, x1:x1 + w, :].copy()
            
            if face_region.size > 0:  # Make sure face region is valid
                # Apply cyberpunk anonymization effects in sequence
                
                # 1. Advanced pixelation (primary anonymization)
                processed_face = create_advanced_pixelation(face_region)
                
                # 2. Apply cyberpunk color grading
                processed_face = add_cyberpunk_filter(processed_face)
                
                # 3. Add digital glitch effects
                if random.random() < 0.8:  # 80% chance
                    processed_face = create_digital_glitch(processed_face)
                
                # 4. Add scan lines for digital display effect
                processed_face = add_scan_lines_effect(processed_face)
                
                # 5. Edge enhancement for sharpness
                processed_face = create_edge_enhancement(processed_face)
                
                # 6. Add digital noise/grain
                processed_face = add_digital_noise(processed_face)
                
                # 7. Final contrast adjustment
                processed_face = cv2.convertScaleAbs(processed_face, alpha=1.1, beta=5)
                
                # Apply the processed face back to the image
                img[y1:y1 + h, x1:x1 + w, :] = processed_face
                
                # Optional: Add subtle glowing border effect
                if random.random() < 0.2:  # 20% chance
                    # Subtle cyan glow border
                    cv2.rectangle(img, (x1-1, y1-1), (x1+w+1, y1+h+1), (255, 255, 0), 1)
    
    return img


args = argparse.ArgumentParser()

args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default=None)

args = args.parse_args()


output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    if args.mode in ["image"]:
        # read image
        img = cv2.imread(args.filePath)

        img = process_img(img, face_detection)

        # save image
        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

    elif args.mode in ['video']:

        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                       25,
                                       (frame.shape[1], frame.shape[0]))

        while ret:

            frame = process_img(frame, face_detection)

            output_video.write(frame)

            ret, frame = cap.read()

        cap.release()
        output_video.release()

    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(1)

        ret, frame = cap.read()
        while ret:
            frame = process_img(frame, face_detection)

            cv2.imshow('frame', frame)
            cv2.waitKey(25)

            ret, frame = cap.read()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()