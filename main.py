import cv2 # type: ignore
import numpy as np # type: ignore
from water_data import WaterDataRecorder
import os
from config_loader import load_config, clahe_cfg


cfg = load_config()
cap = cv2.VideoCapture(0)
recorder = WaterDataRecorder(output_dir="results")

# Debug: Show the mask to verify color detection before analyzing contours
SHOW_MASK = False

# Toggle to enable/disable the tracking ROI (Region of Interest)
USE_ROI = False

# Tracking region size (pixels)
ROI_SIZE = 500

# Minimum contour area to consider (to filter out noise)
MIN_CONTOUR_AREA = 500

# CLAHE brightness normalisation
cl = clahe_cfg(cfg)
_clahe = cv2.createCLAHE(
    clipLimit=cl["clip_limit"],
    tileGridSize=tuple(cl["grid_size"]),
) if cl["enabled"] else None

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(os.path.join(recorder.output_dir, 'video.mp4'), fourcc, 20.0, (width, height))

def draw_outlined_text(img, text, pos, font, scale, color, thickness):
    """Draw text with a dark outline so it's visible on any background."""
    # Draw black outline
    cv2.putText(img, text, pos, font, scale, (0, 0, 0), thickness + 3)
    # Draw foreground text
    cv2.putText(img, text, pos, font, scale, color, thickness)

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w = frame.shape[:2]

    out.write(frame)

    if USE_ROI:
        # Define the tracking ROI (centered in the frame)
        roi_x1 = w // 2 - ROI_SIZE // 2
        roi_y1 = h // 2 - ROI_SIZE // 2
        roi_x2 = roi_x1 + ROI_SIZE
        roi_y2 = roi_y1 + ROI_SIZE

        # Draw the tracking rectangle on the frame
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
    else:
        # No ROI: use full frame as ROI and set offsets to zero so downstream code can always use roi_* variables
        roi_x1, roi_y1 = 0, 0
        roi_x2, roi_y2 = w, h

    # 1. Crop the ROI and convert to CIELAB
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2] if USE_ROI else frame
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    # CLAHE: normalise L channel for brightness consistency
    if _clahe is not None:
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = _clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])

    # 2. Define color range in LAB (Example: Green)
    # L: 30-200 (medium to bright), a: 0-115 (negative a = green), b: 130-220 (slightly yellow-green)
    lower_green = np.array([30, 0, 100])
    upper_green = np.array([255, 115, 200])
    
    # 3. Create a mask and find color pixel coordinates within the ROI
    mask = cv2.inRange(lab, lower_green, upper_green)

    if SHOW_MASK:
        cv2.imshow('Mask', mask) # Debug: Show the mask to verify color detection
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue # Skip analysis for now to focus on mask debugging

    #points = np.column_stack(np.where(mask > 0)) # Get (y, x) of all matches
    total_pixels = 0
    spread_factor = 0.0

    # 4. Find contours (groups of connected color pixels)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Build a new mask containing ONLY pixels that belong to valid groups
    #    This ensures scattered noise pixels are excluded from amount & spread
    valid_mask = np.zeros_like(mask)

    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA: # Filter out small contours (noise)

            # Fill this contour on the valid mask (thickness=-1 fills the interior)
            cv2.drawContours(valid_mask, [cnt], -1, 255, thickness=-1)

            # Draw the green outline on the display frame (offset to full-frame coords)
            cnt_offset = cnt + np.array([roi_x1, roi_y1])
            cv2.drawContours(frame, [cnt_offset], -1, (0, 255, 0), 2)
            
            # Count actual color pixels inside this specific contour
            # Create a single-contour mask to isolate just this group
            single_mask = np.zeros_like(mask)
            cv2.drawContours(single_mask, [cnt], -1, 255, thickness=-1)
            # Bitwise AND with the original color mask to get only the real color pixels
            group_pixels = cv2.countNonZero(cv2.bitwise_and(mask, single_mask))

            # Label: find the centroid of the contour and draw the pixel count there
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + roi_x1
                cy = int(M["m01"] / M["m00"]) + roi_y1
                label = f"{group_pixels}px"
                # Center the text on the centroid
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                draw_outlined_text(frame, label, (cx - tw // 2, cy + th // 2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 6. Compute amount & spread using ONLY valid-group pixels
    #    Bitwise AND: must be in the original color mask AND inside a valid contour
    filtered_mask = cv2.bitwise_and(mask, valid_mask)
    points = np.column_stack(np.where(filtered_mask > 0))

    if len(points) > 0:
        total_pixels = len(points)
        std_dev = np.std(points, axis=0)
        spread_factor = np.mean(std_dev)

    # Record this frame's data for graphing
    recorder.record_frame(filtered_mask, total_pixels, spread_factor)

    # Display Data (outlined text for visibility)
    elapsed_str = f"{recorder.duration:.1f}s" if recorder.has_data else "—"
    draw_outlined_text(frame, f"Amount: {total_pixels}px", (10, 30), 1, 1.5, (255, 255, 255), 2)
    draw_outlined_text(frame, f"Spread: {spread_factor:.2f}", (10, 60), 1, 1.5, (255, 255, 255), 2)
    draw_outlined_text(frame, f"Time: {elapsed_str}", (10, 90), 1, 1.5, (255, 255, 255), 2)

    cv2.imshow('Color Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
out.release()

# Save all recorded data as graphs
recorder.save_graphs()
