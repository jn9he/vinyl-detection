import os
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from datetime import datetime
import argparse


class CameraApp:
    def __init__(self, output_dir="snapshots", cam_index=0, window_title="Camera Snapshot"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.cam_index = cam_index
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {self.cam_index}")

        self.root = tk.Tk()
        self.root.title(window_title)

        # Video frame label
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        # Control frame with buttons
        controls = tk.Frame(self.root)
        controls.pack(fill=tk.X, padx=8, pady=6)

        self.snap_button = tk.Button(controls, text="Snapshot", command=self.take_snapshot)
        self.snap_button.pack(side=tk.LEFT)

        self.quit_button = tk.Button(controls, text="Quit", command=self.close)
        self.quit_button.pack(side=tk.RIGHT)

        # Bind keyboard shortcuts
        self.root.bind('<space>', lambda e: self.take_snapshot())
        self.root.bind('<q>', lambda e: self.close())

        # Start update loop
        self._running = True
        self.update_frame()

        # Graceful close on window X
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def update_frame(self):
        if not self._running:
            return
        ret, frame = self.cap.read()
        if not ret:
            # show an error and stop
            messagebox.showerror("Camera error", "Failed to read frame from webcam")
            self.close()
            return

        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Resize to fit a reasonable window size while preserving aspect
        max_width = 800
        if image.width > max_width:
            ratio = max_width / float(image.width)
            new_size = (max_width, int(image.height * ratio))
            image = image.resize(new_size, Image.ANTIALIAS)

        imgtk = ImageTk.PhotoImage(image=image)
        self.video_label.imgtk = imgtk  # keep reference to avoid GC
        self.video_label.configure(image=imgtk)

        # schedule next frame
        self.root.after(15, self.update_frame)

    def take_snapshot(self):
        # Grab a frame directly from capture to ensure highest resolution
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Snapshot error", "Failed to capture image from webcam")
            return

        # Use timestamped filename
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(self.output_dir, f"snapshot_{ts}.jpg")

        # Write with OpenCV (BGR)
        try:
            cv2.imwrite(fname, frame)
            print(f"Saved snapshot: {fname}")
            # small feedback to the user
            messagebox.showinfo("Snapshot saved", f"Saved snapshot to:\n{fname}")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save snapshot: {e}")

    def close(self):
        self._running = False
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Simple webcam snapshot GUI")
    parser.add_argument("--output", "-o", dest="output_dir", default="snapshots",
                        help="Folder to save snapshots")
    parser.add_argument("--cam", dest="cam_index", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()

    try:
        app = CameraApp(output_dir=args.output_dir, cam_index=args.cam_index)
        app.root.mainloop()
    except Exception as e:
        print(f"Error launching camera app: {e}")


if __name__ == "__main__":
    main()
