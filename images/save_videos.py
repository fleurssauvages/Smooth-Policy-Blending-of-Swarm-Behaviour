import imageio.v2 as imageio
import glob

# Load frames (sorted!)
files = sorted(glob.glob("frames/frame_*.png"))
folder = "images"
out_name = "ppo_wall2"  # Change this to match your saved frames

# --- MP4 ---
# writer = imageio.get_writer(folder + "/" + out_name+".mp4", fps=24, codec="libx264", quality=8)
# for f in files:
#     writer.append_data(imageio.imread(f))
# writer.close()

# --- GIF ---
gif_writer = imageio.get_writer(folder + "/" + out_name+".gif", mode="I", fps=24)
for f in files:
    gif_writer.append_data(imageio.imread(f))
gif_writer.close()