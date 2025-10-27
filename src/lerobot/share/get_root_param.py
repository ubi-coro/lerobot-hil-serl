from pathlib import Path

base_folder = "/media/nvme1/jstranghoener/lerobot/data/jannick-st/hoodie_folding"

out_str = "["
for sub in sorted(Path(base_folder).iterdir()):
    out_str += sub.parent.name + "/" + sub.name + ","
out_str = out_str[:-1] + "]"
print(out_str)

