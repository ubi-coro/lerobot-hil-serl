# Teleoperation Migration Guide

## Alte vs. Neue Teleoperation

### ⚠️ Wichtige Änderung: Kalibrierungs-System

Das Projekt hat von der **alten Hardware-Profile-Kalibrierung** auf ein **neues Kalibrierungs-System** umgestellt:

- **Alte Kalibrierung**: Nutzte `homing_offset` und `drive_mode` Arrays in Hardware-Profiles
- **Neue Kalibrierung**: Nutzt motor-spezifische Kalibrierungsfiles (z.B. `left_v3.json`, `right_v3.json`)

Die neue **Experiment-basierte Teleoperation** migriert automatisch zwischen beiden Systemen mit:
- `MigrateCalibrationObsProcessorStep` - Konvertiert Observations von neuem zu altem Format
- `MigrateInterventionActionProcessorStep` - Konvertiert Actions zwischen beiden Formaten

---

## 🔧 Alte CLI-Teleoperation (Hardware-Profiles)

```bash
# Einfaches bimanual Setup ohne Kameras
python lerobot_teleoperate.py \
  --robot.type=bi_viperx \
  --robot.left_arm_port=/dev/ttyDXL_follower_left \
  --robot.right_arm_port=/dev/ttyDXL_follower_right \
  --robot.id=bi_viperx \
  --teleop.type=bi_widowx \
  --teleop.left_arm_port=/dev/ttyDXL_leader_left \
  --teleop.right_arm_port=/dev/ttyDXL_leader_right \
  --teleop.id=bi_widowx

# Mit Kameras und Rerun Visualisierung
python lerobot_teleoperate.py \
  --robot.type=bi_viperx \
  --robot.left_arm_port=/dev/ttyDXL_follower_left \
  --robot.right_arm_port=/dev/ttyDXL_follower_right \
  --robot.id=bi_viperx \
  --robot.cameras="{
    right: {type: intelrealsense, serial_number_or_name: '130322274116', width: 640, height: 480, fps: 30},
    left: {type: intelrealsense, serial_number_or_name: '218622276088', width: 640, height: 480, fps: 30},
    top: {type: intelrealsense, serial_number_or_name: '218722270994', width: 640, height: 480, fps: 30},
    front: {type: intelrealsense, serial_number_or_name: '130322272007', width: 640, height: 480, fps: 30}
  }" \
  --teleop.type=bi_widowx \
  --teleop.left_arm_port=/dev/ttyDXL_leader_left \
  --teleop.right_arm_port=/dev/ttyDXL_leader_right \
  --teleop.id=bi_widowx \
  --display_data=true
```

**Problem**: Nutzt alte Kalibrierung, nicht kompatibel mit neuen v3 Calibration Files!

---

## ✅ Neue Experiment-basierte Teleoperation (Empfohlen)

### Installation

Nach Änderungen an `pyproject.toml`:

```bash
pip install -e .
```

### Verwendung mit Experiment-Config

Erstelle ein Experiment-Config (z.B. `src/experiments/aloha_bimanual_lemgo_v2/config.py`):

```python
from dataclasses import dataclass
from lerobot.envs.configs import EnvConfig, HilSerlRobotEnvConfig
from lerobot.robots.viperx import ViperXConfig
from lerobot.teleoperators.widowx import WidowXConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

@dataclass
@EnvConfig.register_subclass("aloha_bimanual_lemgo_v2")
class AlohaBimanualEnvConfigLemgoV2(HilSerlRobotEnvConfig):
    def __post_init__(self):
        self.robot = {
            "left": ViperXConfig(
                port="/dev/ttyDXL_follower_left",
                id="left_v3"  # ← Nutzt neue Kalibrierung!
            ),
            "right": ViperXConfig(
                port="/dev/ttyDXL_follower_right",
                id="right_v3"  # ← Nutzt neue Kalibrierung!
            )
        }
        self.teleop = {
            "left": WidowXConfig(
                port="/dev/ttyDXL_leader_left",
                id="left_v3",
                use_aloha2_gripper_servo=True
            ),
            "right": WidowXConfig(
                port="/dev/ttyDXL_leader_right",
                id="right_v3",
                use_aloha2_gripper_servo=True
            )
        }
        self.cameras = {
            "cam_low": RealSenseCameraConfig(
                serial_number_or_name="130322272007",
                fps=30, width=640, height=480,
            ),
            "cam_top": RealSenseCameraConfig(
                serial_number_or_name="218722270994",
                fps=30, width=640, height=480,
            ),
            "cam_right_wrist": RealSenseCameraConfig(
                serial_number_or_name="130322274116",
                fps=30, width=640, height=480,
            ),
            "cam_left_wrist": RealSenseCameraConfig(
                serial_number_or_name="218622276088",
                fps=30, width=640, height=480,
            )
        }
```

### Teleoperation starten

```bash
# Einfachste Version - nutzt alle Parameter aus dem Experiment-Config
lerobot-teleoperate-exp \
  --env.type=aloha_bimanual_lemgo_v2 \
  --display_data=true

# Mit Parameter-Overrides (falls nötig)
lerobot-teleoperate-exp \
  --env.type=aloha_bimanual_lemgo_v2 \
  --env.robot.left.port=/dev/ttyDXL_follower_left \
  --env.robot.right.port=/dev/ttyDXL_follower_right \
  --env.teleop.left.port=/dev/ttyDXL_leader_left \
  --env.teleop.right.port=/dev/ttyDXL_leader_right \
  --fps=30 \
  --display_data=true
```

---

## 📊 Vergleich

| Feature | Alte CLI (`lerobot-teleoperate`) | Neue Experiment (`lerobot-teleoperate-exp`) |
|---------|-----------------------------------|---------------------------------------------|
| **Kalibrierung** | Hardware-Profiles (alte Kalibrierung) | Neue v3 Calibration Files |
| **Config-System** | CLI-Parameter + TOML | Experiment-Configs (Python) |
| **Kalibrierungs-Migration** | ❌ Nein | ✅ Ja (automatisch) |
| **Processor-Pipeline** | Einfache Default-Processor | Volle Processor-Pipeline mit Event-Handling |
| **Kompatibilität mit Recording** | Unterschiedlich | ✅ Identisch (`lerobot-record-exp`) |
| **Intervention-Handling** | Basic | ✅ Advanced (Success-Events, Re-recording, etc.) |

---

## 🎯 Empfehlung

**Nutze die neue Experiment-basierte Teleoperation**, weil:

1. ✅ **Kompatibel mit neuen v3 Calibration Files** (`left_v3.json`, `right_v3.json`)
2. ✅ **Identisches Setup** wie beim Recording (`lerobot-record-exp`)
3. ✅ **Automatische Kalibrierungs-Migration** für alte Policies
4. ✅ **Erweiterte Features**: Success-Events, Episode-Termination, Intervention-Tracking
5. ✅ **Einfachere Konfiguration** durch Experiment-Configs

### Migration in 3 Schritten:

1. **Erstelle dein Experiment-Config** in `src/experiments/my_setup/config.py`
2. **Setze `id` auf neue Calibration** (z.B. `left_v3` statt `left`)
3. **Nutze `lerobot-teleoperate-exp`** statt `lerobot-teleoperate`

---

## 🐛 Troubleshooting

### Problem: "Calibration file not found"

```bash
# Stelle sicher, dass deine Calibration Files existieren:
ls ~/.cache/lerobot/calibration/left_v3.json
ls ~/.cache/lerobot/calibration/right_v3.json
```

Falls nicht vorhanden, führe Kalibrierung aus:

```bash
lerobot-calibrate \
  --robot.type=viperx \
  --robot.port=/dev/ttyDXL_follower_left \
  --robot.id=left_v3
```

### Problem: "Robot moves erratically"

Die Kalibrierungs-Migration kann bei falschen IDs fehlschlagen. Stelle sicher:
- `robot.id` = Name deines Calibration Files (ohne `.json`)
- Calibration File wurde mit gleicher ID erstellt

---

## 📝 Zusammenfassung

**Alter Befehl:**
```bash
python lerobot_teleoperate.py --robot.type=bi_viperx --robot.left_arm_port=/dev/ttyDXL_follower_left ...
```

**Neuer Befehl:**
```bash
lerobot-teleoperate-exp --env.type=aloha_bimanual_lemgo_v2 --display_data=true
```

**Vorteil:** Alles in einem Experiment-Config, automatische Kalibrierungs-Migration, identisch mit Recording-Setup! 🎉
