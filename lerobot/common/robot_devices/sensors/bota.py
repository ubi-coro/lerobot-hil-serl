import os
import serial
import struct
import threading
import time
from collections import deque

import torch
import torch.nn as nn

from crc import Calculator, Configuration

from lerobot.common.robot_devices.utils import busy_wait


class BotaForceTorqueSensor:
    def __init__(self, port: str, calibration_cfg: dict | None = None):
        if calibration_cfg is None:
            calibration_cfg = dict()

        self.port = port
        self._ser = serial.Serial()
        self._thread = None
        self._stop_event = threading.Event()
        self.is_connected = False

        self._fx = 0.0
        self._fy = 0.0
        self._fz = 0.0
        self._mx = 0.0
        self._my = 0.0
        self._mz = 0.0
        self._status = None
        self._timestamp = 0.0
        self._temperature = 0.0

        self.FRAME_HEADER = b'\xAA'
        self.crc_config = Configuration(16, 0x1021, 0xFFFF, 0xFFFF, True, True)
        self.crc_calculator = Calculator(self.crc_config)

        self.robot = None
        self.calibration_model = None
        self.calibration_history = deque(maxlen=calibration_cfg.get("n_obs_history", 2))
        self.calibration_motors = calibration_cfg.get("motors", None)
        self.calibration_cfg = calibration_cfg

        self.calibration_data_collection_time = calibration_cfg.get("t_data_collection_s", 10.0)

        self.calibration_name = None

    def connect(self):
        if self.is_connected:
            raise RuntimeError(f"BotaForceTorqueSensor({self.port}) is already connected.")

        self._ser.port = self.port
        self._ser.baudrate = 460800
        self._ser.timeout = 0.01

        self._ser.open()
        print(f"Opened serial port {self.port}")

        self._ser.reset_input_buffer()
        init_msg = self._ser.read(100)
        print(f"Initial bytes from sensor: {init_msg}")

        self._thread = threading.Thread(target=self._read_loop)
        self._thread.daemon = True
        self._thread.start()

        self.is_connected = True

    def _read_loop(self):
        while not self._stop_event.is_set():
            header = self._ser.read(1)
            if header != self.FRAME_HEADER:
                continue

            data = self._ser.read(34)
            crc_bytes = self._ser.read(2)
            if len(data) != 34 or len(crc_bytes) != 2:
                continue

            expected_crc = struct.unpack_from('H', crc_bytes)[0]
            actual_crc = self.crc_calculator.checksum(data)
            if expected_crc != actual_crc:
                continue

            self._status = struct.unpack_from('H', data, 0)[0]
            self._fx = struct.unpack_from('f', data, 2)[0]
            self._fy = struct.unpack_from('f', data, 6)[0]
            self._fz = struct.unpack_from('f', data, 10)[0]
            self._mx = struct.unpack_from('f', data, 14)[0]
            self._my = struct.unpack_from('f', data, 18)[0]
            self._mz = struct.unpack_from('f', data, 22)[0]
            self._timestamp = struct.unpack_from('I', data, 26)[0]
            self._temperature = struct.unpack_from('f', data, 30)[0]

    def _get_history_tensor(self):
        if len(self.calibration_history) < self.calibration_history.maxlen:
            return [0.0] * (self.calibration_history.maxlen * len(self.calibration_motors))
        return list(sum(self.calibration_history, []))

    def read(self):
        if not self.is_connected:
            raise RuntimeError("Sensor is not connected. Call `connect()` first.")

        if self.calibration_model:
            input_tensor = torch.tensor(self._get_history_tensor(), dtype=torch.float32).unsqueeze(0)
            correction = self.calibration_model(input_tensor).detach().numpy().flatten()

            return {
                "force": (self._fx - correction[0], self._fy - correction[1], self._fz - correction[2]),
                "torque": (self._mx - correction[3], self._my - correction[4], self._mz - correction[5]),
                "status": self._status,
                "timestamp": self._timestamp,
                "temperature": self._temperature
            }
        else:
            return {
                "force": (self._fx, self._fy, self._fz),
                "torque": (self._mx, self._my, self._mz),
                "status": self._status,
                "timestamp": self._timestamp,
                "temperature": self._temperature
            }

    def activate_calibration(self, robot, name, folder, fps: int = 30):
        self.calibration_name = name
        self.robot = robot
        if self.calibration_motors is None:
            self.calibration_motors = robot.motor_names

        model_path = os.path.join(folder, f"{name}_bota_{fps}fps.pt")
        if os.path.exists(model_path):
            self.calibration_model = torch.load(model_path, weights_only=False)
            self.calibration_model.eval()
        else:
            self.run_calibration(robot, name, folder, fps=fps)


    def run_calibration(self, robot, name, folder, fps: int = 30):
        t_data_collection_s = self.calibration_cfg.get("t_data_collection_s", 10.0)
        num_epochs = self.calibration_cfg.get("num_epochs", 3000)
        batch_size = self.calibration_cfg.get("batch_size", 64)

        print(f"Running calibration for sensor {name} for {t_data_collection_s:.1f} s with {fps:.1f} Hz, move the arm in many different configurations...")
        history = deque(maxlen=2)
        X, Y = [], []

        input("Press <enter> to start teleoperation:")
        start_time = time.time()
        while time.time() - start_time < t_data_collection_s:
            start_loop_time = time.perf_counter()
            robot.teleop_step(record_data=False)
            q = robot.follower_arms[name].read("Present_Position", self.calibration_motors)
            history.append(list(q))
            if len(history) == history.maxlen:
                x = sum(history, [])
                f = self.read()
                y = list(f["force"]) + list(f["torque"])
                X.append(x)
                Y.append(y)
            busy_wait(1.0 / fps - (time.perf_counter() - start_loop_time))

        print("Collected calibration data. Fitting model...")

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        # Train/test split
        n = int(0.9 * len(X))  # this is temporally correlated, therefore the test dataset is more ood than random samples
        X_train, X_test = X[:n], X[n:]
        Y_train, Y_test = Y[:n], Y[n:]

        # Normalize
        x_mean, x_std = X_train.mean(0), X_train.std(0) + 1e-8
        y_mean, y_std = Y_train.mean(0), Y_train.std(0) + 1e-8
        Xn_train = (X_train - x_mean) / x_std
        Xn_test = (X_test - x_mean) / x_std
        Yn_train = (Y_train - y_mean) / y_std
        Yn_test = (Y_test - y_mean) / y_std

        # Model
        model = nn.Sequential(
            nn.Linear(X.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
        loss_fn = nn.MSELoss()

        for epoch in range(num_epochs):
            idx = torch.randperm(len(Xn_train))
            for i in range(0, len(Xn_train), batch_size):
                xb = Xn_train[idx[i:i + batch_size]]
                yb = Yn_train[idx[i:i + batch_size]]
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
            if epoch % 25 == 0:
                with torch.no_grad():
                    test_loss = loss_fn(model(Xn_test), Yn_test).item()
                print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}, Test Loss: {test_loss:.6f}")

        wrapped = WrappedModel(model, x_mean, x_std, y_mean, y_std)
        wrapped.eval()

        save_path = os.path.join(folder, f"{name}_bota_{fps}fps.pt")
        torch.save(wrapped, save_path)
        print(f"Calibration model saved to {save_path}")

        self.calibration_model = wrapped

    def disconnect(self):
        if not self.is_connected:
            return

        self._stop_event.set()
        self._thread.join()
        self._ser.close()
        self.is_connected = False

    def __del__(self):
        self.disconnect()


# Wrap model with normalization
class WrappedModel(nn.Module):
    def __init__(self, model, x_mean, x_std, y_mean, y_std):
        super().__init__()
        self.model = model
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def forward(self, x):
        x_norm = (x - self.x_mean) / self.x_std
        y_pred = self.model(x_norm)
        return y_pred * self.y_std + self.y_mean


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import logging

    from lerobot.common.robot_devices.robots.factory import make_robot
    from lerobot.common.utils.utils import init_hydra_config

    robot_cfg = init_hydra_config("../../../configs/robot/aloha-mr.yaml")
    robot = make_robot(robot_cfg)
    robot.connect()

    plt.ion()
    fig, ax = plt.subplots()
    forces_history = np.zeros((100, 3))
    line_fx, = ax.plot(forces_history[:, 0], label='Fx')
    line_fy, = ax.plot(forces_history[:, 1], label='Fy')
    line_fz, = ax.plot(forces_history[:, 2], label='Fz')
    ax.legend()
    ax.set_ylim([-50, 50])
    ax.set_title("Forces from Bota Sensor")

    try:
        while True:
            start_loop_time = time.perf_counter()

            robot.teleop_step()
            print(f"Teleoperation:                  took {1000 * (time.perf_counter() - start_loop_time):.2f} ms, "
                  f"{100 * (time.perf_counter() - start_loop_time) / (1.0 / robot.config.fps):.0f}% of the cycle")

            force = robot.botas["main"].read()["force"]

            print(f"With bota read and calibration: took {1000 * (time.perf_counter() - start_loop_time):.2f} ms, "
                         f"{100 * (time.perf_counter() - start_loop_time) / (1.0 / robot.config.fps):.0f}% of the cycle")

            new_row = np.array([list(force)])
            forces_history = np.roll(forces_history, -1, axis=0)
            forces_history[-1, :] = new_row

            line_fx.set_ydata(forces_history[:, 0])
            line_fy.set_ydata(forces_history[:, 1])
            line_fz.set_ydata(forces_history[:, 2])

            ax.relim()
            ax.autoscale_view()
            plt.pause(0.005)

            print(f"With plotting:                  took {1000 * (time.perf_counter() - start_loop_time):.2f} ms, "
                         f"{100 * (time.perf_counter() - start_loop_time) / (1.0 / robot.config.fps):.0f}% of the cycle")
            busy_wait(1.0 / robot.config.fps - (time.perf_counter() - start_loop_time))

    except KeyboardInterrupt:
        print('Interrupted by user.')