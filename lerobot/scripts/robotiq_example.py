import socket
import struct
import time

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
GRIPPER_IP   = "172.22.22.2"   # The Robotiq’s Modbus‐TCP address (default under URCap)
GRIPPER_PORT = 30002
UNIT_ID      = 0x09            # Robotiq’s fixed unit ID in the URCap setup
REG_STATUS   = 0x07D0          # Starting register for status (gOBJ, gSTA, gPOS, gCUR, gFLT)
NUM_WORDS    = 3               # We’ll read 3 registers = 6 bytes (gOBJ+gSTA, gPOS, gCUR+gFLT)

# ── HELPER TO BUILD A “Read Holding Registers” PACKET ─────────────────────────
def mb_read_pack(unit_id: int, start_addr: int, count: int) -> bytes:
    """
    Build a Modbus‐TCP ADU (Application Data Unit) that requests 'count' words
    starting at 'start_addr' (function code 3).
    """
    function_code = 3  # Read Holding Registers
    # PDU = [ function (1B) | start_addr (2B) | count (2B) ]
    pdu = struct.pack(">BHH", function_code, start_addr, count)
    # ADU header: [ TxID (2B)=0 | ProtoID (2B)=0 | Length (2B) = len(pdu)+1 | UnitID (1B) ]
    header = struct.pack(">HHHB", 0, 0, len(pdu) + 1, unit_id)
    return header + pdu

# ── MAIN LOOP: CONNECT & POLL STATUS ────────────────────────────────────────────
def read_gripper_status():
    # 1) Open TCP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        sock.connect((GRIPPER_IP, GRIPPER_PORT))
    except Exception as e:
        print(f"❌ Cannot connect to {GRIPPER_IP}:{GRIPPER_PORT} → {e}")
        return None

    # 2) Build & send a “Read Holding Registers” request for REG_STATUS
    request = mb_read_pack(UNIT_ID, REG_STATUS, NUM_WORDS)
    sock.sendall(request)

    # 3) Receive the response (should be header + byte count + data)
    try:
        raw = sock.recv(64)
    except socket.timeout:
        print("❌ Timeout waiting for response.")
        sock.close()
        return None
    sock.close()

    # 4) Parse the reply:
    #    [ TxID (2) | ProtoID (2) | Length (2) | UnitID (1) | Func(1)=3 | ByteCnt(1) | Data(N) ]
    if len(raw) < 9 or raw[7] != 3:
        print("❌ Unexpected or invalid response:", raw)
        return None

    byte_count = raw[8]
    data = raw[9 : 9 + byte_count]
    # Data layout for 3 words (6 bytes):
    #   word0 high→low: [ gSTA (high byte), gOBJ (low byte) ]
    #   word1 high→low: [ gPOS (high byte), gPOS (low byte) ]
    #   word2 high→low: [ gCUR (high byte), gFLT (low byte) ]
    gSTA = data[0]
    gOBJ = data[1]
    gPOS = (data[2] << 8) | data[3]
    gCUR = data[4]
    gFLT = data[5]

    # Convert position “steps” (0–255) to mm (0–85 mm open):
    width_mm = 85.0 * (255 - gPOS) / 255.0

    return {
        "gOBJ": gOBJ,
        "gSTA": gSTA,
        "gPOS_steps": gPOS,
        "width_mm": width_mm,
        "gCUR": gCUR,
        "gFLT": gFLT,
    }

if __name__ == "__main__":
    # Example: Poll once per second for 10 seconds
    for i in range(10):
        status = read_gripper_status()
        if status:
            print(f"[{i:02d}] OBJ={status['gOBJ']}, STA={status['gSTA']}, "
                  f"POS={status['gPOS_steps']} (≈{status['width_mm']:.1f} mm), "
                  f"CUR={status['gCUR']}, FLT={status['gFLT']}")
        time.sleep(1.0)
