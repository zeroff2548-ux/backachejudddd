# -*- coding: utf-8 -*-
"""
Posture Shimeji  -  มาสคอตเตือนนั่งหลังค่อม (Windows)

  * ใช้ MediaPipe Pose (33 keypoints) ตัวเดียวกับโปรเจกต์ MindDock/motion-tracker
  * วิเคราะห์ท่านั่งจากเว็บแคม แล้วแปลงเป็น 4 สถานะ
        STATUS: normal | warning | bad | no_person
  * มาสคอตวิ่งเล่นบนหน้าจอ (สไตล์ Shimeji) และเปลี่ยนพฤติกรรมตามสถานะ
  * ทุกอย่างอยู่ในไฟล์เดียว  ->  build เป็น PostureShimeji.exe ไฟล์เดียวได้
"""
import json
import math
import os
import random
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import (QLockFile, QObject, QPoint, QPointF, QRect, QRectF,
                            Qt, QThread, QTimer, QUrl, Signal)
from PySide6.QtGui import (QAction, QActionGroup, QColor, QCursor,
                           QDesktopServices, QFont, QFontMetrics, QGuiApplication,
                           QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap,
                           QPolygonF)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                               QFileDialog, QFrame, QHBoxLayout, QInputDialog,
                               QLabel, QMenu, QMessageBox, QProgressBar,
                               QPushButton, QScrollArea, QSystemTrayIcon,
                               QVBoxLayout, QWidget)

APP_NAME = "PostureShimeji"
FROZEN = getattr(sys, "frozen", False)
APP_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / APP_NAME
BASE_DIR = Path(sys.executable).parent if FROZEN else Path(__file__).resolve().parent

STATUSES = ("normal", "warning", "bad", "no_person")

# ----------------------------------------------------------------------------
#  ตั้งค่าความไว (ปรับได้)
# ----------------------------------------------------------------------------
TARGET_FPS = 10            # ประมวลผลกล้องกี่เฟรมต่อวินาที (พอสำหรับท่านั่ง และประหยัดเครื่อง)
WARN_TH = 0.25             # คะแนนเริ่ม warning
BAD_TH = 0.50              # คะแนนเริ่ม bad
HYST = 0.06                # hysteresis กันสถานะกระพริบ
ENTER_DELAY = {            # ต้องอยู่ในท่านั้นต่อเนื่องกี่วินาทีถึงเปลี่ยนสถานะ
    "normal": 2.0,
    "warning": 4.0,
    "bad": 5.0,
    "no_person": 3.0,
}
SENS = {"low": 1.25, "normal": 1.0, "high": 0.8}

# ----------------------------------------------------------------------------
#  Utility
# ----------------------------------------------------------------------------
APP_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = APP_DIR / "log.txt"
STATUS_FILE = APP_DIR / "status.txt"


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + str(msg)
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 512 * 1024:
            LOG_FILE.unlink()
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def clamp01(v):
    return max(0.0, min(1.0, float(v)))


class Config:
    DEFAULTS = {"baseline": None, "sound": True, "camera": 0, "sensitivity": "normal",
                "tip_shown": False, "volume": "high", "use_custom": True}

    def __init__(self):
        self.path = APP_DIR / "config.json"
        self.data = dict(self.DEFAULTS)
        try:
            if self.path.exists():
                self.data.update(json.loads(self.path.read_text(encoding="utf-8")))
        except Exception as e:
            log(f"config load error: {e}")

    def __getitem__(self, k):
        return self.data.get(k, self.DEFAULTS.get(k))

    def __setitem__(self, k, v):
        self.data[k] = v
        self.save()

    def save(self):
        try:
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        except Exception as e:
            log(f"config save error: {e}")


# ----------------------------------------------------------------------------
#  1) วิเคราะห์ท่านั่ง  (ไม่ผูกกับ Qt / mediapipe -> ทดสอบง่าย)
# ----------------------------------------------------------------------------
NOSE, L_EAR, R_EAR, L_SH, R_SH = 0, 7, 8, 11, 12
DEFAULT_BASE = {"ratio": 0.62, "angle": 15.0, "scale": None, "tilt": 0.0}


def compute_metrics(img, world, w, h):
    """
    img   : ndarray (33,4)  x,y (0-1), z, visibility   -- pose_landmarks
    world : ndarray (33,4)  x,y,z (เมตร) หรือ None       -- pose_world_landmarks
    คืน dict ของ metric หรือ None ถ้าเห็นตัวไม่ชัดพอ
    """
    if img[L_SH, 3] < 0.45 or img[R_SH, 3] < 0.45:
        return None
    heads = [i for i in (L_EAR, R_EAR) if img[i, 3] >= 0.4]
    if not heads:
        if img[NOSE, 3] >= 0.5:
            heads = [NOSE]
        else:
            return None

    P = img[:, :2] * np.array([w, h], dtype=float)
    sh_vec = P[L_SH] - P[R_SH]
    sh_w = float(np.hypot(sh_vec[0], sh_vec[1]))
    if sh_w < 0.08 * w:          # ไกลเกินไป / เล็กเกินไป
        return None
    sh_mid = (P[L_SH] + P[R_SH]) / 2.0
    head = P[heads].mean(axis=0)

    ratio = float((sh_mid[1] - head[1]) / sh_w)       # หัวสูงกว่าไหล่กี่เท่าของความกว้างไหล่
    scale = sh_w / float(h)                            # ไหล่ใหญ่ขึ้น = โน้มตัวเข้าหาจอ
    tilt = float(math.degrees(math.atan2(sh_vec[1], sh_vec[0])))   # ไหล่เอียง

    angle = 0.0                                        # มุมคอยื่นไปข้างหน้า (องศา)
    if world is not None:
        wp = world[:, :3]
        d = wp[heads].mean(axis=0) - (wp[L_SH] + wp[R_SH]) / 2.0
        up = -d[1]
        forward = -d[2]                                # z น้อย = ใกล้กล้อง
        angle = float(math.degrees(math.atan2(forward, max(up, 1e-4))))

    return {"ratio": ratio, "angle": angle, "scale": float(scale), "tilt": tilt}


def score_posture(m, base=None):
    """คะแนน 0 (ท่าดีเท่าตอน calibrate) -> 1 (แย่มาก)"""
    b = dict(DEFAULT_BASE)
    if base:
        b.update({k: v for k, v in base.items() if v is not None})
    drop = (b["ratio"] - m["ratio"]) / max(b["ratio"], 1e-3)
    p_head = clamp01((drop - 0.04) / 0.18)                     # หัวจมลงเทียบไหล่
    p_fwd = clamp01((m["angle"] - b["angle"] - 6.0) / 20.0)    # คอยื่น
    p_scale = 0.0
    if b["scale"]:
        p_scale = clamp01((m["scale"] / b["scale"] - 1.04) / 0.20)   # โน้มเข้าหาจอ
    p_tilt = clamp01((abs(m["tilt"] - b["tilt"]) - 5.0) / 10.0)      # ไหล่เอียง
    score = clamp01(0.45 * p_head + 0.25 * p_fwd + 0.25 * p_scale + 0.15 * p_tilt)
    return score, {"head": p_head, "fwd": p_fwd, "scale": p_scale, "tilt": p_tilt}


def classify(score, current, k=1.0):
    w, b, hy = WARN_TH * k, BAD_TH * k, HYST * k
    if score >= (b - hy if current == "bad" else b):
        return "bad"
    if score >= (w - hy if current in ("warning", "bad") else w):
        return "warning"
    return "normal"


class StatusMachine:
    """กันสถานะกระพริบ: ต้องอยู่ในท่าใหม่ต่อเนื่องครบเวลาถึงจะเปลี่ยน"""

    def __init__(self, initial="no_person"):
        self.status = initial
        self.cand = None
        self.cand_t = 0.0

    def force(self, s):
        self.status, self.cand = s, None

    def update(self, raw, now):
        if raw == self.status:
            self.cand = None
            return self.status
        if raw != self.cand:
            self.cand, self.cand_t = raw, now
        el = now - self.cand_t
        delay = ENTER_DELAY[raw]
        if self.status == "no_person":
            delay = 1.0
        if raw == "bad" and self.status == "normal" and el >= ENTER_DELAY["warning"]:
            self.status = "warning"            # ไต่ระดับ: เตือนก่อน แล้วค่อย bad
            return self.status
        if el >= delay:
            self.status, self.cand = raw, None
        return self.status


# ----------------------------------------------------------------------------
#  2) Thread กล้อง + MediaPipe
# ----------------------------------------------------------------------------
def open_camera(idx):
    import cv2
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" \
        else [cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(idx, be)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


class Tracker(QThread):
    result = Signal(object)     # dict(valid, metrics)
    preview = Signal(QImage)
    calib = Signal(object)      # dict(phase, remaining, baseline)
    error = Signal(str)

    def __init__(self, cam_index=0):
        super().__init__()
        self.cam_index = cam_index
        self.paused = False
        self.want_preview = False
        self._stop_flag = False
        self._reopen = False
        self._calib_req = None

    def stop(self):
        self._stop_flag = True

    def reopen_camera(self, idx):
        self.cam_index = idx
        self._reopen = True

    def request_calibration(self, delay=5.0, duration=5.0):
        self._calib_req = (delay, duration)

    def run(self):
        try:
            import cv2
            import mediapipe as mp
        except Exception as e:                       # pragma: no cover
            self.error.emit(f"โหลด mediapipe/opencv ไม่ได้: {e}")
            return
        pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=1, smooth_landmarks=True,
            enable_segmentation=False, min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        cap, last_try, fails, last_err = None, 0.0, 0, 0.0
        calib = None
        interval = 1.0 / TARGET_FPS
        conns = [(L_SH, R_SH), (L_EAR, L_SH), (R_EAR, R_SH), (L_EAR, R_EAR),
                 (11, 23), (12, 24), (23, 24)]

        while not self._stop_flag:
            t0 = time.time()
            if self.paused or self._reopen:
                if cap is not None:
                    cap.release()
                    cap = None
                self._reopen = False
                if self.paused:
                    time.sleep(0.2)
                    continue
            if cap is None:
                if time.time() - last_try < 3.0:
                    self.result.emit({"valid": False, "metrics": None})
                    time.sleep(0.3)
                    continue
                last_try = time.time()
                cap = open_camera(self.cam_index)
                if cap is None:
                    if time.time() - last_err > 30:
                        self.error.emit("เปิดกล้องไม่ได้ (ถูกโปรแกรมอื่นใช้อยู่หรือยังไม่ได้อนุญาต)")
                        last_err = time.time()
                    self.result.emit({"valid": False, "metrics": None})
                    continue
                log("camera opened")
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 10:
                    cap.release()
                    cap = None
                    fails = 0
                time.sleep(0.05)
                continue
            fails = 0
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose.process(rgb)

            metrics, img = None, None
            if res.pose_landmarks:
                img = np.array([[l.x, l.y, l.z, l.visibility]
                                for l in res.pose_landmarks.landmark])
                world = None
                if res.pose_world_landmarks:
                    world = np.array([[l.x, l.y, l.z, l.visibility]
                                      for l in res.pose_world_landmarks.landmark])
                metrics = compute_metrics(img, world, w, h)
            valid = metrics is not None

            # ---- calibration ----
            now = time.time()
            if self._calib_req:
                d, dur = self._calib_req
                calib = {"t0": now, "delay": d, "dur": dur, "samples": []}
                self._calib_req = None
            if calib:
                el = now - calib["t0"]
                if el < calib["delay"]:
                    self.calib.emit({"phase": "wait", "remaining": calib["delay"] - el})
                elif el < calib["delay"] + calib["dur"]:
                    if valid:
                        calib["samples"].append(metrics)
                    self.calib.emit({"phase": "hold",
                                     "remaining": calib["delay"] + calib["dur"] - el})
                else:
                    sm = calib["samples"]
                    if len(sm) >= 8:
                        base = {k: float(np.median([s[k] for s in sm]))
                                for k in ("ratio", "angle", "scale", "tilt")}
                        self.calib.emit({"phase": "done", "baseline": base})
                    else:
                        self.calib.emit({"phase": "fail"})
                    calib = None

            self.result.emit({"valid": valid, "metrics": metrics})

            if self.want_preview:
                vis = frame.copy()
                if img is not None:
                    pts = (img[:, :2] * np.array([w, h])).astype(int)
                    for a, b in conns:
                        cv2.line(vis, tuple(pts[a]), tuple(pts[b]), (80, 220, 120), 2)
                    for i in (NOSE, L_EAR, R_EAR, L_SH, R_SH):
                        cv2.circle(vis, tuple(pts[i]), 5, (60, 140, 255), -1)
                vis = cv2.flip(vis, 1)
                vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                qi = QImage(vis.data, w, h, 3 * w, QImage.Format_RGB888).copy()
                self.preview.emit(qi)

            time.sleep(max(0.0, interval - (time.time() - t0)))

        if cap is not None:
            cap.release()
        pose.close()


# ----------------------------------------------------------------------------
#  3) วาดมาสคอต (วาดด้วยโค้ด ไม่ต้องมีไฟล์รูป) + รองรับรูป sprite ของคุณเอง
# ----------------------------------------------------------------------------
PALETTE = {
    "normal": ("#FFF6E5", "#5B4636"),
    "warning": ("#FFE7A0", "#6B4E16"),
    "bad": ("#FFB0A6", "#7A1F1A"),
    "no_person": ("#D5DFF2", "#3F4D6B"),
    "calibrating": ("#D4F1E1", "#2F5E49"),
}
CAT_FILL = {"normal": "#F9A97A", "warning": "#F9B98A", "bad": "#F27B5E",
            "no_person": "#E2B9A4", "calibrating": "#F9A97A"}
CAT_INK = "#2B1B17"
FONT_FAMILIES = ["Leelawadee UI", "Tahoma", "Segoe UI", "Noto Sans Thai", "Arial"]


def _pen(c, w):
    return QPen(c, w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)


def draw_mascot(p, status, frame, mode, direction=1, vy=0.0, sprite=None):
    """วาดตัวละคร โดย origin (0,0) = กึ่งกลางเท้า, แกน y ลบ = ขึ้นบน"""
    p.save()
    p.scale(direction, 1)
    fill_hex, line_hex = PALETTE.get(status, PALETTE["normal"])
    fill, line = QColor(fill_hex), QColor(line_hex)
    pen = _pen(line, 3)

    sleeping = (mode == "sleep") or status == "no_person"
    airborne = mode in ("fall", "drag")
    walking = mode in ("walk", "run")
    spd = 0.5 if mode == "walk" else 0.85
    ph = math.sin(frame * spd) if walking else 0.0
    ph2 = math.sin(frame * 0.35)

    bob = abs(ph) * 3.0 if walking else math.sin(frame * 0.09) * 1.3
    sy = 1.0
    if mode in ("fall", "jump"):
        sy = 1.0 + max(-0.10, min(0.18, -vy * 0.012))
    elif sleeping:
        sy = 1.0 + 0.03 * math.sin(frame * 0.08)
        bob = 0
    if status == "bad" and mode != "drag":
        p.translate(random.uniform(-1.5, 1.5), 0)
    p.scale(1.0 / sy, sy)
    p.translate(0, -bob)

    # ---------- sprite ของผู้ใช้ ----------
    if sprite is not None:
        h = 110.0
        w = h * sprite.width() / max(1, sprite.height())
        p.drawPixmap(QRectF(-w / 2, -h, w, h), sprite, QRectF(sprite.rect()))
        if sleeping:
            _draw_zzz(p, frame, line)
        p.restore()
        return

    p.setRenderHint(QPainter.Antialiasing)
    fill = QColor(CAT_FILL.get(status, CAT_FILL["normal"]))
    shade = fill.darker(125)
    light = fill.lighter(115)
    ink = QColor(CAT_INK)
    bad = status == "bad"
    warn = status == "warning"
    HY = -67                                   # กึ่งกลางหัว

    # ---------- หาง (ลายขวางสีส้มเข้ม) ----------
    wag = math.sin(frame * 0.18) * (10 if warn else 6)
    tail = QPainterPath(QPointF(-32, -18))
    tail.cubicTo(QPointF(-78, -14), QPointF(-84, -58 + wag), QPointF(-62, -84 + wag))
    p.setBrush(Qt.NoBrush)
    p.setPen(_pen(ink, 17))
    p.drawPath(tail)
    p.setPen(_pen(fill, 11))
    p.drawPath(tail)
    for t in (0.50, 0.66, 0.82):
        a, b, c = tail.pointAtPercent(t - 0.01), tail.pointAtPercent(t + 0.01), tail.pointAtPercent(t)
        dx, dy = b.x() - a.x(), b.y() - a.y()
        n = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / n * 5.2, dx / n * 5.2
        p.setPen(QPen(shade, 3.6, Qt.SolidLine, Qt.FlatCap))
        p.drawLine(QPointF(c.x() - nx, c.y() - ny), QPointF(c.x() + nx, c.y() + ny))

    # ---------- หู + ตัว + หัว (วาดเส้นขอบรวมก่อน แล้วค่อยลงสี ได้เป็นก้อนเดียวกัน) ----------
    droop = 8 if (warn or status == "no_person") else (4 if bad else 0)
    out = 7 if bad else 0
    ears = []
    for s in (-1, 1):
        ears.append(QPolygonF([QPointF(s * 43, -75), QPointF(s * (36 + out), -108 + droop + out),
                               QPointF(s * 10, -98)]))

    def silhouette():
        for e in ears:
            p.drawPolygon(e)
        p.drawEllipse(QPointF(0, -35), 46, 33)
        p.drawEllipse(QPointF(0, HY), 43, 32)

    p.setPen(_pen(ink, 6.5))
    p.setBrush(ink)
    silhouette()
    p.setPen(Qt.NoPen)
    p.setBrush(fill)
    silhouette()

    # หูด้านใน + ขีดบนหู
    for s in (-1, 1):
        p.setPen(Qt.NoPen)
        p.setBrush(shade)
        p.drawPolygon(QPolygonF([QPointF(s * 37, -78), QPointF(s * (34 + out * 0.8), -100 + droop + out),
                                 QPointF(s * 17, -92)]))
        p.setPen(_pen(ink, 1.6))
        d = droop + out * 0.8
        p.drawLine(QPointF(s * 31, -92 + d), QPointF(s * 35, -93 + d))
        p.drawLine(QPointF(s * 30, -87 + d), QPointF(s * 34, -88 + d))

    # ท้องสีอ่อน + ลายข้างตัว
    p.setPen(Qt.NoPen)
    belly = QColor(light)
    belly.setAlpha(200)
    p.setBrush(belly)
    p.drawEllipse(QPointF(0, -24), 27, 17)
    p.setPen(_pen(shade, 3.2))
    for s in (-1, 1):
        p.drawLine(QPointF(s * 40, -46), QPointF(s * 45, -43))
        p.drawLine(QPointF(s * 41, -38), QPointF(s * 46, -36))
        p.drawLine(QPointF(s * 40, -30), QPointF(s * 44, -29))

    # ---------- แขน ----------
    for s in (-1, 1):
        sh = QPointF(s * 37, -44)
        if mode == "climb":
            sh = QPointF(38, -44)
            hand = QPointF(54, -88 + ph2 * 8) if s > 0 else QPointF(52, -60 - ph2 * 8)
        elif airborne:
            hand = QPointF(s * 52, -76 + math.sin(frame * 0.8 + s) * 6)
        elif walking:
            hand = QPointF(s * 44, -26 + s * ph * 7)
        elif warn and not sleeping:
            hand = QPointF(s * 14, -30 + math.sin(frame * 0.5 + s) * 2)
        elif bad:
            hand = QPointF(s * 46, -28)
        else:
            hand = QPointF(s * 41, -22)
        p.setPen(_pen(ink, 12))
        p.drawLine(sh, hand)
        p.setPen(_pen(fill, 7.5))
        p.drawLine(sh, hand)
        if mode != "climb" and not airborne:
            p.setPen(_pen(ink, 1.8))
            for dx in (-2.4, 2.4):
                p.drawLine(QPointF(hand.x() + dx, hand.y() + 1), QPointF(hand.x() + dx, hand.y() + 5))

    # ---------- เท้า ----------
    p.setPen(_pen(ink, 2.8))
    p.setBrush(fill)
    for s in (-1, 1):
        lift = 0.0
        if walking:
            fx = s * 17 + s * ph * 9
            lift = max(0.0, s * math.cos(frame * spd)) * 5
        elif airborne or mode == "jump":
            fx = s * 15 + math.sin(frame * 0.5 + s) * 3
            lift = 3
        elif mode == "climb":
            fx = s * 15 + 8
            lift = 4 if s * ph2 > 0 else 0
        else:
            fx = s * 17
        p.drawEllipse(QPointF(fx, -6 - lift), 14, 6.5)

    # ---------- หน้า ----------
    ex, ey = 19, HY + 1
    lx = 2.4 + 1.6 * math.sin(frame * 0.03)       # สายตาเหลือบไปมาเบาๆ
    ly = -1.6

    def eye(x, y, rx, ry, pdx, pdy, pr):
        p.setPen(_pen(ink, 2.4))
        p.setBrush(QColor("#F8ECE9"))
        p.drawEllipse(QPointF(x, y), rx, ry)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(232, 196, 198, 200))
        p.drawEllipse(QPointF(x - 1.5, y - ry * 0.35), rx * 0.62, ry * 0.45)
        p.setBrush(QColor("#34201B"))
        p.drawEllipse(QPointF(x + pdx, y + pdy), pr, pr * 1.1)
        p.setBrush(QColor("white"))
        p.drawEllipse(QPointF(x + pdx - pr * 0.38, y + pdy - pr * 0.42), pr * 0.34, pr * 0.34)
        p.drawEllipse(QPointF(x + pdx + pr * 0.42, y + pdy + pr * 0.4), pr * 0.17, pr * 0.17)

    def closed_eyes(w=8.0):
        p.setPen(_pen(ink, 2.6))
        p.setBrush(Qt.NoBrush)
        for s in (-1, 1):
            p.drawArc(QRectF(s * ex - w, ey - 4, w * 2, 10), 200 * 16, 140 * 16)

    def nose():
        p.setPen(_pen(ink, 1.3))
        p.setBrush(QColor("#E98AA3"))
        p.drawPolygon(QPolygonF([QPointF(-3.6, -58.5), QPointF(3.6, -58.5), QPointF(0, -54.6)]))

    def open_mouth(w=6.0, h=6.0, y=-52.5):
        m = QPainterPath(QPointF(-w, y))
        m.quadTo(QPointF(0, y + 1.5), QPointF(w, y))
        m.cubicTo(QPointF(w - 1, y + h), QPointF(-w + 1, y + h), QPointF(-w, y))
        p.setPen(_pen(ink, 1.6))
        p.setBrush(QColor("#E5607F"))
        p.drawPath(m)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#FFA6BA"))
        p.drawEllipse(QPointF(0, y + h * 0.62), w * 0.5, h * 0.3)

    def blush(col):
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        for s in (-1, 1):
            p.drawEllipse(QPointF(s * 32, -56), 6.5, 3.8)

    def brows(inner_y, outer_y, w=3.0):
        p.setPen(_pen(ink, w))
        for s in (-1, 1):
            p.drawLine(QPointF(s * 30, outer_y), QPointF(s * 9, inner_y))

    def forehead_dots():
        p.setPen(_pen(ink, 1.3))
        p.setBrush(light)
        p.drawEllipse(QPointF(-3.5, -88), 1.7, 1.7)
        p.drawEllipse(QPointF(4.5, -89), 1.7, 1.7)

    if sleeping:
        closed_eyes()
        nose()
        p.setPen(_pen(ink, 1.8))
        p.setBrush(Qt.NoBrush)
        p.drawArc(QRectF(-4, -55, 8, 6), 200 * 16, 140 * 16)
        forehead_dots()
        _draw_zzz(p, frame, ink)
    elif airborne:
        for s in (-1, 1):
            eye(s * ex, ey, 13, 12.5, 0, 0, 4.2)
        nose()
        p.setPen(_pen(ink, 1.6))
        p.setBrush(QColor("#E5607F"))
        p.drawEllipse(QPointF(0, -48.5), 3.6, 4.6)
        forehead_dots()
    elif bad:
        for s in (-1, 1):
            eye(s * ex, ey + 1, 11.5, 8.5, s * -1.5, 1, 4.4)
        brows(-74, -83, 3.4)
        nose()
        p.setPen(_pen(ink, 1.6))
        p.setBrush(QColor("white"))
        p.drawRoundedRect(QRectF(-8, -53.5, 16, 7.5), 2.5, 2.5)
        p.drawLine(QPointF(-4, -53.5), QPointF(-4, -46))
        p.drawLine(QPointF(0, -53.5), QPointF(0, -46))
        p.drawLine(QPointF(4, -53.5), QPointF(4, -46))
        blush(QColor(255, 60, 60, 150))
        k = 1.0 + 0.2 * math.sin(frame * 0.5)          # เครื่องหมายโมโห
        p.setPen(_pen(QColor("#E53935"), 3))
        cx, cy = 40, -92
        p.drawLine(QPointF(cx - 3 * k, cy - 8 * k), QPointF(cx - 3 * k, cy + 8 * k))
        p.drawLine(QPointF(cx + 3 * k, cy - 8 * k), QPointF(cx + 3 * k, cy + 8 * k))
        p.drawLine(QPointF(cx - 8 * k, cy - 3 * k), QPointF(cx + 8 * k, cy - 3 * k))
        p.drawLine(QPointF(cx - 8 * k, cy + 3 * k), QPointF(cx + 8 * k, cy + 3 * k))
    elif warn:
        sx = 2.2 * math.sin(frame * 0.6)                # ตากวาดไปมาแบบกังวล
        for s in (-1, 1):
            eye(s * ex, ey, 12.5, 12, sx, 0, 4.6)
        brows(-82, -75, 2.8)
        nose()
        zz = QPainterPath(QPointF(-7, -50))
        for x, y in ((-3.5, -52.5), (0, -50), (3.5, -52.5), (7, -50)):
            zz.lineTo(x, y)
        p.setPen(_pen(ink, 2.2))
        p.setBrush(Qt.NoBrush)
        p.drawPath(zz)
        blush(QColor(255, 140, 140, 90))
        x, y = 36, -92 + math.sin(frame * 0.2) * 2     # เหงื่อ
        d = QPainterPath(QPointF(x, y - 9))
        d.cubicTo(QPointF(x + 7, y - 1), QPointF(x + 6, y + 6), QPointF(x, y + 6))
        d.cubicTo(QPointF(x - 6, y + 6), QPointF(x - 7, y - 1), QPointF(x, y - 9))
        p.setBrush(QColor("#8FD3FF"))
        p.setPen(_pen(QColor("#3E8FC4"), 1.5))
        p.drawPath(d)
        forehead_dots()
    else:   # normal / calibrating
        if (frame % 130) < 5:
            closed_eyes()
        else:
            for s in (-1, 1):
                eye(s * ex, ey, 12.5, 12, lx, ly, 6.6)
        nose()
        if status == "calibrating":
            p.setPen(_pen(ink, 2.2))
            p.drawLine(QPointF(-4, -51), QPointF(4, -51))
        else:
            open_mouth()
        blush(QColor(255, 120, 130, 90))
        forehead_dots()
    p.restore()


def _draw_zzz(p, frame, color):
    f = QFont("Arial")
    f.setBold(True)
    for i in range(3):
        t = (frame * 0.015 + i / 3.0) % 1.0
        f.setPointSizeF(8 + t * 8)
        p.setFont(f)
        c = QColor(color)
        c.setAlphaF(1.0 - t)
        p.setPen(c)
        p.drawText(QPointF(26 + t * 16, -90 - t * 34), "z")


def make_icon(status="normal", size=64):
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.translate(size / 2, size * 0.94)
    p.scale(size / 120.0, size / 120.0)
    draw_mascot(p, status, 0, "idle")
    p.end()
    return pm


# ----------------------------------------------------------------------------
#  4) ตัวมาสคอตบนเดสก์ท็อป
# ----------------------------------------------------------------------------
WIN_W, WIN_H = 300, 270
FOOT_PAD = 6
BW = 46        # ครึ่งความกว้างลำตัว (ใช้คำนวณชนขอบจอ)

BAD_MSGS = ["หลังค่อมแล้ว!\nนั่งตัวตรงๆ เดี๋ยวนี้!",
            "โอ๊ย ปวดหลังแทนเลย\nยืดตัวขึ้นหน่อย!",
            "ไหล่ห่อแล้วนะ!\nเงยหน้า ยืดอกหน่อย!"]
WARN_MSGS = ["เริ่มค่อมแล้วนะ\nยืดหลังหน่อย~",
             "ระวังหลังงอ!\nนั่งให้ตรงอีกนิด",
             "คอเริ่มยื่นแล้วนะ\nถอยหลังมาหน่อย"]
GOOD_MSGS = ["เยี่ยม! ท่านั่งดีแล้ว\nรักษาไว้นะ", "สุดยอด! หลังตรงเลย"]
POKE_MSGS = ["จิ้มทำไมเนี่ย~\nนั่งหลังตรงๆ ด้วยนะ", "เราคอยเฝ้าดูท่านั่งอยู่นะ!",
             "ดื่มน้ำแล้วหรือยัง?", "ลุกเดินสักนิดไหม?"]


class Mascot(QWidget):
    nag = Signal(str)

    def __init__(self, sprites=None):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                         | Qt.NoDropShadowWindowHint | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setMouseTracking(False)
        self.resize(WIN_W, WIN_H)

        self.set_sprites(sprites)

        self.menu = None
        self.status = "no_person"
        self.mode = "idle"
        self.dir = 1
        self.frame = 0
        self.timer = 0
        self.scale = 1.0
        self.target_scale = 1.0
        self.vx = self.vy = 0.0
        self.climb_target = 0.0
        self.next_nag = 0
        self.bubble_text, self.bubble_kind, self.bubble_ticks = "", "info", 0
        self.drag_off = QPoint()
        self.drag_hist = []
        self.press_pos = QPoint()

        geo = self._geo_at(QPoint(0, 0), primary=True)
        self.px = float(geo.right() - WIN_W - 80)
        self.py = float(self._ground(geo))
        self.move(int(self.px), int(self.py))

        self.clock = QTimer(self)
        self.clock.timeout.connect(self.tick)
        self.clock.start(33)

    def set_sprites(self, sprites):
        self.sprites = sprites or {}
        self.bw = BW
        if self.sprites:
            sp = next(iter(self.sprites.values()))
            self.bw = max(30, min(90, 110.0 * sp.width() / max(1, sp.height()) / 2))

    # ----- geometry helpers -----
    def _geo_at(self, c, primary=False):
        sc = None if primary else QGuiApplication.screenAt(c)
        sc = sc or QGuiApplication.primaryScreen()
        return sc.availableGeometry()

    def screen_geo(self):
        return self._geo_at(QPoint(int(self.px) + WIN_W // 2, int(self.py) + WIN_H // 2))

    @staticmethod
    def _ground(geo):
        return geo.bottom() + 1 - WIN_H + FOOT_PAD

    def x_limits(self, geo):
        bw = self.bw * self.scale
        return geo.left() - (WIN_W / 2 - bw), geo.right() + 1 - (WIN_W / 2 + bw)

    # ----- public API -----
    def say(self, text, kind="info", ticks=150):
        self.bubble_text, self.bubble_kind, self.bubble_ticks = text, kind, ticks
        self.update()

    def clear_bubble(self):
        self.bubble_text, self.bubble_ticks = "", 0

    def set_status(self, status):
        if status == self.status:
            return
        prev, self.status = self.status, status
        self.target_scale = 1.4 if status == "bad" else 1.0
        if self.mode == "climb":
            self.start_fall(-self.dir * 2.0, -2.0)
        self.timer = 0
        self.clear_bubble()
        if status == "bad":
            self.say(random.choice(BAD_MSGS), "bad", -1)
            self.next_nag = self.frame + 600
            self.nag.emit("bad")
        elif status == "warning":
            self.say(random.choice(WARN_MSGS), "warn", 120)
            self.next_nag = self.frame + 360
            self.nag.emit("warn")
        elif status == "normal" and prev in ("warning", "bad"):
            self.say(random.choice(GOOD_MSGS), "info", 120)

    # ----- behaviour -----
    def start_fall(self, vx=0.0, vy=0.0):
        self.mode, self.vx, self.vy = "fall", vx, vy

    def choose_action(self, geo):
        s = self.status
        if s == "no_person":
            self.mode, self.timer = "sleep", 10 ** 9
            return
        if s == "bad":
            lo, hi = self.x_limits(geo)
            target = (lo + hi) / 2
            if abs(self.px - target) > 12:
                self.mode, self.dir, self.timer = "run", (1 if target > self.px else -1), 10 ** 9
            else:
                self.mode, self.vy = "jump", -14.0
            return
        if s == "warning":
            self.mode, self.timer = "run", random.randint(40, 120)
            if random.random() < 0.5:
                self.dir = -self.dir
            return
        if random.random() < 0.55:
            self.mode, self.timer = "walk", random.randint(90, 260)
            self.dir = random.choice((-1, 1))
        else:
            self.mode, self.timer = "idle", random.randint(60, 180)

    def tick(self):
        self.frame += 1
        self.scale += (self.target_scale - self.scale) * 0.15
        geo = self.screen_geo()
        ground = self._ground(geo)
        lo, hi = self.x_limits(geo)
        m = self.mode

        if m == "drag":
            pass
        elif m in ("fall", "jump"):
            self.vy += 1.3
            self.py += self.vy
            self.px += self.vx
            self.vx *= 0.98
            if self.px < lo or self.px > hi:
                self.px = min(max(self.px, lo), hi)
                self.vx *= -0.5
            if self.py >= ground:
                self.py = ground
                if self.status == "bad" and m == "jump":
                    self.vy = -14.0               # เด้งต่อเนื่อง
                else:
                    self.vx = self.vy = 0.0
                    self.mode, self.timer = "idle", 15
        else:
            if m != "climb" and self.py < ground - 1:
                self.start_fall(self.vx, 0.0)
            elif m == "climb":
                self.py -= 1.5
                if self.py <= self.climb_target:
                    self.start_fall(-self.dir * 2.5, -3.0)
            else:
                self.timer -= 1
                if m in ("walk", "run"):
                    sp = 1.8 if m == "walk" else (7.0 if self.status == "bad" else 5.0)
                    self.px += self.dir * sp
                    if self.status == "bad" and m == "run":
                        target = (lo + hi) / 2
                        if abs(self.px - target) <= sp + 1:
                            self.mode, self.vy = "jump", -14.0
                    if self.px <= lo or self.px >= hi:
                        self.px = min(max(self.px, lo), hi)
                        if self.status == "normal" and m == "walk" and random.random() < 0.45:
                            self.mode = "climb"
                            self.dir = 1 if self.px >= hi else -1
                            self.climb_target = max(geo.top(), self.py - random.randint(120, 380))
                        else:
                            self.dir = -self.dir
                if self.timer <= 0 and self.mode in ("idle", "walk", "run", "sleep"):
                    self.choose_action(geo)

        if self.mode == "sleep" and self.status != "no_person":
            self.timer = 0

        # ----- bubble / nag -----
        if self.bubble_ticks > 0:
            self.bubble_ticks -= 1
            if self.bubble_ticks == 0:
                self.bubble_text = ""
        if self.status == "warning" and self.frame >= self.next_nag:
            self.say(random.choice(WARN_MSGS), "warn", 120)
            self.next_nag = self.frame + 360
            self.nag.emit("warn")
        elif self.status == "bad" and self.frame >= self.next_nag:
            self.say(random.choice(BAD_MSGS), "bad", -1)
            self.next_nag = self.frame + 600
            self.nag.emit("bad")

        self.move(int(self.px), int(self.py))
        self.update()

    # ----- mouse -----
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.press_pos = e.globalPosition().toPoint()
            self.drag_off = self.press_pos - self.pos()
            self.drag_hist = []
            self._prev_mode = self.mode
            self.mode = "drag"

    def mouseMoveEvent(self, e):
        if self.mode == "drag" and (e.buttons() & Qt.LeftButton):
            g = e.globalPosition().toPoint()
            np_ = g - self.drag_off
            self.px, self.py = float(np_.x()), float(np_.y())
            self.drag_hist.append(g.x())
            self.drag_hist = self.drag_hist[-5:]
            self.move(np_)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.mode == "drag":
            g = e.globalPosition().toPoint()
            if (g - self.press_pos).manhattanLength() < 5:
                self.say(random.choice(POKE_MSGS), "info", 90)
                self.nag.emit("poke")
            vx = 0.0
            if len(self.drag_hist) >= 2:
                vx = (self.drag_hist[-1] - self.drag_hist[0]) / len(self.drag_hist) * 0.6
            self.start_fall(max(-12.0, min(12.0, vx)), 0.0)
            self.timer = 0

    def contextMenuEvent(self, e):
        if self.menu:
            self.menu.exec(e.globalPos())

    # ----- paint -----
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        if self.bubble_text:
            self.draw_bubble(p)
        p.save()
        p.translate(WIN_W / 2, WIN_H - FOOT_PAD)
        p.scale(self.scale, self.scale)
        sprite = self.sprites.get(self.status) or self.sprites.get("normal")
        draw_mascot(p, self.status, self.frame, self.mode, self.dir, self.vy, sprite)
        p.restore()
        p.end()

    def draw_bubble(self, p):
        font = QFont()
        font.setFamilies(FONT_FAMILIES)
        font.setPointSize(10)
        font.setBold(self.bubble_kind != "info")
        p.setFont(font)
        fm = QFontMetrics(font)
        tr = fm.boundingRect(QRect(0, 0, 400, 400), Qt.AlignLeft, self.bubble_text)
        bw_, bh_ = tr.width() + 26, tr.height() + 18
        head_y = WIN_H - FOOT_PAD - 106 * self.scale
        top = head_y - 12 - bh_
        left = WIN_W / 2 - bw_ / 2

        geo = self.screen_geo()
        gl, gr = self.px + left, self.px + left + bw_       # global bounds
        shift = max(0.0, geo.left() - gl) - max(0.0, gr - (geo.right() + 1))
        left += shift
        left = max(2.0, min(WIN_W - bw_ - 2, left))

        kind = self.bubble_kind
        if kind == "bad":
            flash = (self.frame // 8) % 2 == 0
            bg, edge = QColor("#FFE3E0"), QColor("#E53935" if flash else "#FF8A80")
            fg = QColor("#B71C1C")
        elif kind == "warn":
            bg, edge, fg = QColor("#FFF6D6"), QColor("#E0A800"), QColor("#6B4E16")
        else:
            bg, edge, fg = QColor("#FFFFFF"), QColor("#9AA5B1"), QColor("#2E3A46")
        path = QPainterPath()
        path.addRoundedRect(QRectF(left, top, bw_, bh_), 12, 12)
        cx = WIN_W / 2
        tail = QPolygonF([QPointF(cx - 8, top + bh_ - 1), QPointF(cx + 8, top + bh_ - 1),
                          QPointF(cx, top + bh_ + 11)])
        tp = QPainterPath()
        tp.addPolygon(tail)
        tp.closeSubpath()
        path = path.united(tp)
        p.setPen(QPen(edge, 2))
        p.setBrush(bg)
        p.drawPath(path)
        p.setPen(fg)
        p.drawText(QRectF(left, top, bw_, bh_), Qt.AlignCenter, self.bubble_text)


def load_sprites(enabled=True):
    out = {}
    if not enabled:
        return out
    d = BASE_DIR / "sprites"
    for name in ("normal", "warning", "bad", "no_person"):
        f = d / f"{name}.png"
        if f.exists():
            pm = QPixmap(str(f))
            if not pm.isNull():
                out[name] = pm
    return out


# ----------------------------------------------------------------------------
#  5) หน้าต่างพรีวิวกล้อง
# ----------------------------------------------------------------------------
class PreviewWindow(QWidget):
    closed = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Posture Shimeji - Camera")
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        lay = QVBoxLayout(self)
        self.img = QLabel("กำลังเปิดกล้อง...")
        self.img.setMinimumSize(480, 360)
        self.img.setAlignment(Qt.AlignCenter)
        self.info = QLabel("")
        f = QFont()
        f.setFamilies(FONT_FAMILIES)
        self.info.setFont(f)
        lay.addWidget(self.img)
        lay.addWidget(self.info)

    def set_frame(self, qi):
        self.img.setPixmap(QPixmap.fromImage(qi).scaled(
            480, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, e):
        e.ignore()
        self.hide()
        self.closed.emit()


# ----------------------------------------------------------------------------
#  เสียงประกอบ (สร้างเสียงเองด้วย numpy ไม่ต้องมีไฟล์เสียง) + สัญญาณเตือนวนต่อเนื่อง
# ----------------------------------------------------------------------------
SR = 22050
VOLUMES = {"low": 0.35, "mid": 0.65, "high": 1.0}


def _wav_bytes(x):
    import io
    import wave
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return bio.getvalue()


def _note(freq, dur, amp=0.5, harm=(1.0,), decay=6.0):
    t = np.arange(int(SR * dur)) / SR
    ph = 2 * np.pi * freq * t
    y = sum(h * np.sin((i + 1) * ph) for i, h in enumerate(harm)) / sum(harm)
    env = np.minimum(1.0, t / 0.004) * np.exp(-decay * t) * np.minimum(1.0, (dur - t) / 0.01)
    return amp * y * env


def _gap(d):
    return np.zeros(int(SR * d))


def make_sound(name, vol=1.0):
    if name == "meow":                              # เสียงแมวร้อง "เมี้ยว~" (ตอนจิ้มมาสคอต)
        dur = 0.40
        t = np.arange(int(SR * dur)) / SR
        f = 480 + 360 * np.sin(np.pi * t / dur) ** 1.2
        ph = 2 * np.pi * np.cumsum(f) / SR
        y = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)) / 1.75
        return 0.6 * y * np.sin(np.pi * t / dur) ** 0.7
    if name == "good":                              # กลับมานั่งดีแล้ว ติ๊ง-ติ๊ง-ติ๊ง
        return np.concatenate([_note(1047, .10, .45, decay=8), _note(1319, .10, .45, decay=8),
                               _note(1568, .24, .45, decay=5)])
    if name == "warn":                              # เตือนเบาๆ 2 โน้ต
        return np.concatenate([_note(784, .14, .7, (1, .3), 7), _note(622, .26, .7, (1, .3), 5)])
    if name == "done":                              # ปรับท่านั่งเสร็จ
        return np.concatenate([_note(523, .09, .45, decay=7), _note(659, .09, .45, decay=7),
                               _note(784, .09, .45, decay=7), _note(1047, .28, .45, decay=4)])
    if name == "alarm":                             # ไซเรนดังๆ ลูปจนกว่าจะนั่งตรง
        parts = []
        for fq in (1000, 780, 1000, 780, 1000, 780):
            parts += [_note(fq, 0.17, 0.9, (1, .6, .4, .25, .15), 0.0), _gap(0.04)]
        parts.append(_gap(0.30))
        y = np.concatenate(parts)
        y = np.tanh(1.8 * y) / np.tanh(1.8)
        return y * vol
    return _gap(0.1)


class SoundFx:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cache = {}
        self.alarm_on = False
        self._keep = None
        self._beep = QTimer()                       # ใช้เฉพาะระบบที่ไม่ใช่ Windows
        self._beep.timeout.connect(QApplication.beep)

    def enabled(self):
        return bool(self.cfg["sound"])

    def _path(self, name):
        """สร้างไฟล์ WAV ครั้งแรกที่ต้องใช้ แล้วคืน path (winsound เล่น async จากไฟล์ได้เท่านั้น)"""
        vol = VOLUMES.get(self.cfg["volume"], 1.0) if name == "alarm" else 1.0
        key = (name, vol)
        if key not in self.cache:
            d = APP_DIR / "sounds"
            d.mkdir(parents=True, exist_ok=True)
            f = d / f"{name}_{int(vol * 100)}.wav"
            f.write_bytes(_wav_bytes(make_sound(name, vol)))
            self.cache[key] = str(f)
        return self.cache[key]

    @staticmethod
    def _fallback_beep():
        def _run():
            try:
                import winsound
                winsound.Beep(880, 160)
                winsound.Beep(660, 220)
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()

    def play(self, name):
        if not self.enabled() or self.alarm_on or sys.platform != "win32":
            return
        try:
            import winsound
            self._keep = self._path(name)
            winsound.PlaySound(self._keep, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            log(f"sound error: {e}")
            self._fallback_beep()

    def start_alarm(self, force=False):
        if (not self.enabled()) or (self.alarm_on and not force):
            return
        self.alarm_on = True
        try:
            if sys.platform == "win32":
                import winsound
                self._keep = self._path("alarm")
                winsound.PlaySound(self._keep, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
            else:
                self._beep.start(1200)
        except Exception as e:
            log(f"alarm error: {e}")
            self._beep.start(1200)                  # สำรอง: บี๊บซ้ำๆ

    def stop_alarm(self):
        if not self.alarm_on:
            return
        self.alarm_on = False
        try:
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            self._beep.stop()
        except Exception:
            pass


# ----------------------------------------------------------------------------
#  5.5) หน้าต่างควบคุมหลัก (หน้าตั้งค่าแบบปุ่มใหญ่ ใช้ง่าย)
# ----------------------------------------------------------------------------
STATUS_UI = {
    "normal":    ("😊", "นั่งท่าดีมาก",       "รักษาท่านี้ไว้นะ",                    "#2e9e5b", "#e6f5ec"),
    "warning":   ("😟", "เริ่มค่อมแล้วนะ",     "ลองยืดหลัง ยกหัวขึ้นนิดนึง",           "#d98a00", "#fff3dc"),
    "bad":       ("😣", "หลังค่อมแล้ว!",       "ยืดหลัง ดึงคางเข้า ผ่อนคลายไหล่",      "#d64545", "#fde8e8"),
    "no_person": ("😴", "ไม่พบคนหน้ากล้อง",    "นั่งให้เห็นใบหน้าและไหล่ทั้งสองข้าง",  "#7b8190", "#eceef2"),
    "paused":    ("⏸️", "พักการตรวจจับอยู่",    "กด \"เริ่มตรวจจับ\" เพื่อเปิดกล้องอีกครั้ง", "#5b6bd6", "#e9ecfb"),
}

PANEL_QSS = """
QWidget { background: #f6f7fb; color: #23272f; }
QFrame#card { background: white; border: 1px solid #e3e6ee; border-radius: 14px; }
QLabel { background: transparent; }
QLabel#h { font-size: 11pt; font-weight: bold; color: #4a5160; }
QLabel#hint { color: #7b8190; font-size: 9pt; }
QPushButton { background: white; border: 1px solid #cfd4e0; border-radius: 10px;
              padding: 9px 12px; }
QPushButton:hover { background: #eef1f9; }
QPushButton:pressed { background: #e1e6f4; }
QPushButton#primary { background: #4a6cf7; color: white; border: none;
                      font-size: 12pt; font-weight: bold; padding: 13px; }
QPushButton#primary:hover { background: #3d5ce0; }
QPushButton#stop { background: #5b6bd6; }
QPushButton[seg="true"] { padding: 8px 6px; }
QPushButton[seg="true"]:checked { background: #4a6cf7; color: white; border-color: #4a6cf7; }
QCheckBox { spacing: 10px; padding: 3px 0; background: transparent; }
QCheckBox::indicator { width: 18px; height: 18px; border: 2px solid #9aa3b5;
                       border-radius: 5px; background: white; }
QCheckBox::indicator:checked { background: #4a6cf7; border-color: #4a6cf7; }
QScrollArea { background: transparent; }
QComboBox { background: white; border: 1px solid #cfd4e0; border-radius: 8px; padding: 6px 10px; }
QProgressBar { background: #e6e9f1; border: none; border-radius: 5px; height: 10px; }
QProgressBar::chunk { border-radius: 5px; }
"""


def _bind(action, button):
    """ผูก QAction (ตัวเก็บสถานะจริง) กับปุ่ม/เช็กบ็อกซ์ ให้ซิงก์กันสองทาง"""
    button.setChecked(action.isChecked())
    button.toggled.connect(action.setChecked)

    def back(v):
        button.blockSignals(True)
        button.setChecked(v)
        button.blockSignals(False)
    action.toggled.connect(back)


class ControlPanel(QWidget):
    WIDTH = 440

    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self._status = "no_person"
        self._paused = False
        self.setWindowTitle("Posture Shimeji")
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowSystemMenuHint
                            | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)   # ไม่มีปุ่มขยายเต็มจอ
        self.setFixedWidth(self.WIDTH)
        f = QFont()
        f.setFamilies(FONT_FAMILIES)
        f.setPointSize(10)
        self.setFont(f)
        self.setStyleSheet(PANEL_QSS)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.inner = QWidget()
        scroll.setWidget(self.inner)
        outer.addWidget(scroll)
        root = QVBoxLayout(self.inner)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # ---- การ์ดสถานะ ----
        self.card = QFrame()
        self.card.setObjectName("card")
        cl = QVBoxLayout(self.card)
        cl.setContentsMargins(18, 16, 18, 16)
        cl.setSpacing(4)
        self.face = QLabel("😴")
        self.face.setAlignment(Qt.AlignCenter)
        self.face.setStyleSheet("font-size: 44pt;")
        self.title = QLabel("")
        self.title.setAlignment(Qt.AlignCenter)
        self.sub = QLabel("")
        self.sub.setAlignment(Qt.AlignCenter)
        self.sub.setObjectName("hint")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar_lbl = QLabel("ระดับความหลังค่อม")
        self.bar_lbl.setObjectName("hint")
        self.bar_lbl.setAlignment(Qt.AlignCenter)
        cl.addWidget(self.face)
        cl.addWidget(self.title)
        cl.addWidget(self.sub)
        cl.addSpacing(6)
        cl.addWidget(self.bar)
        cl.addWidget(self.bar_lbl)
        root.addWidget(self.card)

        # ---- แถบแจ้งเตือนครั้งแรก (ยังไม่เคยปรับท่านั่ง) ----
        self.first = QLabel("👉 ยังไม่ได้ตั้งท่านั่งมาตรฐาน กดปุ่ม \"ปรับท่านั่งมาตรฐาน\" ด้านล่าง\n"
                            "แล้วนั่งหลังตรงๆ มองกล้อง 5 วินาที")
        self.first.setWordWrap(True)
        self.first.setStyleSheet("background:#fff3dc; border-radius:10px; padding:10px; color:#7a5300;")
        root.addWidget(self.first)

        # ---- ปุ่มหลัก ----
        self.btn_main = QPushButton("⏸  หยุดตรวจจับชั่วคราว")
        self.btn_main.setObjectName("primary")
        self.btn_main.clicked.connect(lambda: ctrl.act_pause.setChecked(not ctrl.act_pause.isChecked()))
        root.addWidget(self.btn_main)

        row = QHBoxLayout()
        b1 = QPushButton("🎯  ปรับท่านั่งมาตรฐาน")
        b1.clicked.connect(ctrl.start_calibration)
        self.btn_cam = QPushButton("📷  ดูภาพกล้อง")
        self.btn_cam.setCheckable(True)
        _bind(ctrl.act_preview, self.btn_cam)
        row.addWidget(b1)
        row.addWidget(self.btn_cam)
        root.addLayout(row)

        # ---- ตั้งค่า ----
        box = self._section("ตั้งค่า")
        bl = box.layout()
        lbl = QLabel("ความไวในการเตือน")
        bl.addWidget(lbl)
        seg = QHBoxLayout()
        seg.setSpacing(6)
        self.sens_group = QButtonGroup(self)
        self.sens_btns = {}
        for key, name in (("low", "ผ่อนปรน"), ("normal", "ปกติ"), ("high", "เข้มงวด")):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setProperty("seg", True)
            self.sens_group.addButton(b)
            b.clicked.connect(lambda _=False, k=key: ctrl.cfg.__setitem__("sensitivity", k))
            seg.addWidget(b)
            self.sens_btns[key] = b
        self.sens_btns.get(ctrl.cfg["sensitivity"], self.sens_btns["normal"]).setChecked(True)
        bl.addLayout(seg)

        self.cb_sound = QCheckBox("เสียงเตือนเมื่อหลังค่อม")
        _bind(ctrl.act_sound, self.cb_sound)
        vr = QHBoxLayout()
        vr.setSpacing(6)
        vr.addWidget(QLabel("ความดังเสียงเตือน"))
        self.vol_group = QButtonGroup(self)
        self.vol_btns = {}
        for key, name in (("low", "เบา"), ("mid", "กลาง"), ("high", "ดัง")):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setProperty("seg", True)
            self.vol_group.addButton(b)
            b.clicked.connect(lambda _=False, k=key: ctrl.set_volume(k))
            vr.addWidget(b)
            self.vol_btns[key] = b
        self.vol_btns.get(ctrl.cfg["volume"], self.vol_btns["high"]).setChecked(True)
        bl.addLayout(vr)
        test = QPushButton("🔊  ลองฟังเสียงเตือน")
        test.clicked.connect(ctrl.test_alarm)
        bl.addWidget(test)
        vhint = QLabel("ตอนหลังค่อมชัดเจน เสียงจะดังต่อเนื่องจนกว่าจะนั่งตรง "
                       "(คลิกที่มาสคอตเพื่อพักเสียง 1 นาที)")
        vhint.setObjectName("hint")
        vhint.setWordWrap(True)
        bl.addWidget(vhint)
        self.cb_auto = QCheckBox("เปิดโปรแกรมอัตโนมัติเมื่อเปิดเครื่อง")
        _bind(ctrl.act_auto, self.cb_auto)
        bl.addWidget(self.cb_sound)
        bl.addWidget(self.cb_auto)

        cr = QHBoxLayout()
        cr.addWidget(QLabel("กล้องที่ใช้"))
        self.combo = QComboBox()
        for i in range(4):
            self.combo.addItem(f"กล้องตัวที่ {i + 1}" + (" (ค่าเริ่มต้น)" if i == 0 else ""), i)
        self.combo.setCurrentIndex(max(0, min(3, int(ctrl.cfg["camera"]))))
        self.combo.currentIndexChanged.connect(lambda i: ctrl.set_camera(self.combo.itemData(i)))
        cr.addWidget(self.combo, 1)
        bl.addLayout(cr)
        hint = QLabel("ไม่เห็นภาพ? ลองเลือกกล้องตัวอื่น หรือปิดโปรแกรมที่ใช้กล้องอยู่ (Zoom/Teams)")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        bl.addWidget(hint)
        root.addWidget(box)

        # ---- มาสคอต ----
        box2 = self._section("มาสคอต")
        b2 = box2.layout()
        mr = QHBoxLayout()
        mr.setSpacing(6)
        self.spr_group = QButtonGroup(self)
        self.spr_default = QPushButton("🐱 ตัวเดิม (แมวส้ม)")
        self.spr_custom = QPushButton("🖼 รูปของฉัน")
        for b, flag in ((self.spr_default, False), (self.spr_custom, True)):
            b.setCheckable(True)
            b.setProperty("seg", True)
            self.spr_group.addButton(b)
            b.clicked.connect(lambda _=False, f=flag: ctrl.set_custom(f))
            mr.addWidget(b)
        b2.addLayout(mr)
        pick = QPushButton("📂  เลือกไฟล์รูปใหม่ (PNG)…")
        pick.clicked.connect(ctrl.pick_sprite)
        b2.addWidget(pick)
        b2.addWidget(QLabel("ลองดูท่าทางมาสคอต (ไม่ต้องนั่งค่อมจริง)"))
        dr = QHBoxLayout()
        dr.setSpacing(6)
        for s, name in (("normal", "😊 ปกติ"), ("warning", "😟 เตือน"), ("bad", "😣 ค่อม")):
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, k=s: ctrl.demo(k))
            dr.addWidget(b)
        stop = QPushButton("หยุดดู")
        stop.clicked.connect(lambda: ctrl.demo(None))
        dr.addWidget(stop)
        b2.addLayout(dr)
        root.addWidget(box2)

        foot = QLabel("ปิดหน้าต่างนี้แล้วโปรแกรมยังทำงานต่อที่ไอคอนข้างนาฬิกา (system tray)\n"
                      "คลิกไอคอนเพื่อเปิดหน้านี้อีกครั้ง")
        foot.setObjectName("hint")
        foot.setAlignment(Qt.AlignCenter)
        root.addWidget(foot)
        quit_b = QPushButton("ออกจากโปรแกรม")
        quit_b.clicked.connect(ctrl.quit)
        root.addWidget(quit_b)
        root.addStretch(1)

        self.refresh()
        self.refresh_sprites()

    def _section(self, title):
        fr = QFrame()
        fr.setObjectName("card")
        lay = QVBoxLayout(fr)
        lay.setContentsMargins(16, 12, 16, 14)
        lay.setSpacing(8)
        h = QLabel(title)
        h.setObjectName("h")
        lay.addWidget(h)
        return fr

    # ---- ถูกเรียกจาก Controller ----
    def fit_to_screen(self):
        """ขนาดพอดีเนื้อหา แต่ไม่สูงเกินจอ (เกินแล้วเลื่อนลงได้)"""
        scr = self.screen() or QGuiApplication.primaryScreen()
        avail = scr.availableGeometry().height() if scr else 800
        want = self.inner.sizeHint().height() + 6
        self.resize(self.WIDTH, max(360, min(want, avail - 70)))

    def refresh_sprites(self):
        has = self.ctrl.has_custom_files()
        self.spr_custom.setEnabled(has)
        use = has and bool(self.ctrl.cfg["use_custom"])
        (self.spr_custom if use else self.spr_default).setChecked(True)

    def set_status(self, status):
        self._status = status
        self.refresh()

    def set_paused(self, v):
        self._paused = bool(v)
        self.refresh()

    def set_score(self, score):
        self.bar.setValue(int(round(max(0.0, min(1.0, score)) * 100)))

    def refresh(self):
        key = "paused" if self._paused else self._status
        emoji, title, sub, color, bg = STATUS_UI.get(key, STATUS_UI["no_person"])
        self.face.setText(emoji)
        self.title.setText(title)
        self.title.setStyleSheet(f"font-size: 17pt; font-weight: bold; color: {color};")
        self.sub.setText(sub)
        self.card.setStyleSheet(f"QFrame#card {{ background: {bg}; border: 1px solid {color}; "
                                f"border-radius: 14px; }}")
        self.bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")
        if self._paused:
            self.bar.setValue(0)
        self.btn_main.setText("▶  เริ่มตรวจจับ" if self._paused else "⏸  หยุดตรวจจับชั่วคราว")
        self.btn_main.setObjectName("primary")
        self.first.setVisible(self.ctrl.baseline is None)

    def closeEvent(self, e):
        e.ignore()
        self.hide()
        self.ctrl.on_panel_closed()


# ----------------------------------------------------------------------------
#  6) ตัวควบคุมหลัก + system tray
# ----------------------------------------------------------------------------
STATUS_TH = {"normal": "ปกติ", "warning": "เตือน", "bad": "หลังค่อม!", "no_person": "ไม่พบคน"}


def set_autostart(enable):
    if sys.platform != "win32":
        return False
    import winreg
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                         r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                         winreg.KEY_SET_VALUE)
    try:
        if enable:
            cmd = f'"{sys.executable}"' if FROZEN else \
                f'"{sys.executable}" "{os.path.abspath(__file__)}"'
            winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, cmd)
        else:
            try:
                winreg.DeleteValue(key, APP_NAME)
            except FileNotFoundError:
                pass
    finally:
        winreg.CloseKey(key)
    return True


def is_autostart():
    if sys.platform != "win32":
        return False
    import winreg
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Run")
        winreg.QueryValueEx(key, APP_NAME)
        winreg.CloseKey(key)
        return True
    except OSError:
        return False


def play_alert():
    def _run():
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(880, 160)
                winsound.Beep(660, 220)
            else:
                QApplication.beep()
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


class Controller(QObject):
    def __init__(self, app, use_camera=True):
        super().__init__()
        self.app = app
        self.cfg = Config()
        self.sound = SoundFx(self.cfg)
        self.baseline = self.cfg["baseline"]
        self.machine = StatusMachine("no_person")
        self.cur_status = None
        self.ema = None
        self.calibrating = False
        self.override = None
        self.last_info = ""

        self.snooze_until = 0.0
        self.mascot = Mascot(load_sprites(bool(self.cfg["use_custom"])))
        self.mascot.nag.connect(self.on_nag)
        self.preview_win = PreviewWindow()
        self.preview_win.closed.connect(lambda: self.act_preview.setChecked(False))
        self.last_score = 0.0

        self.make_actions()
        self.build_menu()
        self.mascot.menu = self.menu
        self.panel = ControlPanel(self)

        self.tray = QSystemTrayIcon(QIcon(make_icon("normal")), self)
        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self.on_tray)
        self.tray.show()

        self.tracker = Tracker(self.cfg["camera"])
        self.tracker.result.connect(self.on_result)
        self.tracker.preview.connect(self.preview_win.set_frame)
        self.tracker.calib.connect(self.on_calib)
        self.tracker.error.connect(self.on_error)
        if use_camera:
            self.tracker.start()

        self.apply_status("no_person", force=True)
        self.mascot.show()
        if self.baseline is None:
            self.mascot.say("สวัสดี! ฉันจะช่วยเฝ้าท่านั่งให้\nนั่งหลังตรงๆ แล้วรอสักครู่นะ", "info", 120)
            self.show_panel()                         # เปิดหน้าควบคุมให้เห็นครั้งแรก
            QTimer.singleShot(3500, self.start_calibration)

    # ---------- menu ----------
    def make_actions(self):
        """ตัวเก็บสถานะการตั้งค่า (หน้าควบคุมกับเมนูใช้ร่วมกัน)"""
        self.act_pause = QAction("หยุดตรวจจับชั่วคราว (ปิดกล้อง)", self)
        self.act_pause.setCheckable(True)
        self.act_pause.toggled.connect(self.toggle_pause)

        self.act_preview = QAction("แสดงภาพกล้อง", self)
        self.act_preview.setCheckable(True)
        self.act_preview.toggled.connect(self.toggle_preview)

        self.act_sound = QAction("เสียงเตือนเมื่อหลังค่อม", self)
        self.act_sound.setCheckable(True)
        self.act_sound.setChecked(bool(self.cfg["sound"]))
        self.act_sound.toggled.connect(self.toggle_sound)

        self.act_auto = QAction("เปิดอัตโนมัติเมื่อเริ่ม Windows", self)
        self.act_auto.setCheckable(True)
        self.act_auto.setChecked(is_autostart())
        self.act_auto.toggled.connect(self.toggle_autostart)

    def build_menu(self):
        """เมนูคลิกขวา: เหลือแค่ที่ใช้บ่อย ที่เหลืออยู่ในหน้าควบคุม"""
        m = QMenu()
        f = QFont()
        f.setFamilies(FONT_FAMILIES)
        m.setFont(f)
        self.act_head = m.addAction("STATUS: ...")
        self.act_head.setEnabled(False)
        m.addSeparator()
        open_a = m.addAction("⚙  เปิดหน้าต่างควบคุม")
        ft = open_a.font()
        ft.setBold(True)
        open_a.setFont(ft)
        open_a.triggered.connect(self.show_panel)
        m.addAction(self.act_pause)
        m.addAction("ปรับท่านั่งมาตรฐาน (Calibrate)").triggered.connect(self.start_calibration)
        m.addSeparator()
        m.addAction("ออกจากโปรแกรม").triggered.connect(self.quit)
        self.menu = m

    def show_panel(self):
        if not self.panel.isVisible():
            self.panel.fit_to_screen()
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()

    def on_panel_closed(self):
        if not self.cfg["tip_shown"]:
            self.cfg["tip_shown"] = True
            self.tray.showMessage(APP_NAME, "โปรแกรมยังทำงานอยู่ที่ไอคอนข้างนาฬิกา คลิกไอคอนเพื่อเปิดหน้าควบคุมอีกครั้ง",
                                  QSystemTrayIcon.Information, 5000)

    def on_tray(self, reason):
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            if self.panel.isVisible():
                self.panel.hide()
            else:
                self.show_panel()

    # ---------- actions ----------
    def start_calibration(self):
        if self.act_pause.isChecked():
            self.act_pause.setChecked(False)
        self.calibrating = True
        self.sound.stop_alarm()
        self.override = None
        self.mascot.set_status("calibrating")
        self.mascot.say("นั่งหลังตรงๆ มองกล้อง\nจะเริ่มจดจำใน 5 วินาที", "info", -1)
        self.tracker.request_calibration(5.0, 5.0)
        log("calibration requested")

    def on_calib(self, d):
        ph = d["phase"]
        if ph == "wait":
            self.mascot.say(f"เตรียมตัว... นั่งหลังตรงๆ\nเริ่มใน {int(d['remaining']) + 1} วิ", "info", -1)
        elif ph == "hold":
            self.mascot.say(f"นิ่งๆ นะ กำลังจดจำท่าที่ดี\nเหลือ {int(d['remaining']) + 1} วิ", "info", -1)
        elif ph == "done":
            self.baseline = d["baseline"]
            self.cfg["baseline"] = self.baseline
            self.finish_calibration()
            self.sound.play("done")
            self.mascot.say("เรียบร้อย! จำท่านั่งที่ดีของคุณแล้ว\nต่อไปฉันจะคอยเตือนนะ", "info", 150)
            log(f"calibrated: {self.baseline}")
        elif ph == "fail":
            self.finish_calibration()
            self.mascot.say("ไม่เห็นตัวคุณชัดเลย\nลองใหม่จากเมนู Calibrate นะ", "warn", 200)

    def finish_calibration(self):
        self.calibrating = False
        self.ema = None
        self.machine.force("normal")
        self.cur_status = None
        self.mascot.status = "calibrating"      # ให้ set_status ทำงานเต็มรูปแบบ
        self.apply_status("normal", force=True, quiet=True)
        self.panel.refresh()

    def toggle_pause(self, v):
        self.tracker.paused = bool(v)
        self.panel.set_paused(v)
        self.sync_alarm()
        if v:
            self.machine.force("no_person")
            self.apply_status("no_person", force=True)
            self.mascot.say("พักก่อนนะ (ปิดกล้องแล้ว)", "info", 90)

    def toggle_preview(self, v):
        self.tracker.want_preview = bool(v)
        self.preview_win.setVisible(bool(v))

    def toggle_autostart(self, v):
        try:
            set_autostart(bool(v))
        except Exception as e:
            self.tray.showMessage(APP_NAME, f"ตั้งค่าเปิดอัตโนมัติไม่สำเร็จ: {e}")

    def set_camera(self, i):
        self.cfg["camera"] = i
        self.tracker.reopen_camera(i)

    def demo(self, status):
        if status is None:
            self.override = None
            self.machine.force("normal")
            return
        self.override = (status, time.time() + 12.0)
        self.machine.force(status)
        self.apply_status(status, force=True)
        QTimer.singleShot(12400, self.end_demo)

    def open_sprites(self):
        d = BASE_DIR / "sprites"
        d.mkdir(exist_ok=True)
        readme = d / "README.txt"
        if not readme.exists():
            readme.write_text(
                "วางรูป PNG พื้นหลังโปร่งใส (หันหน้าไปทางขวา) ชื่อ:\n"
                "  normal.png  warning.png  bad.png  no_person.png\n"
                "แล้วเปิดโปรแกรมใหม่ มาสคอตจะใช้รูปของคุณแทนตัวที่วาดไว้\n", encoding="utf-8")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    # ---------- เสียง ----------
    def sync_alarm(self):
        """สัญญาณเตือนดังต่อเนื่องตราบใดที่ยังหลังค่อม (หยุดเมื่อท่าปกติ/พัก/ปรับท่า/ปิดเสียง)"""
        want = (self.cur_status == "bad" and not self.act_pause.isChecked()
                and not self.calibrating and time.time() >= self.snooze_until)
        if want and self.cfg["sound"]:
            self.sound.start_alarm()
        else:
            self.sound.stop_alarm()

    def toggle_sound(self, v):
        self.cfg["sound"] = bool(v)
        self.sync_alarm()

    def set_volume(self, key):
        self.cfg["volume"] = key
        if self.sound.alarm_on:
            self.sound.start_alarm(force=True)

    def test_alarm(self):
        if not self.cfg["sound"]:
            self.mascot.say("เสียงถูกปิดอยู่นะ\nติ๊กเปิดเสียงก่อน", "info", 90)
            return
        self.sound.start_alarm(force=True)
        QTimer.singleShot(2200, self.sync_alarm)

    def end_demo(self):
        if self.override and time.time() >= self.override[1] - 0.3:
            self.override = None
            st = "no_person" if self.act_pause.isChecked() else "normal"
            self.machine.force(st)
            self.apply_status(st, force=True)

    # ---------- รูปมาสคอต ----------
    def has_custom_files(self):
        d = BASE_DIR / "sprites"
        return any((d / f"{n}.png").exists() for n in STATUSES)

    def set_custom(self, flag):
        self.cfg["use_custom"] = bool(flag)
        self.mascot.set_sprites(load_sprites(bool(flag)))
        self.panel.refresh_sprites()
        self.mascot.say("เปลี่ยนเป็นรูปของคุณแล้ว" if flag else "กลับมาเป็นแมวส้มตัวเดิมแล้ว~", "info", 90)

    def pick_sprite(self):
        fn, _ = QFileDialog.getOpenFileName(self.panel, "เลือกรูปมาสคอต (PNG พื้นหลังโปร่งใส หันหน้าไปทางขวา)",
                                            str(Path.home()), "รูปภาพ (*.png)")
        if not fn:
            return
        if QPixmap(fn).isNull():
            QMessageBox.warning(self.panel, APP_NAME, "เปิดไฟล์รูปนี้ไม่ได้ ลองเลือกไฟล์ PNG อื่น")
            return
        items = ["ใช้รูปนี้ทุกสถานะ (ง่ายที่สุด)", "เฉพาะสถานะปกติ", "เฉพาะสถานะเตือน",
                 "เฉพาะสถานะหลังค่อม", "เฉพาะสถานะไม่พบคน"]
        item, ok = QInputDialog.getItem(self.panel, APP_NAME, "ใช้รูปนี้กับสถานะไหน?", items, 0, False)
        if not ok:
            return
        d = BASE_DIR / "sprites"
        d.mkdir(exist_ok=True)
        try:
            targets = STATUSES if item == items[0] else (STATUSES[items.index(item) - 1],)
            for n in targets:
                shutil.copyfile(fn, d / f"{n}.png")
        except Exception as e:
            QMessageBox.warning(self.panel, APP_NAME, f"บันทึกรูปไม่สำเร็จ: {e}")
            return
        self.set_custom(True)

    def on_nag(self, kind):
        if kind == "warn":
            self.sound.play("warn")
        elif kind == "poke":
            if self.sound.alarm_on:                    # จิ้มมาสคอตตอนเสียงดัง = ขอพัก 1 นาที
                self.snooze_until = time.time() + 60
                self.sound.stop_alarm()
                QTimer.singleShot(60500, self.sync_alarm)
                self.mascot.say("โอเค พักเสียง 1 นาที\nแต่ต้องนั่งตรงๆ นะ!", "warn", 120)
            else:
                self.sound.play("meow")

    def on_error(self, msg):
        log(f"ERROR: {msg}")
        self.tray.showMessage(APP_NAME, msg, QSystemTrayIcon.Warning, 6000)
        self.mascot.say("เปิดกล้องไม่ได้ T_T\nเช็คว่ามีโปรแกรมอื่นใช้อยู่ไหม", "warn", 200)

    # ---------- main flow ----------
    def on_result(self, r):
        if self.calibrating:
            return
        now = time.time()
        if self.override:
            if now < self.override[1]:
                return
            self.override = None
        score = 0.0
        if not r["valid"]:
            raw = "no_person"
            self.ema = None
            self.last_info = "ไม่พบคน / เห็นไหล่ไม่ชัด"
        else:
            sc, parts = score_posture(r["metrics"], self.baseline)
            self.ema = sc if self.ema is None else 0.3 * sc + 0.7 * self.ema
            score = self.ema
            k = SENS.get(self.cfg["sensitivity"], 1.0)
            raw = classify(score, self.machine.status, k)
            self.last_info = (f"score={score:.2f}  head={parts['head']:.2f} fwd={parts['fwd']:.2f} "
                              f"lean={parts['scale']:.2f} tilt={parts['tilt']:.2f}")
        status = self.machine.update(raw, now)
        self.panel.set_score(score)
        self.apply_status(status)
        if self.preview_win.isVisible():
            self.preview_win.info.setText(f"STATUS: {status}    {self.last_info}")

    def apply_status(self, status, force=False, quiet=False):
        if status == self.cur_status and not force:
            return
        prev_status = self.cur_status
        self.cur_status = status
        log(f"STATUS: {status}")
        try:
            STATUS_FILE.write_text(status, encoding="utf-8")
        except Exception:
            pass
        self.mascot.set_status(status)
        self.tray.setIcon(QIcon(make_icon(status)))
        self.tray.setToolTip(f"{APP_NAME} - STATUS: {status} ({STATUS_TH[status]})")
        self.act_head.setText(f"STATUS: {status}  ({STATUS_TH[status]})")
        self.panel.set_status(status)
        self.sync_alarm()
        if status == "normal" and prev_status in ("warning", "bad") and not quiet:
            self.sound.play("good")

    def quit(self):
        self.sound.stop_alarm()
        self.tracker.stop()
        self.tracker.wait(2000)
        self.tray.hide()
        self.app.quit()


# ----------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName(APP_NAME)

    if "--make-icon" in args:                      # ใช้ตอน build เพื่อสร้าง icon.ico
        out = args[args.index("--make-icon") + 1]
        ok = make_icon("normal", 256).save(out)
        print("icon saved" if ok else "icon failed")
        return 0

    if "--selftest" in args:                       # ตรวจว่า exe โหลดโมเดล MediaPipe ได้
        try:
            import cv2
            import mediapipe as mp
            pose = mp.solutions.pose.Pose(model_complexity=1)
            pose.process(np.zeros((480, 640, 3), np.uint8))
            pose.close()
            (APP_DIR / "selftest.txt").write_text("OK", encoding="utf-8")
            print("SELFTEST OK")
            return 0
        except Exception as e:
            (APP_DIR / "selftest.txt").write_text(f"FAIL: {e}", encoding="utf-8")
            print("SELFTEST FAIL", e)
            return 1

    lock = QLockFile(os.path.join(tempfile.gettempdir(), f"{APP_NAME}.lock"))
    if not lock.tryLock(200):
        QMessageBox.information(None, APP_NAME, "โปรแกรมเปิดอยู่แล้ว (ดูไอคอนที่ system tray)")
        return 0

    ctrl = Controller(app, use_camera="--no-camera" not in args)
    code = app.exec()
    lock.unlock()
    return code


if __name__ == "__main__":
    sys.exit(main())
