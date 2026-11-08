# -*- coding: utf-8 -*-
"""
exam_auto_solver.py —— 全屏网页在线考试自动答题脚本（分辨率自适应）

============================== 工作闭环 ==============================
    截全屏  ->  OCR（默认 RapidOCR，可选 PaddleOCR）识别题干 / 选项 + 返回文字坐标
            ->  由选项文字左边缘向左偏移 DOT_OFFSET_X 像素，推算小圆点坐标
            ->  题干 + 选项丢给大模型，强制返回结构化 JSON
            ->  按答案依次点击小圆点（首次点击前先等待，点击之间严格 2 秒）
            ->  只对【题干区】做指纹判断有没有翻页（选项被点亮的像素变化不算翻页）
            ->  没翻页则按 Enter 兜底；同一道题绝不会被重复点击
                （多选题点两次 = 把已选中的选项取消掉，等于亲手把这题答废）
            ->  进入下一轮循环

============================== 循环怎么停 ==============================
    不再用「固定轮数 = 题量」这种粗口径，改成三条件，任一命中就收工：
      1) 已成功作答达到 MAX_ANSWERED 题                    -> 正常收工
      2) 连续 MAX_NO_PROGRESS 轮毫无进展（没作答、页面也没翻）-> 提前收工，避免空转
      3) 总轮数达到 MAX_ROUNDS                            -> 纯兜底硬上限

============================== 依赖安装 ==============================
    pip install pyautogui opencv-python requests numpy

    OCR 二选一（默认优先 RapidOCR）：
      pip install rapidocr-onnxruntime     （推荐：模型随包自带，不用 paddle，打包 exe 最省事）
      pip install paddlepaddle paddleocr   （paddleocr 2.x 配 paddlepaddle 2.5/2.6 最稳）

    * 用 paddleocr 时首次运行会自动下载中文识别模型（约 10~20MB），
      下载慢的话可以先手动跑一遍让它下完，或者设置国内镜像：
      set PADDLE_PDX_MODEL_SOURCE=BOS     (Windows)
      export PADDLE_PDX_MODEL_SOURCE=BOS  (macOS / Linux)

    * RapidOCR 的模型文件直接封装在包里，不用联网下载。

    * Linux 下 pyautogui 依赖 X11，Wayland 桌面下截图/点击会失效，
      请切到 Xorg 会话再跑。

============================== 运行方式 ==============================
    python exam_auto_solve.py           命令行模式
    python exam_bot_gui.py              图形界面模式（填 API Key / Base URL，点「开始」）
    双击打包出来的 exe                  同上，配置存在 exe 旁边的 config.json

    ⚠️ 强烈建议第一次先设 DRY_RUN = True（GUI 里就是勾上「试运行」）跑一遍：
       只打印「识别到几道题 / 答案是什么 / 准备点击哪个坐标」，不动鼠标。
       确认 DOT_OFFSET_X（默认 35）算出来的坐标确实落在小圆点上，再改成 False 正式跑。

    ⚠️ 紧急中止：把鼠标猛甩到屏幕【左上角】(0,0)，pyautogui 的 FAILSAFE 会直接抛异常停住。
       也可以在终端里 Ctrl + C，GUI 里点「停止」。

============================== 大模型 API 配置 ==============================
    方式一（GUI / exe，最省事）：
        在界面里填 API Key、Base URL、模型名，点「保存配置」写到 exe 旁边的 config.json，
        点「测试连接」可以直接验证能不能通。

    方式二（环境变量，Key 不落到代码里）：
        Windows :  set OPENAI_API_KEY=sk-xxxx
                   set OPENAI_BASE_URL=https://api.openai.com/v1
                   set OPENAI_MODEL=gpt-4o-mini
        macOS/Linux:
                   export OPENAI_API_KEY=sk-xxxx
                   export OPENAI_BASE_URL=https://api.openai.com/v1
                   export OPENAI_MODEL=gpt-4o-mini
        （BASE_URL 可换成任意 OpenAI 兼容网关，如 DeepSeek / 通义 / 自建中转）

    方式三：直接把下面的 API_KEY 常量改掉（不推荐，容易泄漏）

    优先级：GUI / config.json > 环境变量 > 代码里的默认值
"""

import os
import re
import sys
import json
import time
import glob
import hashlib
import difflib
import logging
import threading
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import pyautogui
import requests

# =====================================================================
# 一、用户配置区（改这里就行）
# =====================================================================

# ---------------- 大模型 API ----------------
API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-把你的Key填在这里")
BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")  # 以 /v1 结尾，不要带 /chat/completions
MODEL_NAME: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_TIMEOUT: int = 60              # 单次接口超时（秒）
LLM_RETRY: int = 2                 # 接口报错重试次数

# ---------------- 节奏控制（答题快慢全在这几个值上）----------------
# 实测一题的开销：OCR 1.5~4 秒 + 接口 1~2 秒 + 下面这些「等待」。
# 原来这些等待加起来是 7 秒（单选）/ 13 秒（多选多点几下），比算题本身还久，纯空耗。
# 默认按「快速」档给；万一你的页面渲染慢（点早了会点空），在界面上切「标准」即可。
CLICK_INTERVAL: float = 0.8        # 多选题里两次点击之间的间隔（秒）
FIRST_CLICK_DELAY: float = 0.6     # 每道题【首次点击之前】的等待（等上一题的跳转 / 渲染稳定）
AFTER_ANSWER_WAIT: float = 1.0     # 点完「下一题」/ 按 Enter 后，等页面反应的秒数
SINGLE_SETTLE_WAIT: float = 1.0    # 点完选项后等页面反应的秒数（用来判断是否已自动跳转）
LOOP_SLEEP: float = 0.2            # 每轮识别循环开始前的基础等待
START_COUNTDOWN: int = 5           # 点「开始」后留给用户切到浏览器的时间（秒）

# 界面上的「答题节奏」就是这几个值的预设：数值越小越快，也越容易抢跑。
SPEED_PRESETS: Dict[str, Dict[str, float]] = {
    "极速": {"loop_sleep": 0.0, "first_click_delay": 0.3, "single_settle": 0.6,
             "after_answer": 0.8, "click_interval": 0.3},
    "快速": {"loop_sleep": 0.2, "first_click_delay": 0.6, "single_settle": 1.0,
             "after_answer": 1.0, "click_interval": 0.8},
    "标准": {"loop_sleep": 1.5, "first_click_delay": 1.5, "single_settle": 2.0,
             "after_answer": 2.0, "click_interval": 2.0},
}
DEFAULT_SPEED: str = "快速"

# ---------------- 循环终止条件（不再「固定轮数 = 题量」） ----------------
MAX_ANSWERED: int = 90             # 已成功作答达到这么多题就收工（按「已答数」终止）
MAX_NO_PROGRESS: int = 15          # 连续这么多轮毫无进展（没作答、页面也没翻）就收工
MAX_ROUNDS: int = 400              # 总轮数硬上限（纯兜底，防任何极端情况下的死循环）
SAME_QUESTION_LIMIT: int = 3       # 同一题连续被【跳过】这么多次 => 判定卡住，按 Enter 尝试翻页
# 「屏幕上还是刚才点过的那道题」的判定阈值：题干+选项文本相似度 ≥ 该值即认为同一道题。
# 取 0.95 是个折中：OCR 对同一屏文字偶尔会抖一两个字（相似度仍 ≥0.95，不会漏判），
# 而两道不同的题目几乎不可能达到 0.95 以上。判定偏保守，宁可少拦也不误拦。
SAME_QUESTION_TEXT_RATIO: float = 0.95

# ---------------- 坐标 / 识别参数 ----------------
DOT_OFFSET_X: int = 35             # 选项文字左边缘向左偏移 N 像素 = 小圆点 X 坐标（按实际微调）
# 高分屏（4K / 150% 缩放）下截图是物理像素，圆点到文字的真实像素距离会比 35 大得多。
# 如果你的 DOT_OFFSET_X 是在 1920 宽的屏幕上校准出来的，把下面开关打开即可自动等比缩放，
# 免得换个屏幕又得重新校准。（默认关，保持原来的行为不变）
DOT_OFFSET_SCALE_BY_WIDTH: bool = False   # True = 偏移量按「截图宽 / DOT_OFFSET_REF_WIDTH」等比缩放
DOT_OFFSET_REF_WIDTH: int = 1920          # 校准 DOT_OFFSET_X 时所用的屏幕宽度
RIGHT_PANEL_RATIO: float = 0.62    # 屏幕宽度这个比例以右视作「答题卡面板」，不参与题干/选项解析
MIN_CONFIDENCE: float = 0.55       # OCR 置信度阈值，低于此值的文本行直接丢弃
STEM_MAX_CHARS: int = 1200         # 题干最大长度，超长截断（避免撑爆 token）
OPTION_MERGE_GAP_RATIO: float = 1.8   # 选项续行合并：与上一选项行距 < 行高 × 该比例 => 视作续行
# 题干聚段：和下面那行的行距 > 行高 × 该比例 => 认定上面那块文字不属于本题
# （屏幕上方常挂着浏览器标签栏、地址栏、书签栏，不加这条会把它们全算进题干）
STEM_JOIN_GAP_RATIO: float = 4.0
OPTION_TEXT_MATCH_RATIO: float = 0.75 # 答案文本与选项文本的模糊匹配阈值（SequenceMatcher 相似度）
PRESS_ENTER_FALLBACK: bool = True  # 页面没变化时，是否按 Enter 兜底跳题

# ---------------- 翻页按钮（答完点「下一题」） ----------------
# 答完一题、等过 AFTER_ANSWER_WAIT 之后页面还没动，就先点页面上识别到的「下一题」；
# 点了还是不动，才退回原来的「按 Enter」兜底。
CLICK_NEXT_BUTTON: bool = True     # False = 不点按钮，只按 Enter（老行为）
NEXT_BUTTON_TEXTS: Tuple[str, ...] = ("下一题", "下一页", "下一道", "next")
NEXT_BUTTON_MAX_LEN: int = 8       # 按钮上的字都很短；长过这个数的是说明文字，不是按钮

# ---------------- 页面变化检测 ----------------
PAGE_DIFF_METHOD: str = "phash"       # phash(感知哈希·推荐) / hist(灰度直方图) / gray(缩放灰度均值差)
FINGERPRINT_LEFT_ONLY: bool = True    # 只对左侧题目区做指纹（右侧答题卡答完必变，会误判成跳题）
PHASH_MAX_DISTANCE: int = 10          # pHash 汉明距离 > 该值 => 判定页面已跳转（范围 0~64，越小越敏感）
HIST_MIN_CORRELATION: float = 0.995   # 直方图相关性 < 该值 => 判定页面已跳转（越接近 1 越敏感）
GRAY_DIFF_THRESHOLD: float = 3.0      # gray 模式：平均绝对差 > 该值 => 判定页面已跳转

# ---------------- OCR 引擎 ----------------
# auto      : 先试 RapidOCR，不行再试 PaddleOCR（推荐：RapidOCR 模型自带、不用 paddle，打包成 exe 最省事）
# rapidocr  : 只用 RapidOCR          pip install rapidocr-onnxruntime
# paddleocr : 只用 PaddleOCR         pip install paddlepaddle paddleocr
OCR_BACKEND: str = "auto"

# ---------------- 路径（统一落在 exe / 脚本所在目录，避免从别处启动时文件散落一地） ----------------
def _app_dir() -> str:
    """打包成 exe 后取 exe 所在目录；直接跑源码时取脚本所在目录。"""
    if getattr(sys, "frozen", False):        # PyInstaller 打包后的运行态
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR: str = _app_dir()

# ---------------- 日志 ----------------
LOG_FILE: str = os.path.join(BASE_DIR, "exam_auto_solver.log")
LOG_MAX_BYTES: int = 2 * 1024 * 1024   # 单个日志文件上限 2MB，写满自动轮转
LOG_BACKUP_COUNT: int = 5              # 最多保留 5 个历史日志

# ---------------- 调试开关 ----------------
DRY_RUN: bool = True               # True = 只打印不点击（第一次跑务必 True，用来校准坐标）
SAVE_DEBUG_SCREENSHOT: bool = True # 是否把每轮截图存到 debug_shots/ 便于回头核对
DEBUG_SHOT_KEEP: int = 40          # debug_shots/ 只保留最近 N 张（0 = 不清理）
DEBUG_DIR: str = os.path.join(BASE_DIR, "debug_shots")

# ---------------- pyautogui 全局设置 ----------------
pyautogui.FAILSAFE = True          # 鼠标甩到左上角 (0,0) 紧急中止
pyautogui.PAUSE = 0.05             # 每个 pyautogui 动作后的默认停顿（别设 0，但 0.15 太磨蹭）

# 移动鼠标的过渡时长。原来 0.25 秒是「看起来像人」，但每题要点两次、还要滑动，
# 积起来就是一两秒；0.12 秒对手感没影响，点击照样落得准。
MOUSE_MOVE_DURATION: float = 0.12

# =====================================================================
# 一·五、紧急停止（GUI 的「停止」按钮 / 外部调用 request_stop()）
# =====================================================================

STOP_REQUESTED: bool = False       # 置 True 后：当前轮尽快收尾，主循环退出


def request_stop() -> None:
    """请求停止。GUI 按钮、外部脚本都可以调。"""
    global STOP_REQUESTED
    STOP_REQUESTED = True


def reset_stop() -> None:
    """开始答题前清除停止标记。"""
    global STOP_REQUESTED
    STOP_REQUESTED = False


def sleep_with_stop(seconds: float) -> None:
    """
    分段 sleep：时间照睡，但「停止」最多 0.2 秒就能生效，
    不用等一个 2 秒的间隔睡完才反应。
    """
    end = time.time() + max(0.0, float(seconds))
    while not STOP_REQUESTED:
        remain = end - time.time()
        if remain <= 0:
            return
        time.sleep(min(0.2, remain))


def apply_speed(name: str) -> str:
    """
    按「答题节奏」预设改掉各处等待时间，返回实际生效的档位名（认不出来就用默认档）。
    界面上的下拉、还有直接调脚本的人，都走这一个入口。
    """
    global LOOP_SLEEP, FIRST_CLICK_DELAY, SINGLE_SETTLE_WAIT
    global AFTER_ANSWER_WAIT, CLICK_INTERVAL, _SPEED_NAME

    preset = SPEED_PRESETS.get(str(name).strip())
    if preset is None:
        name = DEFAULT_SPEED
        preset = SPEED_PRESETS[name]

    LOOP_SLEEP = float(preset["loop_sleep"])
    FIRST_CLICK_DELAY = float(preset["first_click_delay"])
    SINGLE_SETTLE_WAIT = float(preset["single_settle"])
    AFTER_ANSWER_WAIT = float(preset["after_answer"])
    CLICK_INTERVAL = float(preset["click_interval"])
    _SPEED_NAME = name
    return name


def speed_text() -> str:
    """给人看的节奏说明。"""
    return ("每轮基础等待 %.1f 秒 / 首次点击前 %.1f 秒 / 答完等 %.1f 秒 / "
            "多选两次点击间隔 %.1f 秒"
            % (LOOP_SLEEP, FIRST_CLICK_DELAY, SINGLE_SETTLE_WAIT, CLICK_INTERVAL))


_SPEED_NAME: str = DEFAULT_SPEED       # 当前生效的档位名（apply_speed 会更新）


def now_speed_name() -> str:
    """当前生效的节奏档位名。"""
    return _SPEED_NAME


# ---------------- 紧急停止热键 ----------------
# 浏览器全屏、切不出后台时，界面上的「停止」按钮点不到 —— 所以再加一层：
# 用 Win32 的 GetAsyncKeyState 轮询键盘的物理状态。它跟窗口焦点无关，
# 焦点在谁身上都读得到，所以全屏页面里照样能停。
PANIC_STOP_ENABLE: bool = True          # 关掉就只剩「停止」按钮 + 鼠标甩左上角
PANIC_ESC_COUNT: int = 3                # 连按几次 Esc 触发（0 = 不用这个方式）
PANIC_ESC_WINDOW: float = 1.5           # 这几次必须在多少秒内按完
PANIC_HOTKEYS: Tuple[str, ...] = ("ctrl+alt+q",)   # 组合键，可写多个

_VK_SPECIAL: Dict[str, int] = {
    "ctrl": 0x11, "control": 0x11, "alt": 0x12, "shift": 0x10,
    "esc": 0x1B, "escape": 0x1B, "space": 0x20, "tab": 0x09,
    "enter": 0x0D, "return": 0x0D, "backspace": 0x08, "delete": 0x2E,
    "insert": 0x2D, "home": 0x24, "end": 0x23, "pageup": 0x21,
    "pagedown": 0x22, "pause": 0x13, "capslock": 0x14,
}
_VK_SPECIAL.update({("f%d" % i): 0x6F + i for i in range(1, 13)})

_PANIC_THREAD: Optional[threading.Thread] = None


def parse_hotkey(text: str) -> List[int]:
    """把 "ctrl+alt+q" 解析成虚拟键码列表；看不懂就返回空列表。"""
    out: List[int] = []
    for part in str(text or "").lower().replace(" ", "").split("+"):
        if not part:
            continue
        if part in _VK_SPECIAL:
            out.append(_VK_SPECIAL[part])
        elif len(part) == 1 and (part.isalpha() or part.isdigit()):
            out.append(ord(part.upper()))
        else:
            return []
    return out


def panic_watch_loop(is_down, combos, should_stop) -> Optional[str]:
    """
    盯着热键，触发就返回原因（一直没触发过就返回 None）。

    is_down(vk) -> bool 和 should_stop() -> bool 由调用方给：
    真机上是 GetAsyncKeyState + STOP_REQUESTED，测试里塞假的就能离线验。
    """
    prev_esc = False
    prev_combo = [False] * len(combos)
    esc_hits: List[float] = []

    while not should_stop():
        now = time.time()
        esc = is_down(0x1B)
        if esc and not prev_esc:                      # 只在「按下」的那一下计数
            esc_hits.append(now)
            esc_hits = [t for t in esc_hits if now - t <= PANIC_ESC_WINDOW]
            if PANIC_ESC_COUNT > 0 and len(esc_hits) >= PANIC_ESC_COUNT:
                return "Esc 连按 %d 次" % PANIC_ESC_COUNT
        prev_esc = esc

        for i, (text, vks) in enumerate(combos):
            cur = all(is_down(v) for v in vks)
            if cur and not prev_combo[i]:             # 组合键按齐的那一下
                return "热键 %s" % text
            prev_combo[i] = cur

        time.sleep(0.05)
    return None


def panic_hotkey_text() -> str:
    """给人看的紧急停止说明。"""
    parts: List[str] = []
    if PANIC_ESC_COUNT > 0:
        parts.append("连按 %d 次 Esc" % PANIC_ESC_COUNT)
    parts += ["按 %s" % h.upper() for h in PANIC_HOTKEYS if h]
    return "，或".join(parts) if parts else "（未配置）"


def win32_key_state():
    """
    返回 (is_down, 错误原因)：真机上用 Win32 的 GetAsyncKeyState。
    抽成独立函数是为了能离线测试 —— 测试里塞一个假的 is_down 就能验整个看门狗。
    """
    try:
        import ctypes
        user32 = ctypes.windll.user32
    except Exception as exc:                 # 非 Windows / ctypes 不可用
        return None, exc
    return (lambda vk: bool(user32.GetAsyncKeyState(vk) & 0x8000)), None


def start_panic_watchdog() -> None:
    """起后台线程盯紧急停止热键；重复调用只会起一个。"""
    global _PANIC_THREAD
    if not PANIC_STOP_ENABLE or _PANIC_THREAD is not None:
        return
    is_down, err = win32_key_state()
    if is_down is None:
        log.warning("紧急停止热键起不来（%s）：仍可用界面「停止」按钮，"
                    "或把鼠标甩到屏幕左上角", err)
        return

    combos: List[Tuple[str, List[int]]] = []
    for text in PANIC_HOTKEYS:
        vks = parse_hotkey(text)
        if vks:
            combos.append((text, vks))
        else:
            log.warning("紧急停止热键写法看不懂，已忽略：%s", text)

    def work() -> None:
        reason = panic_watch_loop(is_down, combos, lambda: STOP_REQUESTED)
        if reason:
            log.warning("=" * 68)
            log.warning("【紧急停止】检测到 %s —— 立即停止，不再点鼠标", reason)
            log.warning("=" * 68)
            request_stop()

    _PANIC_THREAD = threading.Thread(target=work, daemon=True, name="panic-watchdog")
    _PANIC_THREAD.start()
    log.info("紧急停止已就绪：%s（浏览器全屏、切不出后台时也有效）", panic_hotkey_text())

# =====================================================================
# 二、日志
# =====================================================================

def _build_log_handlers() -> List[logging.Handler]:
    """
    控制台 + 轮转文件双输出。
    日志文件写满 LOG_MAX_BYTES 自动切分、只保留 LOG_BACKUP_COUNT 个历史文件，
    整场考试跑下来也不会把磁盘写爆。
    """
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(
            RotatingFileHandler(
                LOG_FILE,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
        )
    except Exception as e:
        # 文件被占用 / 没写权限时，退化成只输出到控制台，别让脚本起不来
        print("[warn] 日志文件初始化失败，本次仅输出到控制台：%s" % e, file=sys.stderr)
    return handlers


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=_build_log_handlers(),
)
log = logging.getLogger("exam_bot")

# 抑制 paddle 自带的 DEBUG 刷屏
logging.getLogger("ppocr").setLevel(logging.ERROR)

# =====================================================================
# 三、题型常量
# =====================================================================

TYPE_SINGLE = "single"      # 单选题
TYPE_MULTI = "multiple"     # 多选题
TYPE_JUDGE = "judge"        # 判断题

# 关键词匹配顺序很重要：先判多选/判断，最后才判单选
TYPE_KEYWORDS: List[Tuple[str, List[str]]] = [
    (TYPE_MULTI, ["多选题", "多选", "多项选择", "不定项", "多项"]),
    (TYPE_JUDGE, ["判断题", "判断", "对错题", "判断正误"]),
    (TYPE_SINGLE, ["单选题", "单选", "单项选择"]),
]

TYPE_NAME_CN = {TYPE_SINGLE: "单选题", TYPE_MULTI: "多选题", TYPE_JUDGE: "判断题"}

# 选项行正则：形如 "A. xxx" / "（B）xxx" / "C、xxx" / "D：xxx" / "○A、xxx"
# 行首常常还挂着那个单选/多选的小圆圈（OCR 会把圆圈和文字并成一行，如 "○A、立即到…"），
# 所以字母前面先允许跳过一圈符号；也可能被 OCR 认成字母 O 或数字 0。
# 仍然必须带分隔符、必须真的是 A-H 字母，避免把题干里普通英文单词开头的行误判成选项。
OPTION_LETTER_RE = re.compile(
    r"^\s*[（(\[【]?\s*[○●◯〇⚪⚫◉◎⊙Oo0•·※*⁕\-–—]*\s*"
    r"([A-Ha-h])\s*[)）\]】]?\s*[、.,，:：．]\s*(.*)$"
)

# 行首是这些「圆圈类」符号时，它本身就是那个可点的小圆点，
# 不再需要靠 DOT_OFFSET_X 去「估算」它在哪里。
RADIO_GLYPH_RE = re.compile(r"[○●◯〇◉◎⊙⚪⚫Oo0]")

# 整行就是一个圆圈符号（OCR 把圆圈和选项文字切成了两个框时的样子）
RADIO_ONLY_RE = re.compile(r"^[○●◯〇◉◎⊙⚪⚫Oo0•·※*⁕\-–—]+$")

# 「这一行是题目的开头」的判定：题型标签后面要跟正文（[单选题]1、xxx），
# 或者干脆以题号开头（1、xxx / 1.xxx）。
# 单独一行的「（多选题）」只是题型标注，不算题目开头。
# 题号后面那个分隔符各家写法不一：1.【单选题】 / 1、xxx / 6【多选题】 (2分)，
# 最后这种（数字直接跟题型框、中间没有点或顿号）最容易被漏掉。
QUESTION_HEAD_RE = re.compile(
    r"^\s*(?:[\[【]\s*(?:单选|多选|判断|不定项)\s*题?\s*[\]】]\s*\S"
    r"|\d{1,3}\s*[、.．]\s*\S"
    r"|\d{1,3}\s*[\[【]\s*(?:单选|多选|判断|不定项)\s*题?\s*[\]】])"
)

# 浏览器自己的东西：标签页标题、地址栏。它们永远在「选项上方」，
# 题目开头又没认出来的时候会被一股脑算进题干，必须剔除。
BROWSER_NOISE_RE = re.compile(r"(https?://|www\.|\.com[/\s]|\.cn[/\s]|\.html|#/)", re.I)

# 判断题选项（整行就是一个"正确/错误"）
JUDGE_TRUE_SET = {"正确", "对", "是", "√", "对的", "正确的", "true", "t", "yes"}
JUDGE_FALSE_SET = {"错误", "错", "否", "×", "x", "错的", "错误的", "false", "f", "no"}


# =====================================================================
# 四、基础工具函数
# =====================================================================

def enable_dpi_awareness() -> None:
    """
    Windows 缩放（125% / 150%）会导致 pyautogui 拿到的是逻辑分辨率，
    而截图是物理分辨率，两边坐标对不上，点击就会偏。
    这里把进程标记为 DPI 感知，让截图与坐标统一到物理像素。
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)   # Win8.1+
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()        # 老系统兜底
        log.info("已开启 DPI 感知（避免高分屏点击偏移）")
    except Exception as e:
        log.warning("DPI 感知设置失败（不影响低缩放屏幕）：%s", e)


def _init_rapidocr():
    """
    RapidOCR：PP-OCR 模型的 ONNX 版，模型文件随包自带、只依赖 onnxruntime，
    不用装几百 MB 的 paddlepaddle —— 打包成 exe 时这个最省事也最稳。
    """
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    log.info("RapidOCR 初始化成功（ONNX Runtime + PP-OCR 模型，模型随包自带）")
    return ("rapidocr", engine)


def _init_paddleocr():
    """PaddleOCR。不同版本参数名有差异，逐个降级尝试。"""
    from paddleocr import PaddleOCR

    candidates = [
        dict(use_angle_cls=True, lang="ch", show_log=False),
        dict(use_angle_cls=True, lang="ch"),
        dict(lang="ch"),
        dict(use_angle_cls=True, lang="ch", enable_mkldnn=False),
    ]
    for kwargs in candidates:
        try:
            ocr = PaddleOCR(**kwargs)
            log.info("PaddleOCR 初始化成功，参数：%s", kwargs)
            return ("paddleocr", ocr)
        except Exception as e:
            log.debug("PaddleOCR 初始化尝试失败 %s：%s", kwargs, e)

    raise RuntimeError("PaddleOCR 初始化失败：paddlepaddle 与 paddleocr 版本可能不兼容")


def init_ocr(backend: str = OCR_BACKEND):
    """
    初始化 OCR，返回 (引擎名, 引擎对象)。

    backend = auto 时先试 RapidOCR（自带模型、不用装 paddle），失败再退回 PaddleOCR。
    """
    order = {
        "auto": ["rapidocr", "paddleocr"],
        "rapidocr": ["rapidocr"],
        "paddleocr": ["paddleocr"],
    }.get(str(backend or "auto").strip().lower(), ["rapidocr", "paddleocr"])

    errors: List[str] = []
    for name in order:
        try:
            return _init_rapidocr() if name == "rapidocr" else _init_paddleocr()
        except Exception as e:
            errors.append("%s -> %s" % (name, e))
            log.debug("%s 初始化失败：%s", name, e)

    raise RuntimeError(
        "OCR 引擎初始化失败。任选一个装好即可：\n"
        "  pip install rapidocr-onnxruntime       （推荐：模型自带，打包 exe 最省事）\n"
        "  pip install paddlepaddle paddleocr     （paddleocr 2.x 配 paddlepaddle 2.5/2.6 最稳）\n"
        "失败详情：" + "；".join(errors)
    )


def grab_screen() -> Tuple[np.ndarray, float, float]:
    """
    截取整块屏幕。

    返回:
        img_bgr  : OpenCV 用的 BGR 图像
        scale_x  : 屏幕逻辑宽 / 截图宽（一般 = 1.0，缩放异常时用于坐标换算）
        scale_y  : 屏幕逻辑高 / 截图高
    """
    shot = pyautogui.screenshot()
    img_rgb = np.array(shot)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    screen_w, screen_h = pyautogui.size()
    h, w = img_bgr.shape[:2]
    scale_x = screen_w / float(w) if w else 1.0
    scale_y = screen_h / float(h) if h else 1.0
    return img_bgr, scale_x, scale_y


def imwrite_unicode(path: str, img_bgr: np.ndarray) -> bool:
    """
    保存图片。Windows 上 cv2.imwrite 碰到带中文的路径会静默失败（只返回 False，不报错），
    所以这里自己编码 + 用 Python 写文件，中文目录也能正常落盘。
    """
    try:
        ext = os.path.splitext(path)[1] or ".png"
        ok, buf = cv2.imencode(ext, img_bgr)
        if not ok:
            return False
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception as e:
        log.debug("保存图片失败 %s：%s", path, e)
        return False


def save_debug_image(img_bgr: np.ndarray, step: int) -> None:
    """把本轮截图存下来，方便事后核对 OCR 和坐标是否算对了。"""
    if not SAVE_DEBUG_SCREENSHOT:
        return
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        path = os.path.join(DEBUG_DIR, f"step_{step:03d}.png")
        if imwrite_unicode(path, img_bgr):
            log.info("调试截图已保存：%s", path)
        else:
            log.warning("调试截图保存失败：%s", path)
        _prune_debug_images()
    except Exception as e:
        log.debug("保存调试截图失败：%s", e)


def _prune_debug_images() -> None:
    """
    只保留最近 DEBUG_SHOT_KEEP 张调试截图。
    一场考试几百轮、每轮一张全屏 PNG，不清理的话能写出好几个 GB 来。
    """
    if DEBUG_SHOT_KEEP <= 0:
        return
    try:
        shots = sorted(glob.glob(os.path.join(DEBUG_DIR, "step_*.png")))
        for old in shots[:-DEBUG_SHOT_KEEP]:
            os.remove(old)
    except Exception as e:
        log.debug("清理旧调试截图失败：%s", e)


# =====================================================================
# 五、OCR 调用与结果解析
# =====================================================================

def ocr_image(engine, img_bgr: np.ndarray) -> Optional[Any]:
    """调用 OCR 引擎，兼容 RapidOCR / PaddleOCR 新旧版本的调用签名。"""
    kind, ocr = engine

    if kind == "rapidocr":
        try:
            result = ocr(img_bgr)
        except Exception as e:
            log.error("RapidOCR 调用失败：%s", e)
            return None
        # 标准签名返回 (results, elapse)，个别版本只返回 results
        return result[0] if isinstance(result, tuple) else result

    try:
        return ocr.ocr(img_bgr, cls=True)
    except Exception as e1:
        log.debug("ocr(cls=True) 调用失败：%s，改用无 cls 调用", e1)
        try:
            return ocr.ocr(img_bgr)
        except Exception as e2:
            log.error("OCR 调用失败：%s", e2)
            return None


def _box_to_line(box, text: str, score: float) -> Dict[str, float]:
    """把一个四边形框 + 文本，归一化成便于计算的行结构。"""
    arr = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    xs, ys = arr[:, 0], arr[:, 1]
    return {
        "text": str(text).strip(),
        "x1": float(xs.min()),
        "y1": float(ys.min()),
        "x2": float(xs.max()),
        "y2": float(ys.max()),
        "cx": float(xs.mean()),   # 文字中心 X
        "cy": float(ys.mean()),   # 文字中心 Y
        "score": float(score),
    }


def parse_ocr_result(raw: Any) -> List[Dict[str, float]]:
    """
    把 PaddleOCR 五花八门的返回结构统一成 List[行]。

    兼容：
        PaddleOCR 2.x : [[ [box, (text, score)], ... ]]
        PaddleOCR 3.x : [ {"rec_texts": [...], "rec_scores": [...], "rec_polys": [...]} ]
    """
    lines: List[Dict[str, float]] = []
    if not raw:
        return lines

    first = raw[0] if isinstance(raw, list) and raw else raw

    # ---- RapidOCR：[[box, "文本", 0.98], ...] ----
    if (isinstance(first, (list, tuple)) and len(first) == 3
            and isinstance(first[1], str)):
        for item in raw:
            try:
                box, text, score = item[0], item[1], item[2]
            except Exception:
                continue
            if not str(text).strip() or float(score) < MIN_CONFIDENCE:
                continue
            lines.append(_box_to_line(box, text, float(score)))
        lines.sort(key=lambda d: (d["cy"], d["x1"]))
        return lines

    # ---- 3.x：字典结构 ----
    if isinstance(first, dict) and ("rec_texts" in first or "rec_polys" in first):
        for page in (raw if isinstance(raw, list) else [raw]):
            if not isinstance(page, dict):
                continue
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            polys = page.get("rec_polys")
            if polys is None:
                polys = page.get("dt_polys")
            for i, t in enumerate(texts):
                sc = float(scores[i]) if i < len(scores) else 1.0
                if sc < MIN_CONFIDENCE or not str(t).strip():
                    continue
                if polys is None or i >= len(polys):
                    continue
                lines.append(_box_to_line(polys[i], t, sc))
        lines.sort(key=lambda d: (d["cy"], d["x1"]))
        return lines

    # ---- 2.x：嵌套列表结构 ----
    page = raw[0] if (isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list)) else raw
    if not isinstance(page, list):
        return lines

    for item in page:
        try:
            box = item[0]
            text, score = item[1][0], item[1][1]
        except Exception:
            continue
        if not str(text).strip() or float(score) < MIN_CONFIDENCE:
            continue
        lines.append(_box_to_line(box, text, float(score)))

    # 按从上到下、从左到右排序
    lines.sort(key=lambda d: (d["cy"], d["x1"]))
    return lines


# =====================================================================
# 六、题干 / 选项 / 题型解析
# =====================================================================

def _is_judge_option_text(text: str) -> bool:
    t = text.strip().lower()
    if t in JUDGE_TRUE_SET or t in JUDGE_FALSE_SET:
        return True
    # 容忍 OCR 带点小尾巴，如 "正确√"、"错误×"；
    # 但必须够短——"正确率最高的选项是…" 这种题干绝不能被当成判断题选项。
    if len(t) > 4:
        return False
    return t.startswith(("正确", "错误", "√", "×"))


def _estimate_line_height(lines: List[Dict[str, float]]) -> float:
    """用所有文本行高度的中位数估算行高，用于判断两行是不是同一段。"""
    heights = [l["y2"] - l["y1"] for l in lines if l["y2"] > l["y1"]]
    if not heights:
        return 24.0
    return float(np.median(heights))


def _join_stem_rows(rows: List[str]) -> str:
    """把多行题干拼成一段。中文行直接相接，中英文交界/英文之间补个空格，别把单词粘死。"""
    out = ""
    for row in rows:
        if not out:
            out = row
            continue
        if re.search(r"[A-Za-z0-9]$", out) and re.match(r"[A-Za-z0-9]", row):
            out += " " + row
        else:
            out += row
    return out


def parse_question(lines: List[Dict[str, float]], img_w: float) -> Optional[Dict[str, Any]]:
    """
    从一堆 OCR 行里解析出：题干、选项（含小圆点估算坐标）、题型。

    选项续行合并：OCR 常把一条长选项切成两行，这里把紧跟上一个选项、行距又够近的
    非选项行并回该选项的文本里——既补全选项内容，也避免这段文字被当成「题干残留」。

    返回 None 表示这一屏没解析出有效题目。
    """
    if not lines:
        return None

    # 只保留左侧题目区，右侧答题卡不参与解析
    left_lines = [l for l in lines if l["cx"] < img_w * RIGHT_PANEL_RATIO]
    if not left_lines:
        left_lines = lines
    # 按从上到下、从左到右排好，后面按顺序扫
    left_lines = sorted(left_lines, key=lambda d: (d["cy"], d["x1"]))

    line_h = _estimate_line_height(left_lines)
    merge_gap = line_h * OPTION_MERGE_GAP_RATIO

    # 小圆点偏移量：屏幕比基准宽时按比例放大（高分屏下写死 35px 会偏到选项文字上）
    dot_offset = float(DOT_OFFSET_X)
    if DOT_OFFSET_SCALE_BY_WIDTH and DOT_OFFSET_REF_WIDTH > 0:
        dot_offset *= float(img_w) / float(DOT_OFFSET_REF_WIDTH)

    # ---- 1) 扫选项行，顺便把续行合并回上一个选项 ----
    options: List[Dict[str, float]] = []
    first_option_top: Optional[float] = None    # 选项块顶部（命名：top 而不是 cy，它就是行顶坐标）
    seen_labels = set()

    for l in left_lines:
        text = l["text"].strip()
        m = OPTION_LETTER_RE.match(text)
        label: Optional[str] = None
        opt_text = ""

        if m:
            cand = m.group(1).upper()
            if cand not in seen_labels:
                label = cand
                opt_text = (m.group(2) or "").strip()
            else:
                # 同一个字母出现两次：OCR 把圆点/字母单独切成了一行，按续行处理
                label = None
                opt_text = text
        elif _is_judge_option_text(text) and text not in seen_labels:
            # 判断题：整行就是"正确"/"错误"
            label = text
            opt_text = text

        if label is not None:
            if first_option_top is None:
                first_option_top = l["y1"]
            seen_labels.add(label)
            # 行首带圆圈符号时，那个圆圈就是要点的圆点：它紧贴在文字框左边缘，
            # 圆心约在 x1 + 半个字高处（圆圈宽度≈字高，实测 156 + 23/2 ≈ 167）。
            # 这样比「文字左边缘往左推 DOT_OFFSET_X」准得多，换分辨率也不会偏。
            # 没有圆圈符号才退回偏移量估算。
            prefix = text[: m.start(1)] if m else ""
            if RADIO_GLYPH_RE.search(prefix):
                dot_x = l["x1"] + (l["y2"] - l["y1"]) / 2.0
            else:
                dot_x = l["x1"] - dot_offset
            options.append({
                "label": label,
                "text": opt_text or label,
                "dot_x": dot_x,
                "dot_y": l["cy"],
                "row_y": l["y1"],
                # 下面几个字段给「续行合并」用，同时也方便调试
                "top": l["y1"],
                "bottom": l["y2"],
                "x1": l["x1"],
                "x2": l["x2"],
            })
            continue

        # ---- 不是新选项：行距够近就当上一选项的续行合并，别丢也别塞给题干 ----
        if options and (l["y1"] - options[-1]["bottom"]) <= merge_gap:
            prev = options[-1]
            prev["text"] = (prev["text"] + text).strip()
            prev["bottom"] = max(prev["bottom"], l["y2"])
            prev["x2"] = max(prev["x2"], l["x2"])
            log.debug("选项 %s 合并续行：%s", prev["label"], text)
            continue

    if not options:
        return None

    # 有些页面 OCR 会把圆圈单独切成一个框（"○" 和 "A、xxx" 分成两行），
    # 这时选项行的 x1 是字母的位置，按 DOT_OFFSET_X 推出来的点会落在圆圈外面。
    # 回查一次：每个选项往左边找有没有「孤零零的圆圈」行，有就直接用它的中心当圆点。
    stray_radios = [l for l in left_lines if RADIO_ONLY_RE.match(l["text"].strip())]
    if stray_radios:
        for opt in options:
            best: Optional[Dict[str, float]] = None
            for r in stray_radios:
                if abs(r["cy"] - opt["dot_y"]) > line_h:
                    continue
                if not 0.0 < (opt["x1"] - r["cx"]) < line_h * 4:
                    continue
                if best is None or abs(r["cy"] - opt["dot_y"]) < abs(best["cy"] - opt["dot_y"]):
                    best = r
            if best is not None:
                log.debug("选项 %s 找到独立圆点框，用它的中心 (%.0f, %.0f) 当点击点",
                          opt["label"], best["cx"], best["cy"])
                opt["dot_x"] = best["cx"]
                opt["dot_y"] = best["cy"]

    # ---- 2) 题干 = 第一个选项行上方的所有文字（选项续行上面已经合并走了，不会混进来）----
    boundary = (first_option_top if first_option_top is not None else 0.0) - 2

    # 先捞候选行：题干区里除「像选项的行」之外的所有行
    cand: List[Dict[str, float]] = []
    for l in left_lines:
        if l["y1"] >= boundary:
            continue
        t = l["text"].strip()
        if not t:
            continue
        # 排除被误判成选项的残留行
        if OPTION_LETTER_RE.match(t):
            continue
        # 排除浏览器标签/地址栏：题目开头没认出来时它们会被算进题干，
        # 送进大模型就是一堆与题目无关的网址噪声
        if BROWSER_NOISE_RE.search(t):
            log.debug("题干候选里剔除浏览器噪声行：%s", t[:40])
            continue
        cand.append(l)

    # 题干该从哪一行算起？「选项上方」往往还有一堆无关文字：浏览器标签栏、地址栏、
    # 书签栏、页面统计行……全塞进题干既脏，还可能把真正的问题挤出 STEM_MAX_CHARS。
    # 优先找最近的一个「题目开头」行（[单选题] / 【多选题】 / 1、 / 1.）从它开始；
    # 找不到就退回「按行距聚成一段」：往上走，行距一旦明显变大就说明上面那块不是本题的。
    start = 0
    for i in range(len(cand) - 1, -1, -1):
        if QUESTION_HEAD_RE.match(cand[i]["text"].strip()):
            start = i
            break
    else:
        gap_limit = line_h * STEM_JOIN_GAP_RATIO
        start = len(cand)
        prev_top = boundary + 2.0
        while start > 0:
            row = cand[start - 1]
            if (prev_top - row["y2"]) > gap_limit:
                break
            start -= 1
            prev_top = row["y1"]

    stem_rows: List[str] = [l["text"].strip() for l in cand[start:]]

    stem = _join_stem_rows(stem_rows).strip()
    if not stem:
        # 题干一个字都没读到 —— 这时"选项"多半是误判（比如题干自己命中了字母模式），
        # 硬答只会答错，直接放弃本轮等下一次截图。
        log.warning("题干区域没识别到任何文字，放弃本轮")
        return None
    stem = stem[:STEM_MAX_CHARS]

    # ---- 3) 判断题型 ----
    qtype = None
    joined = stem
    for t, kws in TYPE_KEYWORDS:
        if any(k in joined for k in kws):
            qtype = t
            break

    if qtype is None:
        # OCR 没读到题型标签时，靠选项形态兜底判断
        judge_like = all(_is_judge_option_text(o["label"]) for o in options)
        if judge_like:
            qtype = TYPE_JUDGE
        else:
            qtype = TYPE_SINGLE
        log.warning("未识别到题型标签，按选项形态兜底判定为：%s", TYPE_NAME_CN[qtype])

    return {
        "stem": stem,
        "options": options,
        "type": qtype,
        "first_option_top": first_option_top,
    }


# =====================================================================
# 七、大模型解题
# =====================================================================

SYSTEM_PROMPT = (
    "你是一名极其严谨的考试作答专家。"
    "你必须根据题干和选项给出正确答案，并且只输出 JSON，不要输出任何解释、"
    "不要输出 Markdown 代码块、不要输出多余文字。"
    "JSON 格式固定为：{\"type\": \"single|multiple|judge\", \"answer\": [\"A\"]}。"
    "单选题 type=single，answer 数组里只能有一个字母；"
    "多选题 type=multiple，answer 数组里放全部正确字母，按字母升序；"
    "判断题 type=judge，answer 数组里放选项原文（如 [\"正确\"] 或 [\"错误\"])。"
)


def build_user_prompt(stem: str, options: List[Dict[str, float]], qtype: str) -> str:
    """把题干 + 选项拼成给大模型的提问。"""
    option_lines = []
    for o in options:
        option_lines.append(f"{o['label']}. {o['text']}")
    option_block = "\n".join(option_lines)

    return (
        f"题型：{TYPE_NAME_CN.get(qtype, '单选题')}\n"
        f"题干：{stem}\n"
        f"选项：\n{option_block}\n\n"
        "请判断正确答案，并严格按以下 JSON 结构返回（不要 Markdown 代码块）：\n"
        "{\"type\": \"single\", \"answer\": [\"A\"]}"
    )


def extract_json(text: str) -> Dict[str, Any]:
    """从模型返回里抠出 JSON（容忍 ```json 包裹和前后废话）。"""
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```$", "", s).strip()

    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        raise ValueError(f"模型返回里找不到 JSON：{s[:200]}")
    return json.loads(m.group(0))


# 探测出来的可用 Base URL。填错过地址时，探到一次之后就固定用它，省得每题都试错。
RESOLVED_BASE_URL: str = ""


def normalize_base_url(base_url: str) -> str:
    """
    把 Base URL 收拾成「能直接拼 /chat/completions」的形式：
        https://api.deepseek.com/v1/chat/completions -> https://api.deepseek.com/v1
        https://api.deepseek.com/chat/completions    -> https://api.deepseek.com
        https://api.deepseek.com/                    -> https://api.deepseek.com
    很多人是把文档里整段接口地址粘进来的，这一步能避免拼出 /chat/completions/chat/completions。
    """
    text = (base_url or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)]
    return text.rstrip("/")


def base_url_candidates(base_url: str) -> List[str]:
    """
    同一个服务，地址带不带 /v1 两种写法都很常见
    （DeepSeek 两种都收，OpenAI 必须带 /v1），所以给出两三个候选挨个试。
    """
    text = normalize_base_url(base_url)
    if not text:
        return []
    candidates = [text, text[:-3].rstrip("/") if text.lower().endswith("/v1") else text + "/v1"]
    ordered: List[str] = []
    for item in candidates:
        if item and item not in ordered:
            ordered.append(item)
    return ordered


def _chat_completions_urls() -> List[str]:
    """要试的 /chat/completions 地址：已探明可用就只用它，否则按候选顺序挨个试。"""
    bases = [RESOLVED_BASE_URL] if RESOLVED_BASE_URL else base_url_candidates(BASE_URL)
    return [base.rstrip("/") + "/chat/completions" for base in bases]


def resolve_base_url(base_url: str, api_key: str, model: str,
                     timeout: int = 20) -> Tuple[str, str]:
    """
    拿一次最小请求去探路，返回 (可用的 Base URL, 说明文字)：
    地址通就返回它（顺带把模型回复放在说明里），都不通就返回空串 + 错误详情。
    """
    global RESOLVED_BASE_URL

    if not api_key:
        return "", "API Key 是空的"
    if not model:
        return "", "模型名是空的"
    candidates = base_url_candidates(base_url)
    if not candidates:
        return "", "Base URL 是空的"

    original = normalize_base_url(base_url)
    errors: List[str] = []
    for candidate in candidates:
        url = candidate + "/chat/completions"
        try:
            resp = requests.post(
                url,
                headers={"Authorization": "Bearer " + api_key,
                         "Content-Type": "application/json"},
                json={"model": model,
                      "temperature": 0,
                      "messages": [{"role": "user", "content": "只回复两个字：可用"}]},
                timeout=max(5, int(timeout)),
            )
        except Exception as e:
            errors.append("%s 连不上（%s：%s）" % (url, type(e).__name__, e))
            continue
        if resp.status_code < 400:
            RESOLVED_BASE_URL = candidate
            try:
                reply = str(resp.json()["choices"][0]["message"]["content"]).strip()
            except Exception:
                reply = "(返回体不是标准 OpenAI 格式)"
            fixed = "" if candidate == original else "，已自动纠正地址"
            return candidate, "%s 可用%s，模型回复：%s" % (url, fixed, reply[:60])
        errors.append("HTTP %s：%s" % (resp.status_code, resp.text[:200]))

    return "", "；".join(errors[:2])


def call_llm(stem: str, options: List[Dict[str, float]], qtype: str) -> Dict[str, Any]:
    """请求大模型拿答案，返回解析后的 dict。失败会抛异常，由上层捕获跳过本题。"""
    if not API_KEY or API_KEY.startswith("sk-把你的"):
        raise RuntimeError("未配置大模型 API Key（请设置环境变量 OPENAI_API_KEY）")

    global RESOLVED_BASE_URL
    urls = _chat_completions_urls()
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(stem, options, qtype)},
        ],
        # 强制 JSON 输出（部分网关不支持该参数，失败会自动降级重试）
        "response_format": {"type": "json_object"},
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, LLM_RETRY + 1):
        for idx, url in enumerate(urls):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
                # 只有「请求本身不合法」的 400/422 才是网关不认某个参数；
                # 429 限流 / 5xx 服务端故障 / 401 Key 错，去参数也没用，直接走外层重试。
                if resp.status_code in (400, 422):
                    if "response_format" in payload:
                        log.info("网关返回 %d，疑似不支持 response_format，去掉该参数重试一次",
                                 resp.status_code)
                        payload.pop("response_format", None)
                        resp = requests.post(url, headers=headers, json=payload,
                                             timeout=LLM_TIMEOUT)
                    elif "temperature" in payload:
                        log.info("网关返回 %d，疑似不支持 temperature（推理类模型常见），"
                                 "去掉该参数重试一次", resp.status_code)
                        payload.pop("temperature", None)
                        resp = requests.post(url, headers=headers, json=payload,
                                             timeout=LLM_TIMEOUT)

                # 地址写错最典型的表现就是 404（多写了 /chat/completions、或少了 /v1），
                # 换个候选地址再试一次，别让整场考试都败在这一处。
                if resp.status_code == 404 and idx + 1 < len(urls):
                    log.warning("地址 %s 返回 404，换候选地址 %s 再试", url, urls[idx + 1])
                    continue

                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                data = extract_json(content)
                RESOLVED_BASE_URL = url[: -len("/chat/completions")]   # 记下来，后续直接用它
                log.info("大模型原始返回：%s", content.strip().replace("\n", " ")[:160])
                return data
            except Exception as e:
                last_err = e
                log.warning("大模型调用失败（第 %d/%d 次，%s）：%s", attempt, LLM_RETRY, url, e)
        time.sleep(1.5)

    raise RuntimeError(f"大模型接口连续失败：{last_err}")


# =====================================================================
# 八、答案 -> 选项索引 -> 坐标
# =====================================================================

TRUTH_WORDS = {"正确", "对", "是", "√", "true", "t", "yes", "y", "1", "对的", "正确的"}
FALSE_WORDS = {"错误", "错", "否", "×", "x", "false", "f", "no", "n", "0", "错的", "错误的"}


_TXT_CLEAN_RE = re.compile(r"[\s,，.。、;；:：!！?？'\"“”‘’()（）\[\]【】<>《》\-—_/\\|]+")


def _normalize_text(s: str) -> str:
    """去掉空白和标点、统一小写，用于文本比对（避免标点差异导致匹配不上）。"""
    return _TXT_CLEAN_RE.sub("", str(s)).lower()


def _strip_option_prefix(s: str) -> str:
    """
    剥掉答案里自带的选项前缀："A. 软件测试" -> "软件测试"。
    模型偶尔会把整个选项原文（连字母带点）抄回来，不剥掉就再也匹配不上选项。
    """
    m = OPTION_LETTER_RE.match(str(s).strip())
    if m and (m.group(2) or "").strip():
        return m.group(2).strip()
    return str(s).strip()


def find_next_button(lines: List[Dict[str, float]]) -> Optional[Dict[str, Any]]:
    """
    在整屏 OCR 行里找「下一题」这类翻页按钮，返回它文字框的中心坐标。

    两个要点：
      1) 整屏找，不套 RIGHT_PANEL_RATIO。这类按钮常常正好压在「右侧面板」那一片，
         用左侧题目区的过滤条件会把按钮本身滤掉。
      2) 必须够短。页面底部常有「点击下一题继续答题」这种说明文字，
         不加长度限制就会被当成按钮去点。

    找不到返回 None。
    """
    hits: List[Tuple[bool, Dict[str, float]]] = []
    for l in lines:
        text = _normalize_text(l["text"])
        if not text or len(text) > NEXT_BUTTON_MAX_LEN:
            continue
        if not any(text == kw or text.startswith(kw) or text.endswith(kw)
                   for kw in NEXT_BUTTON_TEXTS):
            continue
        hits.append((text in NEXT_BUTTON_TEXTS, l))
    if not hits:
        return None
    # 优先「整行就等于下一题」的；同样精确时取最靠下的（按钮一般在页面底部）。
    # 排序键里 exists 用 True/False 排序，False 在前，正好把精确命中的排前面。
    hits.sort(key=lambda item: (not item[0], -item[1]["cy"]))
    return hits[0][1]


def _question_text(stem: str, options: List[Dict[str, float]]) -> str:
    """
    「同一道题」的文本特征 = 题干 + 全部选项文本（去掉空白和标点）。
    比原来的 stem[:80] 靠得住：两道题干开头一样、只有后半段不同的题，
    不会再被错认成同一道。
    """
    return _normalize_text(stem) + "|" + "|".join(
        o["label"] + ":" + _normalize_text(o["text"]) for o in options
    )


def _question_key(stem: str, options: List[Dict[str, float]]) -> str:
    """题目身份（skip_counter 用的紧凑 key）。"""
    return hashlib.md5(_question_text(stem, options).encode("utf-8")).hexdigest()[:12]


def _is_same_question(text_a: Optional[str], text_b: Optional[str]) -> bool:
    """
    判断两段文本是不是同一道题。
    刻意不用「完全相等」：OCR 对同一屏文字偶尔会抖一两个字（多认一个标点、
    少认一个空格），完全相等的判定会漏，于是同一道题被当成新题、重复点击。
    改成相似度阈值后，抖一两个字照样认得出，而不同题目之间的相似度远低于阈值。
    """
    if not text_a or not text_b:
        return False
    if text_a == text_b:
        return True
    if abs(len(text_a) - len(text_b)) > max(8, int(len(text_a) * 0.3)):
        return False
    return difflib.SequenceMatcher(None, text_a, text_b).ratio() >= SAME_QUESTION_TEXT_RATIO


def _match_option(item: str, options: List[Dict[str, float]]) -> Optional[int]:
    """把一个答案条目（可能是字母、可能是文本）匹配到选项下标。"""
    item = str(item).strip()
    if not item:
        return None
    low = item.lower()

    # (1) 字母型：A / (B) / C.
    m = re.fullmatch(r"[（(【\[]?\s*([a-h])\s*[)）】\]]?[.、]?", low)
    if m:
        letter = m.group(1).upper()
        for i, o in enumerate(options):
            if str(o["label"]).upper() == letter:
                return i

    # (2) 是非词型：正确 / 错误 / √ / ×
    if low in TRUTH_WORDS:
        for i, o in enumerate(options):
            if any(w in o["text"] for w in ("正确", "对", "√", "是")):
                return i
    if low in FALSE_WORDS:
        for i, o in enumerate(options):
            if any(w in o["text"] for w in ("错误", "错", "×", "否")):
                return i

    # (3) 文本型：先剥掉 "A." 前缀，再做「规范化后相等 / 答案被选项文本包含」。
    #     ⚠️ 这里刻意【不做反向包含】（原 `o["text"] in item` 已去掉）：
    #     选项文本 "A"、"正确"、"对" 这种超短文本会被任意长答案反向命中，
    #     答案就会飞到别的选项上去。反向那种情况交给下面的模糊匹配按相似度处理。
    text_item = _normalize_text(_strip_option_prefix(item))
    if len(text_item) >= 2:
        for i, o in enumerate(options):
            norm_opt = _normalize_text(o["text"])
            if not norm_opt or (len(norm_opt) == 1 and norm_opt == o["label"].lower()):
                continue          # 选项没识别出正文（文本就是字母本身），不参与文本匹配
            if text_item == norm_opt or text_item in norm_opt:
                return i

    # (4) 模糊匹配兜底：相似度得超过 OPTION_TEXT_MATCH_RATIO 才算命中
    if len(text_item) >= 2:
        best_idx: Optional[int] = None
        best_ratio = 0.0
        for i, o in enumerate(options):
            norm_opt = _normalize_text(o["text"])
            if not norm_opt or (len(norm_opt) == 1 and norm_opt == o["label"].lower()):
                continue
            ratio = difflib.SequenceMatcher(None, text_item, norm_opt).ratio()
            if ratio > best_ratio:
                best_idx, best_ratio = i, ratio
        if best_idx is not None and best_ratio >= OPTION_TEXT_MATCH_RATIO:
            log.debug("模糊匹配命中选项 %s（相似度 %.2f）",
                      options[best_idx]["label"], best_ratio)
            return best_idx

    return None


def normalize_answer(answer_field: Any, options: List[Dict[str, float]]) -> List[int]:
    """
    把模型返回的 answer 字段统一成「选项下标列表」。

    支持的输入形态：
        ["A"] / ["A", "C"] / "AC" / "A、C、D" / ["正确"] / ["A. 选项全文"]
    """
    items: List[str] = []

    if isinstance(answer_field, str):
        s = answer_field.strip()
        # 纯字母串（ACD / A,C,D / A、C）直接拆成单个字母
        if re.fullmatch(r"[A-Ha-h\s,，、;；/|]+", s):
            items = [c.upper() for c in re.findall(r"[A-Ha-h]", s)]
        else:
            items = [s]
    elif isinstance(answer_field, (list, tuple, set)):
        for x in answer_field:
            s = str(x).strip()
            if re.fullmatch(r"[A-Ha-h\s,，、;；/|]+", s) and len(s) > 1:
                items.extend([c.upper() for c in re.findall(r"[A-Ha-h]", s)])
            else:
                items.append(s)
    elif answer_field is not None:
        items = [str(answer_field).strip()]

    idxs: List[int] = []
    for it in items:
        i = _match_option(it, options)
        if i is not None and i not in idxs:
            idxs.append(i)

    idxs.sort()
    return idxs


# =====================================================================
# 九、鼠标 / 键盘操作
# =====================================================================

def click_dot(option: Dict[str, float], scale_x: float, scale_y: float, step: int) -> None:
    """计算小圆点的屏幕坐标并点击。"""
    sx = int(round(option["dot_x"] * scale_x))
    sy = int(round(option["dot_y"] * scale_y))

    screen_w, screen_h = pyautogui.size()
    sx = max(1, min(sx, screen_w - 2))
    sy = max(1, min(sy, screen_h - 2))

    log.info(
        "  -> 点击选项 %s：圆点坐标 (%d, %d)  [屏幕 %dx%d]",
        option["label"], sx, sy, screen_w, screen_h,
    )

    if DRY_RUN:
        log.info("  -> [DRY_RUN] 只打印不点击")
        return

    pyautogui.moveTo(sx, sy, duration=MOUSE_MOVE_DURATION)
    pyautogui.click()


def press_enter() -> None:
    """按 Enter 尝试进入下一题。"""
    log.info("  -> 页面未变化，按 Enter 尝试跳题")
    if DRY_RUN:
        return
    pyautogui.press("enter")


def click_next_button(button: Dict[str, float], scale_x: float, scale_y: float) -> None:
    """点「下一题」这类翻页按钮：按它文字框的中心点。"""
    sx = int(round(button["cx"] * scale_x))
    sy = int(round(button["cy"] * scale_y))
    screen_w, screen_h = pyautogui.size()
    sx = max(1, min(sx, screen_w - 2))
    sy = max(1, min(sy, screen_h - 2))

    log.info("  -> 点击「%s」按钮 (%d, %d)  [屏幕 %dx%d]",
             str(button.get("text", "")).strip(), sx, sy, screen_w, screen_h)
    if DRY_RUN:
        log.info("  -> [DRY_RUN] 只打印不点击")
        return
    pyautogui.moveTo(sx, sy, duration=MOUSE_MOVE_DURATION)
    pyautogui.click()


def advance_page(sig_before: Optional[Dict[str, Any]], first_option_top: Optional[float],
                 next_button: Optional[Dict[str, float]],
                 scale_x: float, scale_y: float) -> bool:
    """
    答完一题（或判定卡住）后翻到下一题，返回页面到底有没有翻过去。

    优先点页面上识别到的「下一题」按钮 —— 比盲按 Enter 靠谱得多，
    因为很多页面根本不吃 Enter，只认那个按钮。

    【绝不能做的事】点过按钮之后再补按 Enter：按钮被点过就会拿到键盘焦点，
    Enter 等于又激活它一次，页面会连跳两题（实测到的「下一题被多点一次」）。
    所以点过按钮就到此为止，没翻动交给下一轮重试 —— 下一轮看到还是这道题，
    会再走一遍这里；真按不动最后会由「连续无进展」兜底收工，不会空转。

    只有页面上压根没有可点的按钮时，才走原来的「按 Enter」兜底。
    """
    if CLICK_NEXT_BUTTON and next_button is not None:
        click_next_button(next_button, scale_x, scale_y)
        sleep_with_stop(AFTER_ANSWER_WAIT)
        after_img, _, _ = grab_screen()
        if page_changed(sig_before, make_stem_fingerprint(after_img, first_option_top)):
            return True
        log.info("  -> 点了「下一题」，指纹没看出变化（这类页面题干区常常不敏感）；"
                 "不补按 Enter —— 按钮此时带着键盘焦点，按 Enter 等于再点它一次、"
                 "会直接跳过一题。翻没翻页交给下一轮核对。")
        return False

    if not PRESS_ENTER_FALLBACK:
        return False
    if next_button is None:
        log.info("  -> 页面上没识别到「下一题」按钮（可能已经到本部分最后一题了），按 Enter 试一次")
    press_enter()
    sleep_with_stop(AFTER_ANSWER_WAIT)
    after_img2, _, _ = grab_screen()
    return page_changed(sig_before, make_stem_fingerprint(after_img2, first_option_top))


# =====================================================================
# 十、页面变化检测
# =====================================================================

def _crop_question_area(img_bgr: np.ndarray) -> np.ndarray:
    """
    只取左侧题目区做指纹。
    右侧答题卡在作答后一定会变（选项被点亮、题号变灰），不裁掉的话
    每次点完都以为"页面跳转了"，Enter 兜底就永远不会触发。
    """
    if not FINGERPRINT_LEFT_ONLY:
        return img_bgr
    w = img_bgr.shape[1]
    return img_bgr[:, : max(1, int(w * RIGHT_PANEL_RATIO))]


def _stem_region(img_bgr: np.ndarray, first_option_top: Optional[float]) -> np.ndarray:
    """
    裁出「题干区」：左侧题目区里，第一个选项行之上的那块像素。

    翻页判定为什么只看这一块、而不看整块左区：
        选项被点选后小圆点/文字会高亮，左区的像素必然变化。
        拿整块左区做指纹，就会把「我刚选中了一个选项」当成「页面已跳转」，
        于是脚本以为答完了、下一轮又把同一道题重新点一遍 ——
        多选题点两次等于把已选中的选项取消掉，白答一题还丢分。
        题干区不受选中状态影响，只有真正翻页时才会变。
    """
    h, w = img_bgr.shape[:2]
    x_right = max(1, int(w * RIGHT_PANEL_RATIO)) if FINGERPRINT_LEFT_ONLY else w
    y_bottom = int(first_option_top - 2) if first_option_top else 0
    if y_bottom < 8:
        return img_bgr[:, :x_right]      # 题干区太窄（选项几乎顶到页面顶部），退回整块左区
    return img_bgr[: min(y_bottom, h), :x_right]


def make_fingerprint(img_bgr: np.ndarray, crop: bool = True) -> Dict[str, Any]:
    """
    按 PAGE_DIFF_METHOD 生成指纹（返回 dict，跑日志时能看清用的哪种）。

    crop=True ：按 FINGERPRINT_LEFT_ONLY 先裁掉右侧答题卡（默认）
    crop=False：传进来的图已经裁好了，直接算
    """
    region = _crop_question_area(img_bgr) if crop else img_bgr
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    if PAGE_DIFF_METHOD == "phash":
        # 感知哈希：缩到 32x32 做 DCT，取左上 8x8 低频块，与中位数比较 => 64 位哈希。
        # 对整体亮度/轻微抖动不敏感，只对画面结构变化敏感，比直接比像素稳得多。
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        low = cv2.dct(small / 255.0)[:8, :8]
        return {"method": "phash", "data": (low > np.median(low)).astype(np.uint8).flatten()}

    if PAGE_DIFF_METHOD == "hist":
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        return {"method": "hist", "data": hist}

    small = cv2.resize(gray, (80, 45), interpolation=cv2.INTER_AREA)
    return {"method": "gray", "data": small.astype(np.float32)}


def make_stem_fingerprint(img_bgr: np.ndarray,
                          first_option_top: Optional[float]) -> Dict[str, Any]:
    """只对题干区做指纹，用于判断「这道题到底翻没翻过去」。"""
    return make_fingerprint(_stem_region(img_bgr, first_option_top), crop=False)


def page_changed(fp_a: Optional[Dict[str, Any]], fp_b: Optional[Dict[str, Any]]) -> bool:
    """
    比对两张指纹，判断题目有没有翻页。
    阈值全部集中在配置区，换机器 / 换页面样式直接改配置，不用动代码。
    """
    if not fp_a or not fp_b or fp_a.get("method") != fp_b.get("method"):
        return True

    if fp_a["method"] == "phash":
        dist = int(np.count_nonzero(fp_a["data"] != fp_b["data"]))
        changed = dist > PHASH_MAX_DISTANCE
        log.info("  -> pHash 汉明距离 = %d（阈值 %d）=> %s",
                 dist, PHASH_MAX_DISTANCE, "页面已跳转" if changed else "页面未变化")
        return changed

    if fp_a["method"] == "hist":
        corr = float(cv2.compareHist(fp_a["data"], fp_b["data"], cv2.HISTCMP_CORREL))
        changed = corr < HIST_MIN_CORRELATION
        log.info("  -> 直方图相关性 = %.5f（阈值 %.4f）=> %s",
                 corr, HIST_MIN_CORRELATION, "页面已跳转" if changed else "页面未变化")
        return changed

    diff = float(np.mean(np.abs(fp_a["data"] - fp_b["data"])))
    changed = diff > GRAY_DIFF_THRESHOLD
    log.info("  -> 灰度平均差 = %.2f（阈值 %.1f）=> %s",
             diff, GRAY_DIFF_THRESHOLD, "页面已跳转" if changed else "页面未变化")
    return changed


# =====================================================================
# 十一、主流程
# =====================================================================

def answer_one_question(
    step: int,
    ocr,
    skip_counter: Dict[str, int],
    click_state: Dict[str, Any],
) -> Tuple[str, bool]:
    """
    完整处理一题：截图 -> OCR -> 解析 -> 问模型 -> 点击 -> 判断是否跳转。

    参数 skip_counter：同一道题「连续被跳过」的次数表
        * 只在 skipped 时累加
        * answered 一律清零、卡住翻页后也清零
        * 累加到 SAME_QUESTION_LIMIT => 判定卡住，按 Enter 兜底翻页

    参数 click_state：跨轮状态 {"last_clicked_text": 上一轮真正点过答案的题目文本}
        * 这一轮又识别到和它几乎相同的题目，说明我们还停在那道题上（上一轮点了答案、
          页面没翻）。此时绝不能再点第二次，否则多选题会把已选中的选项点成
          「取消勾选」，等于把这题答废

    返回 (status, advanced)：
        status   : "answered" / "skipped" / "stuck"
        advanced : 这一轮页面到底有没有翻过去（主循环靠它判「连续无进展」）
    """
    if STOP_REQUESTED:
        log.warning("收到停止请求，本轮不动作")
        return "stopped", False

    # ---------- 1. 截图 + OCR ----------
    img_bgr, scale_x, scale_y = grab_screen()
    save_debug_image(img_bgr, step)

    lines = parse_ocr_result(ocr_image(ocr, img_bgr))
    if not lines:
        log.warning("第 %d 轮：OCR 没识别到任何文字，跳过本轮", step)
        return "skipped", False

    # ---------- 2. 解析题干 / 选项 ----------
    question = parse_question(lines, img_bgr.shape[1])
    if not question or not question["options"]:
        log.warning("第 %d 轮：没识别到选项，跳过本轮（可能需要调大 MIN_CONFIDENCE 或检查右侧面板比例）", step)
        return "skipped", False

    stem, options, qtype = question["stem"], question["options"], question["type"]
    first_option_top = question.get("first_option_top")
    log.info("第 %d 轮 | 题型=%s | 选项数=%d", step, TYPE_NAME_CN[qtype], len(options))
    log.info("题干：%s", stem[:120].replace("\n", " "))
    for o in options:
        log.info("   %s. %-40s 圆点≈(%.0f, %.0f)",
                 o["label"], o["text"][:40], o["dot_x"], o["dot_y"])

    # ---------- 2.5 找「下一题」按钮 ----------
    # 整屏找，不套左侧题目区过滤：这类按钮常常正好落在右侧面板那一片。
    next_button = find_next_button(lines) if CLICK_NEXT_BUTTON else None
    if next_button is not None:
        log.info("  -> 识别到翻页按钮「%s」@ (%.0f, %.0f)",
                 str(next_button["text"]).strip(), next_button["cx"], next_button["cy"])

    # 上面两条早退（连题都没解析出来）没有题干可归属，不记跳过计数
    key = _question_key(stem, options)
    q_text = _question_text(stem, options)
    # 记下这一轮「看到的题目」：主循环拿它跟上一轮比，判断页面到底有没有换题。
    # 指纹判定对「题干只有短短一行的页面」不敏感，需要这条文字判据兜底。
    click_state["last_seen_key"] = key

    def _skip(reason: str) -> Tuple[str, bool]:
        """跳过当前题：只有走这条路才给这道题累加跳过次数。"""
        log.error("%s，跳过本题", reason)
        skip_counter[key] = skip_counter.get(key, 0) + 1
        return "skipped", False

    # ---------- 3. 屏幕上还是刚才那道题？那绝不能再点第二次 ----------
    # 判定只看一个稳定信号：这一轮识别到的题目，和上一轮真正点过答案的那道题
    # 文本几乎完全相同。不用「上一轮判定为没翻页」做前置条件 —— 选项整行高亮的
    # 像素有可能顶进题干区，把指纹距离推过阈值，那样反而漏拦、又去点第二遍。
    # 只被跳过、没作答过的题（last_clicked_text 没更新）允许重试，不受影响。
    # DRY_RUN 下本来就不点鼠标，不拦，方便连着几轮校准坐标。
    if not DRY_RUN and _is_same_question(q_text, click_state.get("last_clicked_text")):
        log.warning("第 %d 轮：还是刚才那道题（点了答案但页面没翻页）。"
                    "为避免重复点击（多选会被点成取消勾选）本轮不再点击，"
                    "改点「下一题」/ 按 Enter 尝试翻页", step)
        before = make_stem_fingerprint(img_bgr, first_option_top)
        return "stuck", advance_page(before, first_option_top, next_button,
                                     scale_x, scale_y)

    # ---------- 4. 卡住检测（计数只统计「跳过」，作答成功会清零）----------
    if skip_counter.get(key, 0) >= SAME_QUESTION_LIMIT:
        log.warning("同一题已连续跳过 %d 次，判定为卡住，点「下一题」/ 按 Enter 尝试翻页",
                    skip_counter[key])
        skip_counter.pop(key, None)      # 翻页尝试完先清零，免得后面每轮都狂按 Enter
        before = make_stem_fingerprint(img_bgr, first_option_top)
        return "stuck", advance_page(before, first_option_top, next_button,
                                     scale_x, scale_y)

    # 点击前的页面指纹（答完后拿来比对是否跳转）
    # 注意：只看题干区。选项被点亮 / 被取消的像素变化不能当成「页面翻过去了」
    sig_before = make_stem_fingerprint(img_bgr, first_option_top)

    # ---------- 5. 大模型解题 ----------
    try:
        result = call_llm(stem, options, qtype)
    except Exception as e:
        return _skip(f"大模型解题失败：{e}")

    model_type = str(result.get("type", "")).strip().lower()
    if model_type in (TYPE_SINGLE, TYPE_MULTI, TYPE_JUDGE) and model_type != qtype:
        log.warning("模型判定题型(%s) 与 OCR 判定(%s) 不一致，以模型为准",
                    model_type, qtype)
        qtype = model_type

    idxs = normalize_answer(result.get("answer"), options)
    if not idxs:
        return _skip(f"模型答案 {result.get('answer')} 无法匹配到任何选项")

    chosen = [options[i] for i in idxs]
    log.info("最终作答：%s", "、".join(c["label"] for c in chosen))

    # ---------- 6. 点击：首次点击前先等，两次点击之间严格 CLICK_INTERVAL 秒 ----------
    if FIRST_CLICK_DELAY > 0:
        log.info("  -> 首次点击前等待 %.1f 秒（等上一题的跳转 / 渲染稳定）", FIRST_CLICK_DELAY)
        sleep_with_stop(FIRST_CLICK_DELAY)
    if STOP_REQUESTED:
        log.warning("  -> 收到停止请求，本题不点击")
        return "stopped", False

    for k, opt in enumerate(chosen):
        if STOP_REQUESTED:
            log.warning("  -> 收到停止请求，剩余 %d 个选项不再点击", len(chosen) - k)
            break
        if k > 0:
            sleep_with_stop(CLICK_INTERVAL)   # 【需求要求】每次点击之间固定 2 秒
        click_dot(opt, scale_x, scale_y, step)

    # ---------- 7. 答完后等待，再判断页面是否跳转 ----------
    settle = AFTER_ANSWER_WAIT if qtype == TYPE_MULTI else SINGLE_SETTLE_WAIT
    log.info("  -> 作答后等待 %.1f 秒再判断是否跳转", settle)
    sleep_with_stop(settle)

    img_after, _, _ = grab_screen()
    advanced = page_changed(sig_before, make_stem_fingerprint(img_after, first_option_top))

    if not advanced:
        # 页面没自己跳：先点识别到的「下一题」按钮（很多页面不吃 Enter），
        # 点了还不动才退回按 Enter。
        advanced = advance_page(sig_before, first_option_top, next_button,
                                scale_x, scale_y)

    if advanced:
        skip_counter.pop(key, None)      # answered => 跳过计数清零
        click_state["last_clicked_text"] = q_text
        log.info("  -> 页面已跳转，本题作答完成")
        return "answered", True

    # 记住「这题点过了」：下一轮再看到同一道题，只按 Enter、绝不重复点击
    click_state["last_clicked_text"] = q_text
    if CLICK_NEXT_BUTTON and next_button is not None:
        # 指纹没看出变化不等于真的没翻页（这类页面题干区就是不敏感）。
        # 别写成「停在原地」，那会把人带偏。
        log.warning("  -> 指纹没看出翻页，但已经点过「下一题」了。已记住本题："
                    "下一轮若还看到它，只会再翻页，绝不重复点选项")
    else:
        log.warning("  -> 题干区没变化，这题答完停在原地（页面上也没有「下一题」按钮，"
                    "像本部分最后一题只剩「保存」，那就得手动交卷 / 翻到下一部分）。"
                    "已记住本题，下一轮若还看到它只按 Enter，绝不重复点击")
    return "answered", False


def main() -> None:
    reset_stop()
    log.info("=" * 68)
    log.info("全屏在线考试自动答题脚本启动")
    log.info("=" * 68)

    # 尽早起：倒计时和 OCR 初始化都要能被打断（这几步也可能占好几秒）
    start_panic_watchdog()

    enable_dpi_awareness()

    screen_w, screen_h = pyautogui.size()
    log.info("检测到当前屏幕分辨率：%d x %d（脚本全程按此分辨率自适应，无硬编码坐标）",
             screen_w, screen_h)
    log.info("小圆点偏移量 DOT_OFFSET_X = %d 像素，右侧面板分界 = %.0f%%",
             DOT_OFFSET_X, RIGHT_PANEL_RATIO * 100)
    if DOT_OFFSET_SCALE_BY_WIDTH:
        log.info("小圆点偏移量按屏幕宽度等比缩放：基准 %d 像素 @ 宽度 %d",
                 DOT_OFFSET_X, DOT_OFFSET_REF_WIDTH)
    log.info("页面变化检测方式 = %s | 循环终止：已答满 %d 题 / 连续无进展 %d 轮 / 总轮数 %d 轮",
             PAGE_DIFF_METHOD, MAX_ANSWERED, MAX_NO_PROGRESS, MAX_ROUNDS)
    log.info("答题节奏 = %s（%s）", now_speed_name(), speed_text())
    if DRY_RUN:
        log.info("当前为【试运行模式 DRY_RUN=True】：只识别、只打印，不会真的点鼠标")
    else:
        log.info("当前为【正式模式】：即将开始模拟点击，3 秒后行动，鼠标甩到左上角可紧急中止")

    log.info("正在确认大模型接口（会自动纠正 Base URL 里 /v1、/chat/completions 的写法）...")
    usable_base, api_note = resolve_base_url(BASE_URL, API_KEY, MODEL_NAME, LLM_TIMEOUT)
    if usable_base:
        log.info("大模型接口可用：%s（%s）", usable_base, api_note)
    else:
        log.warning("大模型接口没探通：%s", api_note)
        log.warning("继续跑也可以，但每题都会失败并被跳过。最常踩的坑是「换了网关忘了换模型名」"
                    "（例如地址指向 DeepSeek，模型名却还是 gpt-4o-mini）；请回界面点「测试连接」，"
                    "核对 模型名 / API Key / Base URL 三样。")

    log.info("正在初始化 OCR 引擎（首次运行会下载中文模型，请耐心等待）...")
    log.info("OCR 后端 = %s（auto 会优先用 RapidOCR）", OCR_BACKEND)
    ocr = init_ocr()

    log.info("请把考试页面切到前台全屏，脚本将在 %d 秒后开始...", START_COUNTDOWN)
    for i in range(START_COUNTDOWN, 0, -1):
        if STOP_REQUESTED:
            log.warning("倒计时期间收到停止请求，取消本次答题")
            return
        log.info("  倒计时 %d ...", i)
        sleep_with_stop(1)

    skip_counter: Dict[str, int] = {}
    # 跨轮状态：上一轮真正点过答案的题目文本（下一轮靠它避免重复点击）
    click_state: Dict[str, Any] = {"last_clicked_text": None}
    stats = {"answered": 0, "skipped": 0, "stuck": 0, "stopped": 0}
    no_progress_streak = 0
    step = 0

    while step < MAX_ROUNDS:
        if STOP_REQUESTED:
            log.warning("收到停止请求，收工")
            break
        step += 1
        log.info("-" * 68)
        log.info("第 %d 轮 | 已作答 %d/%d | 已跳过 %d | 连续无进展 %d/%d",
                 step, stats["answered"], MAX_ANSWERED, stats["skipped"],
                 no_progress_streak, MAX_NO_PROGRESS)
        try:
            prev_key = click_state.get("last_seen_key")
            status, advanced = answer_one_question(step, ocr, skip_counter, click_state)
            stats[status] = stats.get(status, 0) + 1
            if status == "stopped":
                log.warning("收到停止请求，停止答题")
                break
        except pyautogui.FailSafeException:
            log.error("检测到鼠标甩到左上角，紧急中止脚本")
            break
        except KeyboardInterrupt:
            log.error("收到 Ctrl+C，手动中止脚本")
            break
        except Exception as e:
            # 容错兜底：任何意外都只跳过当前题，绝不中断整个循环
            log.exception("第 %d 轮发生未预期异常，跳过本轮：%s", step, e)
            stats["skipped"] = stats.get("skipped", 0) + 1
            status, advanced = "skipped", False

        # ---- 终止判定：页面真的翻过去了才算「有进展」 ----
        # 除了指纹判定，再加一条文字判据：OCR 看到题目换成新的一道了，也算有进展。
        # 题干只有短短一行的页面，题干区大片空白，pHash 对这种变化不敏感
        # （实测距离可能只有 2，远低于阈值），只看指纹会把「明明答对了」算成没进展，
        # 于是 15 轮就误收工。
        seen_key = click_state.get("last_seen_key")
        swapped = (seen_key is not None and prev_key is not None and seen_key != prev_key)
        if swapped and not advanced:
            log.info("  -> 本轮题目已换成新的一道，按「有进展」计")
        no_progress_streak = 0 if (advanced or swapped) else no_progress_streak + 1

        if stats["answered"] >= MAX_ANSWERED:
            log.info("已作答 %d 题，达到 MAX_ANSWERED，正常收工", stats["answered"])
            break
        if no_progress_streak >= MAX_NO_PROGRESS:
            log.warning("连续 %d 轮没有任何进展（没作答、页面也没翻），自动收工，避免空转烧时间",
                        no_progress_streak)
            if DRY_RUN:
                log.info("（提示：试运行模式不会真点击，页面本来就不会变，这个计数涨满属正常）")
            break

        sleep_with_stop(LOOP_SLEEP)
    else:
        log.warning("达到总轮数硬上限 MAX_ROUNDS=%d，强制收工", MAX_ROUNDS)

    log.info("=" * 68)
    log.info("循环结束 | 共 %d 轮 | 作答 %d 题，跳过 %d 题，卡住翻页 %d 次",
             step, stats.get("answered", 0), stats.get("skipped", 0), stats.get("stuck", 0))
    if stats.get("stopped"):
        log.info("（本次是被「停止」按钮 / request_stop() 中止的）")
    if DRY_RUN:
        # 最关键的一句：试运行跑完画面上一动不动，必须说清楚「一个鼠标都没点」，
        # 否则很容易以为是「不点击」的 bug。
        log.warning("注意：本次全程是【试运行】，一个鼠标动作都没做（识别和解题都是真的）。")
        log.warning("要真的开始作答：回界面取消勾选「试运行（DRY_RUN）」，"
                    "此时按钮旁边会变成「正式模式 —— 会真的点鼠标」，再点「开始答题」。")
    log.info("详细日志见 %s", LOG_FILE)
    log.info("=" * 68)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.exception("脚本启动失败：%s", exc)
        sys.exit(1)
