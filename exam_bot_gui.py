# -*- coding: utf-8 -*-
"""
exam_bot_gui.py —— 图形界面启动器（也是打包 exe 的入口）

    python exam_bot_gui.py            打开界面
    python exam_bot_gui.py --selftest 不点鼠标、不联网，只自检 OCR + 题目解析
    python exam_bot_gui.py --test-api 不开界面，拿 config.json 直接打一次大模型接口

界面里填 API Key / Base URL / 模型名，点「测试连接」确认能通，再点「开始答题」。
配置保存在同目录的 config.json（明文，别外传）。
"""

import json
import logging
import os
import queue
import sys
import threading
import time
import traceback

import tkinter as tk
from tkinter import ttk, messagebox

try:
    import exam_auto_solve as bot
    IMPORT_ERROR = None
except Exception as _exc:      # 缺依赖时给一句人话，而不是一堆 traceback
    bot = None
    IMPORT_ERROR = _exc


def _app_dir() -> str:
    """打包成 exe 后取 exe 目录，源码运行取脚本目录。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


APP_DIR = _app_dir()
CONFIG_PATH = os.path.join(APP_DIR, "config.json")
CRASH_LOG = os.path.join(APP_DIR, "crash.log")
APP_TITLE = "在线考试自动答题助手"

CONFIG_DEFAULTS = {
    "api_key": "",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "llm_timeout": 60,
    "dry_run": True,
    "dot_offset_x": 35,
    "max_answered": 90,
    "start_countdown": 5,
    "ocr_backend": "auto",
    "save_debug_screenshot": True,
    "click_next_button": True,
    "speed": "快速",
}

CONFIG_TIPS = (
    "换了网关，地址和【模型名】要一起换（这是最容易踩的坑）。\n"
    "Base URL 带不带 /v1 都行，程序会自动纠正，也会自动剥掉多余的 /chat/completions。\n"
    "常见的填法：\n"
    "  OpenAI   https://api.openai.com/v1                        模型 gpt-4o-mini\n"
    "  DeepSeek https://api.deepseek.com/v1                      模型 deepseek-chat\n"
    "  通义千问  https://dashscope.aliyuncs.com/compatible-mode/v1 模型 qwen-plus\n"
    "  智谱GLM  https://open.bigmodel.cn/api/paas/v4              模型 glm-4-flash\n"
    "  自建中转  按网关文档填"
)

# 服务商预设：选中后自动把 Base URL 和模型名一起填好。
# 之所以把两者绑在一起，是因为最常见的报错就是「换了地址没换模型名」——
# 比如地址指向 DeepSeek，模型名却还是 gpt-4o-mini，会一直返回模型不存在。
PROVIDERS = [
    ("DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat"),
    ("OpenAI", "https://api.openai.com/v1", "gpt-4o-mini"),
    ("通义千问", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    ("智谱 GLM", "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash"),
    ("自定义", "", ""),
]


def _host_of(url: str) -> str:
    """取 URL 的主机名，用来判断用户填的是哪家网关。"""
    try:
        from urllib.parse import urlparse
        return (urlparse((url or "").strip()).hostname or "").lower()
    except Exception:
        return ""


def match_provider(base_url: str) -> str:
    """按主机名猜用户用的是哪家，猜不出返回「自定义」。"""
    host = _host_of(base_url)
    if host:
        for label, base, _model in PROVIDERS:
            if base and _host_of(base) == host:
                return label
    return PROVIDERS[-1][0]


def provider_hint(base_url: str, model: str) -> str:
    """地址和模型名对不上时的补救提示（这正是用户踩过的坑）。"""
    label = match_provider(base_url)
    for name, _base, good_model in PROVIDERS:
        if name == label and good_model and (model or "").strip() != good_model:
            return ("\n        提示：%s 的模型名一般是 %s，你现在填的是 %s；"
                    "换了网关记得连模型名一起换。" % (name, good_model, model or "(空)"))
    return ""


def _now() -> str:
    return time.strftime("%H:%M:%S")


def load_config() -> dict:
    cfg = dict(CONFIG_DEFAULTS)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            cfg.update({k: v for k, v in data.items() if k in CONFIG_DEFAULTS})
    except FileNotFoundError:
        pass
    except Exception as e:
        print("[warn] 读取 config.json 失败，改用默认值：%s" % e)
    # 配置文件里没填的，退回环境变量；再没有才用默认值
    if not cfg.get("api_key"):
        cfg["api_key"] = os.getenv("OPENAI_API_KEY", "")
    if not cfg.get("base_url"):
        cfg["base_url"] = os.getenv("OPENAI_BASE_URL", CONFIG_DEFAULTS["base_url"])
    if not cfg.get("model"):
        cfg["model"] = os.getenv("OPENAI_MODEL", CONFIG_DEFAULTS["model"])
    return cfg


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def probe_api(cfg: dict) -> tuple:
    """
    拿这份配置打一次 /chat/completions，返回 (是否成功, 给人看的一句话)。
    界面上的「测试连接」和命令行的 --test-api 共用这一份逻辑。
    地址带不带 /v1、有没有多写 /chat/completions，这里都会自动纠正后再试；
    返回里会写清楚真正打通的地址，方便对着改配置。
    """
    key = (cfg.get("api_key") or "").strip()
    base = (cfg.get("base_url") or "").strip()
    model = (cfg.get("model") or "").strip()
    if not key or not base or not model:
        return False, "[失败] API Key / Base URL / 模型名 三个都得填"
    if bot is None:
        return False, "[失败] 引擎加载失败：%s" % IMPORT_ERROR

    timeout = min(60, max(5, int(cfg.get("llm_timeout") or 60)))
    usable, note = bot.resolve_base_url(base, key, model, timeout)
    if usable:
        return True, ("[成功] 接口可用：%s/chat/completions\n        %s"
                      % (usable, note))
    hint = provider_hint(base, model)
    if not hint and ("model" in note.lower() or "模型" in note):
        hint = ("\n        提示：网关说模型名不认，去网关文档确认可用的模型名再填；"
                "换了网关只改地址、没改模型名就会这样。")
    return False, ("[失败] 试过的地址都没打通：\n        %s%s" % (note, hint))


def test_api_from_config() -> int:
    """--test-api：不开界面，直接拿 config.json 打一次接口，方便排错。"""
    cfg = load_config()
    print("配置文件：%s" % CONFIG_PATH)
    print("Base URL：%s" % cfg.get("base_url"))
    print("模型：%s" % cfg.get("model"))
    key = cfg.get("api_key") or ""
    print("API Key：%s" % ("已填写（%s……）" % key[:6] if key else "没填"))
    ok, message = probe_api(cfg)
    print(message)
    return 0 if ok else 1


class QueueLogHandler(logging.Handler):
    """把引擎日志丢进队列，由主线程取出来显示（Tk 控件只能在主线程碰）。"""

    def __init__(self, log_queue: "queue.Queue"):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.log_queue.put(self.format(record))
        except Exception:
            pass


class App(tk.Tk):
    def __init__(self, cfg: dict):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("860x700")
        self.minsize(780, 600)

        self.ui_queue: "queue.Queue" = queue.Queue()
        self.worker = None
        self.bot_finished = threading.Event()

        self.var_key = tk.StringVar(value=cfg.get("api_key", ""))
        self.var_base = tk.StringVar(value=cfg.get("base_url", ""))
        self.var_model = tk.StringVar(value=cfg.get("model", ""))
        self.var_timeout = tk.StringVar(value=str(cfg.get("llm_timeout", 60)))
        self.var_dry = tk.BooleanVar(value=bool(cfg.get("dry_run", True)))
        self.var_offset = tk.StringVar(value=str(cfg.get("dot_offset_x", 35)))
        self.var_max = tk.StringVar(value=str(cfg.get("max_answered", 90)))
        self.var_countdown = tk.StringVar(value=str(cfg.get("start_countdown", 5)))
        self.var_ocr = tk.StringVar(value=str(cfg.get("ocr_backend", "auto")))
        self.var_shots = tk.BooleanVar(value=bool(cfg.get("save_debug_screenshot", True)))
        self.var_next_btn = tk.BooleanVar(value=bool(cfg.get("click_next_button", True)))
        self.var_speed = tk.StringVar(value=str(cfg.get("speed", "快速")))
        self.var_show_key = tk.BooleanVar(value=False)

        self._build_ui()
        self._attach_log_handler()
        self._poll_ui_queue()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._append("[%s] 工作目录：%s\n" % (_now(), APP_DIR))
        self._append("[%s] 配置：先选服务商（地址和模型名会自动填好），填上 API Key，"
                     "点「测试连接」，通了再点「开始答题」。\n" % _now())
        self._append(CONFIG_TIPS + "\n")
        if self.var_dry.get():
            self._append("[%s] 当前是【试运行】：只识别、只打印，不点鼠标。\n" % _now())

    # ---------------- 界面 ----------------
    def _build_ui(self) -> None:
        pad = {"padx": 6, "pady": 3}

        api = ttk.LabelFrame(self, text=" 大模型接口 ")
        api.pack(fill="x", padx=10, pady=(10, 4))

        ttk.Label(api, text="服务商").grid(row=0, column=0, sticky="w", **pad)
        self.combo_provider = ttk.Combobox(api, width=20, state="readonly",
                                           values=[item[0] for item in PROVIDERS])
        self.combo_provider.set(match_provider(self.var_base.get()))
        self.combo_provider.grid(row=0, column=1, sticky="w", **pad)
        self.combo_provider.bind("<<ComboboxSelected>>", self._on_provider)
        ttk.Label(api, text="（选中后自动填好地址和模型名，也可以直接自己填）").grid(
            row=0, column=2, columnspan=2, sticky="w", **pad)

        ttk.Label(api, text="API Key").grid(row=1, column=0, sticky="w", **pad)
        self.entry_key = ttk.Entry(api, textvariable=self.var_key, show="*", width=58)
        self.entry_key.grid(row=1, column=1, columnspan=3, sticky="we", **pad)
        ttk.Checkbutton(api, text="显示", variable=self.var_show_key,
                        command=self._toggle_key).grid(row=1, column=4, sticky="w", **pad)

        ttk.Label(api, text="Base URL").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(api, textvariable=self.var_base, width=58).grid(
            row=2, column=1, columnspan=3, sticky="we", **pad)

        ttk.Label(api, text="模型名").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(api, textvariable=self.var_model, width=28).grid(
            row=3, column=1, sticky="we", **pad)
        ttk.Label(api, text="超时(秒)").grid(row=3, column=2, sticky="e", **pad)
        ttk.Spinbox(api, from_=10, to=300, width=7, textvariable=self.var_timeout).grid(
            row=3, column=3, sticky="w", **pad)

        self.btn_test = ttk.Button(api, text="测试连接", command=self._on_test)
        self.btn_test.grid(row=4, column=1, sticky="w", **pad)
        ttk.Button(api, text="保存配置", command=self._on_save).grid(
            row=4, column=2, sticky="w", **pad)
        ttk.Button(api, text="打开所在目录", command=self._open_dir).grid(
            row=4, column=3, sticky="w", **pad)
        api.columnconfigure(1, weight=1)

        run = ttk.LabelFrame(self, text=" 运行参数 ")
        run.pack(fill="x", padx=10, pady=4)

        ttk.Checkbutton(run, text="试运行（DRY_RUN：只识别、不点击）",
                        variable=self.var_dry,
                        command=self._refresh_mode_label).grid(
                            row=0, column=0, columnspan=2, sticky="w", **pad)
        ttk.Checkbutton(run, text="保存每轮截图到 debug_shots/",
                        variable=self.var_shots).grid(row=0, column=2, columnspan=2,
                                                      sticky="w", **pad)

        ttk.Label(run, text="小圆点偏移 DOT_OFFSET_X").grid(row=1, column=0, sticky="w", **pad)
        ttk.Spinbox(run, from_=0, to=200, width=7, textvariable=self.var_offset).grid(
            row=1, column=1, sticky="w", **pad)
        ttk.Label(run, text="最多作答(题)").grid(row=1, column=2, sticky="e", **pad)
        ttk.Spinbox(run, from_=1, to=500, width=7, textvariable=self.var_max).grid(
            row=1, column=3, sticky="w", **pad)

        ttk.Label(run, text="开始倒计时(秒)").grid(row=2, column=0, sticky="w", **pad)
        ttk.Spinbox(run, from_=1, to=60, width=7, textvariable=self.var_countdown).grid(
            row=2, column=1, sticky="w", **pad)
        ttk.Label(run, text="OCR 引擎").grid(row=2, column=2, sticky="e", **pad)
        ttk.Combobox(run, width=12, state="readonly", textvariable=self.var_ocr,
                     values=("auto", "rapidocr", "paddleocr")).grid(
            row=2, column=3, sticky="w", **pad)

        ttk.Label(run, text="答题节奏").grid(row=3, column=0, sticky="w", **pad)
        ttk.Combobox(run, width=7, state="readonly", textvariable=self.var_speed,
                     values=tuple(bot.SPEED_PRESETS) if bot is not None
                     else ("极速", "快速", "标准")).grid(
            row=3, column=1, sticky="w", **pad)
        ttk.Label(run, text="（默认「快速」；点早了会点空就切「标准」）").grid(
            row=3, column=2, columnspan=2, sticky="w", **pad)

        ttk.Checkbutton(run, text="答完后点页面上的「下一题」按钮翻页",
                        variable=self.var_next_btn).grid(
                            row=4, column=0, columnspan=4, sticky="w", **pad)
        ttk.Label(run, text="（识别不到「下一题」时自动退回按 Enter；"
                            "页面会自己跳题的不用管它）").grid(
            row=5, column=0, columnspan=4, sticky="w", padx=(22, 0))

        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", padx=10, pady=4)
        self.btn_start = ttk.Button(ctrl, text="开始答题", command=self._on_start)
        self.btn_start.pack(side="left", padx=(0, 6))
        self.btn_stop = ttk.Button(ctrl, text="停止", command=self._on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(0, 6))
        # 「到底会不会点鼠标」必须一眼看到：试运行跑完一个鼠标都没点，很容易误以为程序坏了
        style = ttk.Style(self)
        style.configure("ModeDry.TLabel", foreground="#b26a00", font=("", 9, "bold"))
        style.configure("ModeReal.TLabel", foreground="#c00000", font=("", 9, "bold"))
        self.lbl_mode = ttk.Label(ctrl, text="", style="ModeDry.TLabel")
        self.lbl_mode.pack(side="left", padx=(4, 10))
        self._refresh_mode_label()
        ttk.Label(ctrl, text="（停止后当前这一轮会尽快收尾）").pack(side="left")

        # 全屏考试页里点不到这个窗口的「停止」按钮，所以热键必须写在最显眼的地方
        hotkey = bot.panic_hotkey_text() if bot is not None else "连按 3 次 Esc"
        ttk.Label(self, text="紧急停止：%s（浏览器全屏、切不出后台时也能停）" % hotkey,
                  style="ModeDry.TLabel").pack(anchor="w", padx=14, pady=(0, 4))

        box = ttk.LabelFrame(self, text=" 运行日志 ")
        box.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.text = tk.Text(box, height=18, wrap="word", state="disabled")
        bar = ttk.Scrollbar(box, command=self.text.yview)
        self.text.configure(yscrollcommand=bar.set)
        self.text.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        bar.pack(side="right", fill="y", pady=6)

    def _refresh_mode_label(self) -> None:
        """把当前模式写在按钮旁边：切换「试运行」勾选框时实时更新。"""
        if self.var_dry.get():
            self.lbl_mode.configure(text="当前：试运行 —— 不会点鼠标！",
                                    style="ModeDry.TLabel")
        else:
            self.lbl_mode.configure(text="当前：正式模式 —— 会真的点鼠标。",
                                    style="ModeReal.TLabel")

    def _toggle_key(self) -> None:
        self.entry_key.configure(show="" if self.var_show_key.get() else "*")

    def _on_provider(self, event=None) -> None:
        """选服务商就把 Base URL 和模型名填好，省得记错。"""
        name = self.combo_provider.get()
        for label, base_url, model in PROVIDERS:
            if label == name and base_url:
                self.var_base.set(base_url)
                self.var_model.set(model)
                self._append("[%s] 已填好 %s 的地址和模型名，接着填 API Key 即可。\n"
                             % (_now(), label))
                return

    # ---------------- 日志 ----------------
    def _attach_log_handler(self) -> None:
        handler = QueueLogHandler(self.ui_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                               datefmt="%H:%M:%S"))
        logging.getLogger("exam_bot").addHandler(handler)
        logging.getLogger("exam_bot").setLevel(logging.INFO)

    def _append(self, text: str) -> None:
        """线程安全：所有要显示的东西都走队列，由主线程写控件。"""
        self.ui_queue.put(text)

    def _ui_call(self, fn, *args, **kwargs) -> None:
        """线程安全：把界面操作排到主线程执行（支持关键字参数）。"""
        self.ui_queue.put((fn, args, kwargs))

    def _poll_ui_queue(self) -> None:
        while True:
            try:
                item = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple):
                try:
                    fn, args, kwargs = item
                    fn(*args, **kwargs)
                except Exception as e:
                    print("[warn] 界面回调出错：%s" % e)
                continue
            self.text.configure(state="normal")
            self.text.insert("end", item)
            self.text.see("end")
            self.text.configure(state="disabled")

        if self.bot_finished.is_set():
            self.bot_finished.clear()
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self._append("\n[%s] ===== 本次运行结束 =====\n" % _now())

        self.after(120, self._poll_ui_queue)

    # ---------------- 配置 ----------------
    def _collect(self) -> dict:
        def _int(var, default):
            try:
                return int(str(var.get()).strip())
            except Exception:
                return default

        return {
            "api_key": self.var_key.get().strip(),
            "base_url": self.var_base.get().strip().rstrip("/"),
            "model": self.var_model.get().strip(),
            "llm_timeout": _int(self.var_timeout, 60),
            "dry_run": bool(self.var_dry.get()),
            "dot_offset_x": _int(self.var_offset, 35),
            "max_answered": _int(self.var_max, 90),
            "start_countdown": _int(self.var_countdown, 5),
            "ocr_backend": self.var_ocr.get().strip() or "auto",
            "save_debug_screenshot": bool(self.var_shots.get()),
            "click_next_button": bool(self.var_next_btn.get()),
            "speed": self.var_speed.get().strip() or "快速",
        }

    def _on_save(self) -> None:
        cfg = self._collect()
        if not cfg["api_key"]:
            messagebox.showwarning(APP_TITLE, "API Key 是空的，先填上再保存。")
            return
        try:
            save_config(cfg)
        except Exception as e:
            messagebox.showerror(APP_TITLE, "保存失败：%s" % e)
            return
        self._append("[%s] 配置已保存到 %s（明文，别外传）\n" % (_now(), CONFIG_PATH))

    def _open_dir(self) -> None:
        try:
            if sys.platform == "win32":
                os.startfile(APP_DIR)            # type: ignore[attr-defined]
            else:
                import subprocess
                subprocess.Popen(["xdg-open", APP_DIR])
        except Exception as e:
            messagebox.showerror(APP_TITLE, "打不开目录：%s" % e)

    # ---------------- 测试连接 ----------------
    def _on_test(self) -> None:
        cfg = self._collect()
        if not cfg["api_key"] or not cfg["base_url"]:
            messagebox.showwarning(APP_TITLE, "API Key 和 Base URL 都要填。")
            return
        self.btn_test.configure(state="disabled")
        self._append("[%s] 正在测试接口（模型 %s）；地址里的 /v1、/chat/completions "
                     "写法会自动纠正，以实际打通的为准。\n"
                     % (_now(), cfg["model"]))
        threading.Thread(target=self._test_worker, args=(cfg,), daemon=True).start()

    def _test_worker(self, cfg: dict) -> None:
        ok, message = probe_api(cfg)
        self._append("[%s] %s\n" % (_now(), message))
        if not ok:
            self._append("        排查顺序：① 模型名是不是这家网关的（换了地址最容易忘这条）"
                         " ② API Key 有没有失效/欠费 ③ Base URL 是不是这家的域名\n")
        self._ui_call(self.btn_test.configure, state="normal")

    # ---------------- 开始 / 停止 ----------------
    def _on_start(self) -> None:
        if bot is None:
            messagebox.showerror(APP_TITLE, "引擎加载失败：%s" % IMPORT_ERROR)
            return
        cfg = self._collect()
        if not cfg["api_key"] or not cfg["base_url"]:
            messagebox.showwarning(APP_TITLE, "先把 API Key 和 Base URL 填好。")
            return
        if not cfg["dry_run"]:
            if not messagebox.askyesno(
                    APP_TITLE,
                    "现在是【正式模式】：脚本会真的移动鼠标并点击选项。\n\n"
                    "建议先用「试运行」确认小圆点坐标是对的。\n\n确定要正式开跑吗？"):
                return
        try:
            save_config(cfg)
        except Exception as e:
            self._append("[%s] [warn] 配置保存失败：%s\n" % (_now(), e))

        self._apply_to_engine(cfg)
        bot.reset_stop()

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._append("[%s] 启动答题。窗口会自动最小化，倒计时 %d 秒内请把考试页面"
                     "切到前台全屏；跑完会自动恢复窗口。\n"
                     % (_now(), cfg["start_countdown"]))
        # 必须让开！窗口浮在考试页上会被一起截图，OCR 会把界面上的下拉框、
        # 按钮、日志文字当成题干和选项（实测选项会被串成一大条，整题都答废）。
        try:
            self.iconify()
        except Exception:
            pass
        self.worker = threading.Thread(target=self._run_engine, daemon=True)
        self.worker.start()

    def _apply_to_engine(self, cfg: dict) -> None:
        bot.API_KEY = cfg["api_key"]
        bot.BASE_URL = cfg["base_url"]
        bot.MODEL_NAME = cfg["model"]
        bot.LLM_TIMEOUT = int(cfg["llm_timeout"])
        bot.DRY_RUN = bool(cfg["dry_run"])
        bot.DOT_OFFSET_X = int(cfg["dot_offset_x"])
        bot.MAX_ANSWERED = int(cfg["max_answered"])
        bot.START_COUNTDOWN = int(cfg["start_countdown"])
        bot.OCR_BACKEND = cfg["ocr_backend"]
        bot.SAVE_DEBUG_SCREENSHOT = bool(cfg["save_debug_screenshot"])
        bot.CLICK_NEXT_BUTTON = bool(cfg["click_next_button"])
        bot.apply_speed(cfg["speed"])
        bot.LLM_RETRY = max(1, bot.LLM_RETRY)

    def _run_engine(self) -> None:
        try:
            bot.main()
        except Exception as e:
            try:
                bot.log.exception("答题过程出错：%s", e)
            except Exception:
                pass
            self._append("\n[%s] [错误] 答题过程出错：%s\n%s\n"
                         % (_now(), e, traceback.format_exc()))
        finally:
            self._ui_call(self._restore_window)
            self.bot_finished.set()

    def _restore_window(self) -> None:
        """
        答题结束把窗口恢复出来并推到最前面（跑在主线程）。

        只 deiconify + lift 不够：考试页往往还是浏览器全屏，窗口被压在底下，
        用户会以为程序没了（尤其按了紧急停止热键之后，人还在全屏页面里）。
        所以先短暂置顶，确保一定看得见。
        """
        try:
            self.deiconify()
            self.lift()
            self.attributes("-topmost", True)
            self.after(1200, self._drop_topmost)
            self.focus_force()
        except Exception:
            pass
        try:                       # Tk 有时候抢不过全屏窗口，再用 Win32 推一把
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = user32.GetParent(self.winfo_id()) or self.winfo_id()
            user32.ShowWindow(hwnd, 9)          # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
        except Exception:
            pass

    def _drop_topmost(self) -> None:
        """置顶只在刚恢复时用一下，别让窗口一直压着别人。"""
        try:
            self.attributes("-topmost", False)
        except Exception:
            pass

    def _on_stop(self) -> None:
        if bot is not None:
            bot.request_stop()
        self._append("[%s] 已发出停止请求，正在收尾……\n" % _now())

    def _on_close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            if not messagebox.askyesno(APP_TITLE, "答题还在跑，确定要关掉窗口吗？\n"
                                                  "（会先发出停止请求）"):
                return
            bot.request_stop()
        self.destroy()


# =====================================================================
# 自检 / 入口
# =====================================================================

def _render_test_page():
    """画一张模拟答题页（真中文字，用系统字体渲染），用来验证 OCR + 解析。"""
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1920, 1080
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    font = None
    fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc", "msyhl.ttc"):
        path = os.path.join(fonts_dir, name)
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, 26)
                break
            except Exception:
                continue
    if font is None:
        raise RuntimeError("找不到可用的中文字体（试过 msyh.ttc / simhei.ttf / simsun.ttc）")

    draw.text((60, 80), "第1题 关于软件测试，下列哪些说法是正确的？", fill="black", font=font)
    draw.text((60, 120), "（多选题）", fill="black", font=font)
    options = [
        "A. 软件测试是为了发现错误",
        "B. 软件测试不需要任何文档",
        "C. 测试用例要覆盖边界值",
        "D. 测试能完全消除缺陷",
    ]
    dots = []
    for i, text in enumerate(options):
        y = 220 + i * 60
        draw.text((80, y), text, fill="black", font=font)
        dot_x, dot_y = 45, y + 16
        draw.ellipse([dot_x - 8, dot_y - 8, dot_x + 8, dot_y + 8], outline="black", width=2)
        dots.append((dot_x, dot_y))

    draw.rectangle([1300, 100, 1900, 900], outline="black", width=2)
    draw.text((1320, 120), "答题卡 1 2 3 4", fill="black", font=font)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), dots


def selftest() -> int:
    """--selftest：验证 OCR 引擎和题目解析，不点鼠标、不联网。"""
    print("== 自检开始（OCR + 题目解析）==")

    img, dots = _render_test_page()
    page_path = os.path.join(APP_DIR, "selftest_page.png")
    if bot.imwrite_unicode(page_path, img):
        print("模拟答题页已保存：%s" % page_path)
    else:
        print("[warn] 模拟页保存失败（不影响自检）")

    engine = bot.init_ocr()
    print("OCR 后端：%s" % engine[0])
    lines = bot.parse_ocr_result(bot.ocr_image(engine, img))
    print("OCR 识别到 %d 行：" % len(lines))
    for line in lines[:12]:
        print("   (%4.0f,%4.0f)-(%4.0f,%4.0f) %s"
              % (line["x1"], line["y1"], line["x2"], line["y2"], line["text"]))

    question = bot.parse_question(lines, img.shape[1])
    if not question:
        print("[失败] 没能从模拟页面里解析出题目")
        return 1

    print("题型：%s；选项数：%d" % (bot.TYPE_NAME_CN[question["type"]], len(question["options"])))
    for opt in question["options"]:
        print("   %s. %-24s 圆点≈(%.0f, %.0f)"
              % (opt["label"], opt["text"][:24], opt["dot_x"], opt["dot_y"]))

    problems = []
    if question["type"] != bot.TYPE_MULTI:
        problems.append("题型应为多选题，实际 %s" % question["type"])
    labels = [o["label"] for o in question["options"]]
    if labels[:4] != ["A", "B", "C", "D"]:
        problems.append("选项字母应为 A/B/C/D，实际 %s" % labels)
    if question["options"]:
        # OCR 框边缘本身有几像素抖动，按中位数比较，别拿单个选项判定
        got = sorted(o["dot_x"] for o in question["options"])
        median = got[len(got) // 2]
        want = dots[0][0]
        if abs(median - want) > 10:
            problems.append("小圆点 X 偏差过大：算出中位数 %.0f，实际 %.0f"
                            "（DOT_OFFSET_X 需要按你的页面微调）" % (median, want))

    print()
    if problems:
        print("[未通过] " + "；".join(problems))
        return 1
    print("[通过] OCR 与解析链路正常，可以直接打开界面用了")
    return 0


def _hide_console() -> None:
    """打包成 exe 后把黑窗口藏起来（--selftest / --debug 时不藏）。"""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return
    try:
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)
    except Exception:
        pass


def _install_excepthooks() -> None:
    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            with open(CRASH_LOG, "a", encoding="utf-8") as f:
                f.write("\n=== %s ===\n%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), text))
        except Exception:
            pass
        try:
            messagebox.showerror(APP_TITLE,
                                 "程序出错了：\n%s\n\n详情已写入 %s" % (exc, CRASH_LOG))
        except Exception:
            print(text)

    sys.excepthook = hook
    try:
        threading.excepthook = lambda args: hook(args.exc_type, args.exc_value,
                                                 args.exc_traceback)
    except Exception:
        pass


def main() -> int:
    if "--selftest" in sys.argv:
        _install_excepthooks()
        try:
            return selftest()
        except Exception as e:
            print("自检失败：%s" % e)
            traceback.print_exc()
            return 1

    if "--test-api" in sys.argv:
        return test_api_from_config()

    if bot is None:
        message = ("引擎没加载起来：%s\n\n请先装依赖：\n"
                   "  pip install rapidocr-onnxruntime pyautogui opencv-python requests numpy\n"
                   "（OCR 也可以用 paddleocr：pip install paddlepaddle paddleocr）" % IMPORT_ERROR)
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(APP_TITLE, message)
            root.destroy()
        except Exception:
            print(message)
        return 1

    _install_excepthooks()
    _hide_console()
    app = App(load_config())
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
