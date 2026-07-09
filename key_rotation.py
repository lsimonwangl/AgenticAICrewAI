"""key_rotation.py — 讓 crewai 的 LLM 每次呼叫輪流換一把 api_key。

NVIDIA 免費額度單帳號很快 429；備妥多把 key 後 round-robin 輪流使用，
等於把可用額度乘上 key 數。

以 class 層級 monkey-patch `_prepare_completion_params`（唯一組出 params 的地方，
串流/非串流都經過它）注入。輪替狀態放模組層 cycle，因此連 crewai 深拷貝出的
LLM 副本（Guardrail 等）也一起輪替，不必碰它的 __copy__/__deepcopy__。
"""

import itertools
import threading


def _key_rotator(api_keys: list[str]):
    """回傳一個把 params['api_key'] 換成下一把 key 的函式（round-robin）。"""
    assert api_keys, "至少要一把 api_key"
    keys = itertools.cycle(api_keys)
    lock = threading.Lock()  # ponytail: cycle 非執行緒安全，crewai 可能並發呼叫

    def rotate(params: dict) -> dict:
        with lock:
            params["api_key"] = next(keys)
        return params

    return rotate


def install_key_rotation(api_keys: list[str]) -> None:
    """把 key 輪替掛到 crewai LLM（只掛一次）。"""
    from crewai import LLM  # 延後匯入：自我檢查免依賴 crewai

    if getattr(LLM, "_key_rotation_installed", False):
        return
    rotate = _key_rotator(api_keys)
    orig = LLM._prepare_completion_params

    def patched(self, *args, **kwargs):
        return rotate(orig(self, *args, **kwargs))

    LLM._prepare_completion_params = patched
    LLM._key_rotation_installed = True


if __name__ == "__main__":  # 自我檢查（不需 crewai）
    r = _key_rotator(["a", "b", "c"])
    got = [r({})["api_key"] for _ in range(4)]
    assert got == ["a", "b", "c", "a"], got
    print("rotation OK:", got)
