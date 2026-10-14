"""SkillGrad runners package.

Importing this package patches `asyncio.base_subprocess.BaseSubprocessTransport.__del__`
to suppress the harmless "Event loop is closed" RuntimeError that fires when GC
cleans up subprocess transports after `asyncio.run()` returns.
"""


def _suppress_async_shutdown_errors() -> None:
    """Patch subprocess transport __del__ to suppress 'Event loop is closed'.

    When asyncio.run() finishes, the event loop closes. Subprocess transport
    objects cleaned up later by GC try to call loop.call_soon() in __del__,
    raising RuntimeError. This is harmless but noisy. Patch once at import.
    """
    try:
        from asyncio import base_subprocess
        _orig = base_subprocess.BaseSubprocessTransport.__del__

        def _quiet_del(self):
            try:
                _orig(self)
            except RuntimeError:
                pass

        base_subprocess.BaseSubprocessTransport.__del__ = _quiet_del
    except (ImportError, AttributeError):
        pass


_suppress_async_shutdown_errors()
