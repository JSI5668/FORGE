"""Thin wrapper around `lerobot-train` used by scripts/train.sh.

It disables pinned memory in the DataLoader and non-blocking host-to-device copies.
On our machine (RTX PRO 4500 Blackwell, driver 595.91) both made the CUDA driver
segfault every few minutes during training; with them disabled, training runs
normally at a small speed cost. All command-line arguments are passed unchanged
to `lerobot-train`.
"""
import sys

import torch.utils.data

_DL_init = torch.utils.data.DataLoader.__init__


def _init_no_pin(self, *a, **k):
    k["pin_memory"] = False
    _DL_init(self, *a, **k)


torch.utils.data.DataLoader.__init__ = _init_no_pin

from lerobot.processor import device_processor  # noqa: E402

_post = device_processor.DeviceProcessorStep.__post_init__


def _post_no_nb(self):
    _post(self)
    self.non_blocking = False


device_processor.DeviceProcessorStep.__post_init__ = _post_no_nb

from lerobot.scripts import lerobot_train  # noqa: E402

if __name__ == "__main__":
    sys.argv[0] = "lerobot-train"
    sys.exit(lerobot_train.main())
