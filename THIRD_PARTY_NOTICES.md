# Third-Party Notices

This project draws on design and code from two upstream open-source projects.
The code under `src/prismatic/` is a slimmed-down vendoring of the Prismatic
VLM / VLA-Adapter codebase; design patterns under `src/vla_project/` (action
head structure, EE6D action layout, multi-domain projector convention,
per-stage LR curriculum, etc.) are inspired by both VLA-Adapter and X-VLA.
Individual references are noted in module docstrings throughout the source
tree.

This repository is distributed under the Apache License 2.0 (see
[`LICENSE`](LICENSE)). The third-party works listed below retain their
original licenses; this notice fulfils the attribution requirements of those
licenses.

---

## VLA-Adapter

- **Upstream**: <https://github.com/OpenHelix-Team/VLA-Adapter>
- **License**: MIT
- **Used by**: `src/prismatic/` (vendored, slimmed); design references in
  `src/vla_project/models/action_heads/`, `src/vla_project/data/datasets/`,
  `src/vla_project/data/packing/`, etc.

```
MIT License

Copyright (c) 2025 Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui,
Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou,
Siteng Huang, Yifan Tang, Wenhui Wang, Ru Zhang, Jianyi Liu, and
Donglin Wang.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## X-VLA

- **Upstream**: <https://github.com/2toinf/X-VLA>
- **License**: Apache License 2.0
- **Used by**: design references for the action-head transformer block
  layout (self-attention pool, DA-Linear projector convention, EE6D 20-dim
  action encoding, two-step LLM warmup curriculum). No code is vendored
  verbatim.

X-VLA is distributed under the Apache License, Version 2.0; the full text is
available at <http://www.apache.org/licenses/LICENSE-2.0> and matches this
repository's own [`LICENSE`](LICENSE) file.
