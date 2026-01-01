# IsaacNPC

**IsaacNPC** is a lightweight NPC (Non-Player Character) framework for **IsaacLab**, enabling robots to act as **autonomous environment entities** driven by pre-trained policies or rule-based controllers.

It introduces a **Null Action** mechanism that decouples environment actions from robot control, allowing NPCs to move autonomously **without defining explicit action dimensions**.

## Core Idea: Null Action

- Exposed action has **zero dimensions**
- Environment always receives the same null action
- Control logic is executed internally by the NPC

> Action is not control.

Controllers can be:
- Pre-trained neural policies
- Rule-based / heuristic systems
- Hybrid approaches

## Features

- Zero-dimensional action space
- Plug-and-play NPC configuration
- Controller-agnostic design
- Native IsaacLab integration
- Reusable across environments and robots

---

## Installation

```bash
cd IsaacNPC
pip install -e .
````

### Minimal Usage

Refer to `IsaacNPC/template`

## Author

**Ziang Zheng**
Tsinghua University

### Contact

You can join the WeChat group for detialed contact!

| Renforce Dynamics | **Join our WeChat Group**                                                                                                                        |
| :---------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- |
|                   | <img src="https://github.com/Renforce-Dynamics/.github/blob/main/pics/wechat_group/group.jpg" alt="Renforce Dynamics WeChat Group" height="180"> |
