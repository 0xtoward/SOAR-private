# 飞书长连接 Bot

## 目标

给 SOAR 仓库提供一个最小可用的飞书机器人，支持：

- 查看状态
- 启动 autorun
- 停止 autorun
- 查看 blocker
- 查看实验记录
- 查看最新日志

## 文件

- `tools/feishu_bot.py`
- `tools/feishu_bot.sh`
- `tools/feishu_bot.env.example`

## 依赖

```bash
uv pip install lark-oapi requests
```

## 环境变量

不要把凭证写进仓库文件。推荐在当前 shell 或 tmux 会话里导出：

```bash
export FEISHU_APP_ID="你的 App ID"
export FEISHU_APP_SECRET="你的 App Secret"
export FEISHU_ALLOWED_OPEN_IDS=""
```

如果你只想允许自己执行控制类命令（如 `开始`、`停止`），把 `FEISHU_ALLOWED_OPEN_IDS` 设置为自己的 `open_id`，多个值用逗号分隔。

## 飞书后台配置

需要在飞书开放平台完成这些步骤：

1. 开启机器人能力
2. 订阅事件 `im.message.receive_v1`
3. 事件订阅方式选择“长连接”
4. 申请发送消息权限，建议至少包含以下任一项：
   - `im:message`
   - `im:message:send_as_bot`
   - `im:message:send`
5. 发布应用版本
6. 在企业内安装应用
7. 把机器人加入你要使用的群，或者给机器人发单聊

## 启动

前台运行：

```bash
bash tools/feishu_bot.sh
```

后台 tmux：

```bash
tmux new-session -d -s feishu-bot 'bash -lc "export FEISHU_APP_ID=...; export FEISHU_APP_SECRET=...; bash /root/autodl-tmp/SOAR-Toolkit/tools/feishu_bot.sh"'
```

## 命令

向机器人发送以下文本命令：

- `帮助`
- `状态`
- `开始`
- `停止`
- `blocker`
- `实验记录`
- `日志`

## 说明

- `状态` 会刷新一次 `qwen_guard.sh`，然后返回最新 guard 状态。
- `开始` 会启动 `qwen-soar`；如果未运行 `qwen-guard`，也会顺带拉起。
- `blocker` 只返回“本轮 blocker”；旧 blocker 会被忽略。
- 目前只支持文本消息。
