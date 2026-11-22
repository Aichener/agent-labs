# agent-labs

两个 Agent 项目，都是纯 Python 标准库能跑起来的。不接外部模型也能看到完整结果，接真模型只改一个参数。

两个都是先定评测口径再写实现的。我觉得没有数字的功能演示不算验证，只能算跑通。

## 合同核查（contract-agent）

输入一份合同和要查的规则，输出没通过的条款、原文、位置和修改建议。

模型负责出计划，判断在代码里。六条规则的命中、取证、结论合成都是普通函数，所以同一份合同跑多少次结果都一样，也能写单测覆盖。

4 份合同、20 条任务自己造的。Mock 下全过，关键词基线 35%；接本地 qwen2.5:7b 之后成功率还是 100%，但计划兜底率 25%——20 条里有 5 次模型给的计划不合法，靠校验和兜底计划救回来。

```bash
cd contract-agent
python -m agent.cli --doc doc_a
python eval/run_eval.py
python -m unittest discover -s tests     # 60 个
```

## 文档分级（doc-classify-agent）

输入一份文档，判敏感等级（L1–L4），给出判定的原句和是否要人工复核。

三条路并行：词库匹配、句库匹配、语义规则。三路各出结论，按证据类型加权后定档。任何一路和最终结论不一致就转人工——系统不自己往下压等级。

8 份文档，其中两份是故意造的「单通道必漏」样本。只用语义词库 88%，只用词库 75%，三路融合 100%，严重漏检从 12% 降到 0。

```bash
cd doc-classify-agent
python -m agent.cli --doc d7
python eval/run_eval.py
python -m unittest discover -s tests     # 60 个
```

## 跑成服务

两个项目都是 FastAPI + uvicorn，带 OpenAPI 文档（`/docs`），依赖在 `requirements.txt` 里。

```bash
cd contract-agent
pip install -r requirements.txt
set AGENT_API_KEYS=k1:tenant-a
python -m agent.api          # 127.0.0.1:8080，文档在 /docs

cd doc-classify-agent
set AGENT_API_KEYS=k1:tenant-a
python -m agent.api          # 127.0.0.1:8081
```

```bash
curl -X POST http://127.0.0.1:8080/v1/review \
  -H "X-API-Key: k1" -H "Content-Type: application/json" \
  -d "{\"doc_id\":\"doc_a\"}"
```

有三件事是刻意这么做的：

结果按 key 对应的租户存，拿别的租户的 `request_id` 去取会被挡回来（403）。同样的正文和规则再提交一次，直接返回上次结果，`replayed: true`，`request_id` 还是原来那个，不会再跑一遍模型。结果落在 SQLite 里，进程杀掉重启，按 id 还能取回。

也有几件没做完，写在前面免得误会：超时回 504，但后台线程不会被真正取消，只是先让请求结束；上传的正文只在内存里过一下，请求结束就释放。上传支持 txt、md、docx、pdf，后两种要装依赖（pip install -r requirements.txt：python-docx、PyMuPDF）。pdf 用 PyMuPDF 取文本层，中文能正常提；扫描件没有文本层会返回 422，不硬判。

## 部署

本地跑看上面那几条命令。要放到服务器上，`deploy/` 里有现成的 compose 和反代示例。

```bash
cd deploy
cp .env.example .env          # 填 AGENT_API_KEYS，按需改模型地址
docker compose up -d --build
```

两个服务分别是 8080 和 8081。结果库挂在命名卷 `contract-state` / `classify-state` 里，删容器不丢；语料和代码在镜像里，不挂载。镜像构建时会先跑一遍测试，不过就构建失败——这是故意的，省得把坏版本推上去。

环境变量两个服务一样，只有端口不同：

| 变量 | 作用 | 默认 |
|---|---|---|
| `AGENT_API_KEYS` | key:租户，逗号分隔 | 无（缺省用仅本机的 dev-key） |
| `AGENT_DB` | SQLite 路径 | `data/tasks.db`（镜像里是 `/app/state/tasks.db`） |
| `AGENT_TIMEOUT` | 单请求超时秒数 | 30 |
| `AGENT_HOST` / `AGENT_PORT` | 监听地址 | `127.0.0.1` / 8080、8081 |
| `LLM_BASE_URL` / `LLM_MODEL` | 合同核查的模型端点，不填走 Mock | Ollama 默认值 |

反代看 `deploy/nginx.conf`：TLS 终止、请求体 10m（服务端上限 8m，留点余量）、读写超时 60s（让 504 由服务自己回），`/healthz` 由 nginx 直接返回 200，不打到业务上。

升级就是换镜像 tag 再 `up -d`；回滚把 tag 换回上一个。备份用命名卷：

```bash
docker run --rm -v deploy_contract-state:/state -v "$PWD:/backup" alpine \
  sh -c "tar czf /backup/contract-state-\$(date +%F).tgz -C /state ."
```

如果不想用 Docker，两个服务都是纯 Python，`python -m agent.api` 起进程，配 systemd 或 supervisor 守着即可，注意把 `AGENT_DB` 指到一个可写的绝对路径。

有几条限制得说在前面，免得按「随便扩」的思路部署：

- 结果库是 SQLite，单写者。每个服务只能跑一个副本，`replicas` 不能加，要横向扩得先把 store 换成 Postgres。
- 服务内是 4 线程池，uvicorn 单进程，没有 CPU / 内存配额。一份 200 页文档进来会占满一个 worker，并发高了要自己排队或前置限流。别开 `--workers`，SQLite 会打架。
- 服务本身不带 TLS 和访问频率限制，这两件事交给反代。
- 超时返回 504 只是让请求先结束，后台线程不会被真正取消。

这些配置只做过 YAML 语法检查，没有在 CI 里真实构建过（本机也没装 Docker），第一次上服务器时请先 `docker compose config` 看一眼解析结果。

## 推送前检查

仓库根有一个自检脚本，扫敏感表述、拦不该提交的文件、跑两个项目的测试和门禁，任何一个不过就退出非零。装了 pre-push 钩子，正常 `git push` 会自动执行。

```bash
python tools/preflight.py
```

## 数据

两个项目用的都是合成的合同和文档，词库、句库、规则也是示例。不含客户数据或任何内部资料，所有指标只对仓库里这几份样例有效。

接真模型：

```bash
set LLM_BASE_URL=http://127.0.0.1:11434/v1
set LLM_MODEL=qwen2.5:7b
python -m agent.cli --doc doc_a --llm openai
```
