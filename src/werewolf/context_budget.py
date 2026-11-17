"""TASK-046: 実効コンテキスト予算の単一値解決と、リクエスト予算の構成分解。

SPECIFICATION.md の優先順位（プレイヤー個別設定 > CLI設定 > APIリクエスト明示値 >
サーバー側環境・既定値）を「1つの採用値」として解決する。観測されたサーバー側実効値
（`/api/ps` の CONTEXT、プロセス env、Modelfile の PARAMETER num_ctx）は設定値と
比較し、設定値が観測値を超える場合は `conflict` を記録する（設定値と実効値の曖昧な
境界をログで検証可能にするため）。発言の自由度・口調には一切干渉しない。

名称規約（TASK-046レビュー指摘に基づく命名統一）:
- `configured_context_limit`: プレイヤー/CLI設定由来の上限（明示設定はOllama既定値を上書きする）
- `server_observed_context_limit`: /api/ps・プロセスenv・Modelfileから観測したサーバー側実効値
- `effective_requested_num_ctx`: fit適用後に実際に送信したnum_ctx（ログキー）
- `effective_context_limit`: 互換名称。優先順位で採用された単一値（`limit`）であり、
  設定があれば configured、設定なければ server_observed と同値になる。
"""

import os
from dataclasses import dataclass, field

# 解決優先順位（高い順）。SPECIFICATION.md §6.2 の num_ctx 優先順位と同じ趣旨。
CONTEXT_SOURCE_PRIORITY: tuple[str, ...] = (
    "player_setting",
    "cli_setting",
    "api_ps_loaded",
    "process_env",
    "modelfile",
)

# Ollama のコンテキスト関連環境変数（Pythonプロセスから観測できる名前）。
# 旧名 OLLAMA_NUM_CTX と新名 OLLAMA_CONTEXT_LENGTH の両方を受け取る。
CONTEXT_ENV_KEYS: tuple[str, ...] = ("OLLAMA_NUM_CTX", "OLLAMA_CONTEXT_LENGTH")

# サーバー側観測値（設定値の実効上限として比較対象になる候補）。
SERVER_OBSERVED_SOURCES: frozenset[str] = frozenset(
    {"api_ps_loaded", "process_env", "modelfile"}
)


def parse_positive_int(value: object) -> int | None:
    """正の整数へ解釈できる値だけを返す（環境変数文字列・bool・Noneを除外）。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def process_env_context(
    environ: dict[str, str] | None = None,
) -> tuple[int | None, str | None]:
    """Pythonプロセス環境の Ollama コンテキスト設定を (値, 変数名) で返す。"""
    env = environ if environ is not None else os.environ
    for key in CONTEXT_ENV_KEYS:
        parsed = parse_positive_int(env.get(key))
        if parsed is not None:
            return parsed, key
    return None, None


@dataclass(frozen=True)
class EffectiveContextResolution:
    """実効コンテキスト上限の単一値解決結果。

    名称の解釈を明確化するため、由来別の値を分けて保持する（TASK-046レビュー指摘）:
    - `configured`: プレイヤー/CLI設定由来の上限（明示指定はサーバー既定値を上書きできる）
    - `server_observed`: /api/ps・env・Modelfile から観測したサーバー側実効値
    - `limit`: 優先順位で採用された単一値（実際にリクエストへ送る根拠値）。
      互換名称 `effective_context_limit` の実体で、設定があれば `configured`、
      設定なければ `server_observed` と同値になる。
    """

    limit: int | None
    source: str | None
    candidates: dict[str, int] = field(default_factory=dict)
    # 採用値がサーバー側観測値（env・Modelfile・/api/ps）の最大値を超えているか。
    conflict: bool = False
    configured: int | None = None
    configured_source: str | None = None
    server_observed: int | None = None
    server_observed_source: str | None = None


# サーバー側観測値の優先順（観測済みのロード時実効値 > プロセスenv > Modelfile既定値）。
SERVER_OBSERVED_PRIORITY: tuple[str, ...] = ("api_ps_loaded", "process_env", "modelfile")

# 設定由来の候補（明示指定はOllamaの既定値を上書きする）。
CONFIGURED_SOURCES: tuple[str, ...] = ("player_setting", "cli_setting")


def resolve_effective_context_limit(
    candidates: dict[str, object],
) -> EffectiveContextResolution:
    """複数ソースの実効num_ctxを1つの採用値（`limit`）へ解決する。

    設定値（player/cli）があればそれを採用する（明示指定はOllama既定値を上書きできるため
    `limit` = `configured`）。設定値がなければサーバー観測値（/api/ps > env > Modelfile の順）を
    採用する（`limit` = `server_observed`）。採用値がサーバー観測値の最大値より大きい場合は
    `conflict=True`（送信num_ctxはOllama既定値を上書きできるが、観測実効値との差を
    ログで検証可能にする）。旧名称 `effective_context_limit` はこの `limit` の互換名称として
    リクエスト設定・イベントに記録される。
    """
    present: dict[str, int] = {}
    for name in CONTEXT_SOURCE_PRIORITY:
        parsed = parse_positive_int(candidates.get(name))
        if parsed is not None:
            present[name] = parsed
    if not present:
        return EffectiveContextResolution(limit=None, source=None, candidates={})
    configured: int | None = None
    configured_source: str | None = None
    for name in CONFIGURED_SOURCES:
        if name in present:
            configured = present[name]
            configured_source = name
            break
    server_observed: int | None = None
    server_observed_source: str | None = None
    for name in SERVER_OBSERVED_PRIORITY:
        if name in present:
            server_observed = present[name]
            server_observed_source = name
            break
    for name in CONTEXT_SOURCE_PRIORITY:
        if name in present:
            adopted = present[name]
            server_values = [
                value
                for source, value in present.items()
                if source in SERVER_OBSERVED_SOURCES
            ]
            conflict = (
                name in {"player_setting", "cli_setting"}
                and bool(server_values)
                and adopted > max(server_values)
            )
            return EffectiveContextResolution(
                limit=adopted,
                source=name,
                candidates=present,
                conflict=conflict,
                configured=configured,
                configured_source=configured_source,
                server_observed=server_observed,
                server_observed_source=server_observed_source,
            )
    return EffectiveContextResolution(limit=None, source=None, candidates=present)
