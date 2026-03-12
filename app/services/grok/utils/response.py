"""
Response formatting utilities for OpenAI-compatible API responses.
"""

import os
import re
import time
import uuid
from datetime import datetime
from typing import Optional


def make_response_id() -> str:
    """Generate a unique response ID."""
    return f"chatcmpl-{int(time.time() * 1000)}{os.urandom(4).hex()}"


def make_chat_chunk(
    response_id: str,
    model: str,
    content: str,
    index: int = 0,
    role: str = "assistant",
    is_final: bool = False,
) -> dict:
    """
    Create an OpenAI-compatible chat completion chunk.

    Args:
        response_id: Unique response ID
        model: Model name
        content: Content to send
        index: Choice index
        role: Role (assistant)
        is_final: Whether this is the final chunk (includes finish_reason)

    Returns:
        Chat completion chunk dict
    """
    choice: dict = {
        "index": index,
        "delta": {
            "role": role,
            "content": content,
        },
    }

    if is_final:
        choice["finish_reason"] = "stop"

    chunk: dict = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }

    if is_final:
        chunk["usage"] = {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "input_tokens_details": {"text_tokens": 0, "image_tokens": 0},
        }

    return chunk


def make_chat_response(
    model: str,
    content: str,
    response_id: Optional[str] = None,
    index: int = 0,
    usage: Optional[dict] = None,
) -> dict:
    """
    Create an OpenAI-compatible non-streaming chat completion response.

    Args:
        model: Model name
        content: Response content
        response_id: Unique response ID (generated if not provided)
        index: Choice index
        usage: Custom usage dict (defaults to zeros)

    Returns:
        Chat completion response dict
    """
    if response_id is None:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    if usage is None:
        usage = {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "input_tokens_details": {"text_tokens": 0, "image_tokens": 0},
        }

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "refusal": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def wrap_image_content(content: str, response_format: str = "url") -> str:
    """
    Wrap image content in markdown format for chat interface.

    Args:
        content: Image URL or base64 data
        response_format: "url" or "b64_json"/"base64"

    Returns:
        Markdown-wrapped image content
    """
    if not content:
        return content

    if response_format == "url":
        return f"![image]({content})"
    else:
        return f"![image](data:image/png;base64,{content})"


def is_link_only_text(text: str) -> bool:
    """
    判断文本是否只包含链接（如 https://t.co/xxx）

    Args:
        text: 要检查的文本

    Returns:
        True if text only contains a URL, False otherwise
    """
    if not text:
        return True
    # 匹配只包含URL的文本（可能有空格）
    url_pattern = r'^\s*https?://[^\s]+\s*$'
    return bool(re.match(url_pattern, text.strip()))


def format_view_count(count: int) -> str:
    """
    格式化浏览量（如 1234567 -> 1.2M）

    Args:
        count: 浏览量数字

    Returns:
        格式化后的字符串
    """
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    else:
        return str(count)


def format_timestamp(iso_time: str) -> str:
    """
    格式化ISO时间戳为可读格式

    Args:
        iso_time: ISO 8601格式的时间字符串

    Returns:
        YYYY-MM-DD格式的日期字符串
    """
    try:
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except:
        return iso_time


def format_web_search_results(results: list[dict], max_count: int = None) -> str:
    """
    格式化网页搜索结果为markdown

    Args:
        results: 网页搜索结果列表
        max_count: 最大显示数量，None表示不限制

    Returns:
        Markdown格式的网页搜索结果
    """
    if not results:
        return ""

    # 限制数量
    display_results = results[:max_count] if max_count else results

    lines = ["### 网页搜索结果\n"]
    for item in display_results:
        url = item.get("url", "")
        title = item.get("title", "")
        preview = item.get("preview", "")

        # 标题后加冒号
        lines.append(f"- **[{title}]({url}):**")
        if preview:
            lines.append(f"  {preview}")
        lines.append("")  # 空行分隔

    return "\n".join(lines)


def format_x_search_results(results: list[dict], max_count: int = None) -> str:
    """
    格式化X搜索结果为markdown

    Args:
        results: X搜索结果列表
        max_count: 最大显示数量，None表示不限制

    Returns:
        Markdown格式的X搜索结果
    """
    if not results:
        return ""

    lines = ["### X 搜索结果\n"]
    count = 0

    for item in results:
        # 如果已达到最大数量，停止
        if max_count and count >= max_count:
            break

        username = item.get("username", "")
        name = item.get("name", "")
        text = item.get("text", "")
        post_id = item.get("postId", "")
        create_time = item.get("createTime", "")
        view_count = item.get("viewCount", 0)

        # 忽略只包含链接的帖子
        if is_link_only_text(text):
            continue

        # 截断过长文本
        if len(text) > 150:
            text = text[:147] + "..."

        # 处理文本中的换行符，确保每行都以 > 开头
        text_lines = text.split('\n')
        formatted_text_lines = [f"> {line}" if line.strip() else ">" for line in text_lines]

        # 格式化浏览量
        view_str = format_view_count(view_count)

        # 格式化时间
        time_str = format_timestamp(create_time)

        # 构建帖子链接
        post_url = f"https://x.com/{username}/status/{post_id}" if post_id else ""

        lines.append(f"> **{name}** (@{username})")
        lines.extend(formatted_text_lines)
        if post_url:
            lines.append(f"> 📅 {time_str} | 👁️ {view_str} views | [查看原帖]({post_url})")
        else:
            lines.append(f"> 📅 {time_str} | 👁️ {view_str} views")
        lines.append("")  # 空行分隔

        count += 1

    # 如果没有有效帖子，返回空字符串
    if count == 0:
        return ""

    return "\n".join(lines)


__all__ = [
    "make_response_id",
    "make_chat_chunk",
    "make_chat_response",
    "wrap_image_content",
    "is_link_only_text",
    "format_view_count",
    "format_timestamp",
    "format_web_search_results",
    "format_x_search_results",
]
