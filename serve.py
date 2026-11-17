# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A minimal harness: one model, three tools, one loop.

The model gets the task and three tools that act in the task container. Each
tool call runs there and its result goes back to the model, until the model
answers without calling a tool. That answer is the final message.
"""

import asyncio
import json
import os
import re
import shlex
import time

import openai
import taskbridge

MODEL = os.environ["MODEL_NAME"]
# The model's highest reasoning level: max for Claude, GLM, GPT-6 and
# DeepSeek-V4, high for the others (Gemini refuses max). REASONING_EFFORT
# overrides it.
MAX_EFFORT_MODELS = ("claude", "glm", "gpt-6", "deepseek-v4")
EFFORT = os.environ.get("REASONING_EFFORT") or (
    "max" if any(m in MODEL.lower() for m in MAX_EFFORT_MODELS) else "high"
)
WRAP_UP_SEC = 240  # ask for the summary this long before the time limit
WRAP_UP = {
    "role": "user",
    "content": "Time is almost up. Stop and reply with your summary.",
}
RESULT_CHARS = 20_000  # a longer tool result loses its middle
# Seconds between retries once the SDK's own quick retries are exhausted.
RETRY_WAITS = (5, 15, 30, 60, 120)
# How endpoints word a 400 saying the conversation no longer fits the context
# window: Claude, OpenAI and vLLM, Gemini, xAI.
CONTEXT_FULL = re.compile(
    r"context.?(length|window)|prompt is too long|maximum prompt length"
    r"|exceed[^.]{0,40}(token|context)",
    re.I,
)
CUT_KEEP = 1000  # characters a tool result keeps once the context is full
# times in a row an empty or cut-off reply gets a "continue" before the run
# stops
MAX_NUDGES = 3
# lower it for a small context window
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "32000"))
# Claude caches a prompt prefix only when marked with cache_control. Each call
# marks the system prompt and the latest message (using 2 of Claude's 4 cache
# breakpoints) so each turn reads the prior conversation from cache. Other
# providers cache automatically without markers.
CACHE_MARK = "claude" in MODEL.lower()

# The grader reads only the files the task names; a final message alone scores
# 0 for most tasks. When the model tries to finish without having written any
# deliverable, the loop asks it to write one first (at most this many times).
MAX_DELIVERABLE_NUDGES = 2
# A run can also die before ever trying to finish: the model spends its whole
# request budget on analysis and is cut off by the daily quota or the time
# limit (run 2: 49 requests, 417 s, no file). The finish-time guard above
# cannot save such a run, so the loop also forces an early checkpoint: after
# this many tool-call rounds, and every CHECKPOINT_EVERY_ROUNDS after that,
# a run with no deliverable on disk is told to write one NOW.
CHECKPOINT_FIRST_ROUND = 5
CHECKPOINT_EVERY_ROUNDS = 15
CHECKPOINT_NUDGE = (
    "Checkpoint: stop analyzing and write the deliverable NOW. The grader "
    "reads only the files this task names, so a run that ends with no file "
    "scores 0 however good your analysis is. Write a first COMPLETE version "
    "of {paths} with write_file or bash right now, using the results you "
    "have so far — it does not have to be perfect — then keep improving it "
    "in place. Do not run more analysis before the file exists."
)
DELIVERABLE_NUDGE = (
    "You have not written any deliverable file yet. The grader reads only the "
    "files this task names — for most tasks it does not read your final message, "
    "so a run that ends with no files scores 0. Write the deliverable now, at "
    "the exact path the task names, in a first complete version (it does not "
    "have to be perfect), then reply with your summary."
)
# Extensions a task instruction names for its deliverables.
DELIVERABLE_EXTS = (
    "zip",
    "xlsx",
    "docx",
    "pdf",
    "md",
    "dxf",
    "ifc",
    "json",
    "py",
    "csv",
    "txt",
    "html",
    "xml",
)
_PATH_RE = re.compile(
    r"[A-Za-z0-9_.\-/]*[A-Za-z0-9_\-]/[A-Za-z0-9_.\-/]*\."
    r"(?:" + "|".join(DELIVERABLE_EXTS) + r")"
)
# A 400 from a spent daily model quota never recovers today; do not retry it.
DAILY_LIMIT = re.compile(r"per-day|daily", re.I)


def deliverable_paths(instruction):
  """Extracts candidate deliverable file paths from the task instruction.

  The instructions name their deliverables as absolute container paths
  (``/filesystem/.../output/submission.zip``) or as paths relative to the task
  folder (``output/submission.zip``, ``submission/model.ifc``). A path counts
  when it has a slash and a deliverable extension; bare names such as
  ``task.md`` (an input file) do not.

  Args:
    instruction: Task instruction text from the evaluator.

  Returns:
    De-duplicated candidate paths in order of appearance.
  """
  found = []
  for m in _PATH_RE.finditer(instruction or ""):
    p = m.group(0).strip("`\"'.,;:!?()[]")
    if p and p not in found:
      found.append(p)
  return found


def _resolve_candidates(paths, workdir, instruction=""):
  """Resolves candidate paths against the container locations they may mean.

  An absolute path is used as-is. A relative path is tried against the task
  folder and against ``workdir``. The task folder comes from the
  instruction's "working in ..." line when present, else from the directory
  of the first absolute path in the instruction — that is where relative
  ``output/...`` paths point.

  Args:
    paths: Candidate paths from :func:`deliverable_paths`.
    workdir: The session's working directory in the container.
    instruction: Task instruction text, used to find the task folder.

  Returns:
    De-duplicated absolute container paths to check.
  """
  taskdir = None
  m = re.search(r"working in [`\"']?(/[^\s`\"']+)", instruction or "")
  if m:
    taskdir = m.group(1).rstrip("/")
  if taskdir is None:
    for p in paths:
      if p.startswith("/"):
        taskdir = os.path.dirname(p)
        break
  out = []
  for p in paths:
    cands = [p] if p.startswith("/") else []
    if not p.startswith("/"):
      cands.append(os.path.join(workdir, p))
      if taskdir:
        cands.append(os.path.join(taskdir, p))
    for c in cands:
      if c not in out:
        out.append(c)
  return out


async def _inventory(session):
  """Snapshots the container files under ``workdir`` as ``(sha256, count)``.

  The hash is over the sorted file list, so any file created, changed, or
  deleted changes it; the list itself never leaves the container.

  Args:
    session: Active `taskbridge.Session` connected to the task container.

  Returns:
    A ``(hash, count)`` tuple, or `None` when the snapshot cannot be taken.
  """
  try:
    result = await session.exec(
        "list=$(find "
        + shlex.quote(session.workdir)
        + " -type f 2>/dev/null | sort); "
        "printf '%s' \"$list\" | sha256sum | cut -d' ' -f1; "
        "printf '%s' \"$list\" | grep -c ."
    )
  except (taskbridge.TaskError, taskbridge.Cancelled):
    return None
  lines = (result.get("output") or "").strip().splitlines()
  if len(lines) < 2:
    return None
  try:
    return lines[0].strip(), int(lines[1].strip())
  except ValueError:
    return None


async def deliverables_present(session, instruction, start_epoch, start_inventory):
  """Returns whether the agent has written a deliverable for this task.

  When the instruction names deliverable paths, a path counts only if it
  exists and was written after the run started, so pre-existing input files
  the instruction happens to mention do not count. When it names none, any
  change to the container's file inventory since the run started counts. A
  check that cannot run fails open (the run is allowed to finish) rather than
  blocking on a broken probe.

  Args:
    session: Active `taskbridge.Session` connected to the task container.
    instruction: Task instruction text from the evaluator.
    start_epoch: Unix time just before the agent loop started.
    start_inventory: The :func:`_inventory` snapshot from the run start, or
      `None` when it could not be taken.

  Returns:
    `True` when a deliverable looks present, else `False`.
  """
  paths = deliverable_paths(instruction)
  if paths:
    quoted = " ".join(
        shlex.quote(c)
        for c in _resolve_candidates(paths, session.workdir, instruction)
    )
    try:
      result = await session.exec(f"stat -c '%Y %n' {quoted} 2>/dev/null")
    except (taskbridge.TaskError, taskbridge.Cancelled):
      return True
    for line in (result.get("output") or "").splitlines():
      try:
        mtime = int(line.split(" ", 1)[0])
      except ValueError:
        continue
      if mtime >= start_epoch - 5:
        return True
    return False
  if start_inventory is None:
    return True
  now = await _inventory(session)
  return now is None or now != start_inventory

# The evaluator sets OPENAI_BASE_URL and OPENAI_API_KEY for every model.
# The SDK handles quick retries; recover() handles longer backoffs. Because
# calls are non-streaming, a max-effort response can take over 15 minutes
# before any bytes arrive, so the timeout matches the egress proxy's 30-minute
# idle limit.
client = openai.AsyncOpenAI(max_retries=5, timeout=1800)

SYSTEM = """You are an expert engineer working on a task in a Linux container with no internet access. Your tools \
act inside it, and bash starts in {workdir}. Read the task and the files it points to before you start. Compute with \
code, not by hand. Write each deliverable exactly where and in the format the task asks, then open it again to check \
it. When you are done, reply without calling a tool: a short summary of each deliverable, where it is, and the key \
results.

Deliverables are files, not chat: the grader reads only the files the task names, and for most tasks it does not \
read your final message at all, so a run that ends with no deliverable file on disk scores 0 however good your \
summary is. Write a first COMPLETE version of every deliverable the task asks for as early as you can — within your \
first few tool calls, before deep analysis — then improve it in place; the last version on disk when the run ends is \
what gets graded, and an imperfect early file earns partial credit while no file earns none. Work efficiently: batch \
independent tool calls in a single turn, and keep tool outputs short by running focused queries and summarizing with \
code instead of dumping whole files into your context."""


def tool(name, description, **params):
  """Builds an OpenAI function-tool definition dictionary.

  Args:
    name: Tool function name.
    description: Human-readable tool description.
    **params: Parameter names mapped to their JSON Schema type strings.

  Returns:
    An OpenAI function-tool specification dictionary.
  """
  return {
      "type": "function",
      "function": {
          "name": name,
          "description": description,
          "parameters": {
              "type": "object",
              "properties": {k: {"type": v} for k, v in params.items()},
              "required": list(params),
          },
      },
  }


TOOLS = [
    tool(
        "bash",
        "Run a bash command (120 s limit) and return its exit code and output.",
        command="string",
    ),
    tool("read_file", "Read a text file.", path="string"),
    tool(
        "write_file",
        "Create or overwrite a file; parent folders are created.",
        path="string",
        content="string",
    ),
]


def clip(text):
  """Truncates `text` in the middle if it exceeds `RESULT_CHARS` characters.

  Args:
    text: Tool output string to clip.

  Returns:
    The original string if within `RESULT_CHARS`, or a middle-truncated string.
  """
  if len(text) <= RESULT_CHARS:
    return text
  half = RESULT_CHARS // 2
  return (
      f"{text[:half]}\n[... {len(text) - RESULT_CHARS} characters cut"
      f" ...]\n{text[-half:]}"
  )


def marked(messages):
  """Returns the messages a model call sends.

  For Claude, this is a copy whose system prompt and newest message carry
  `cache_control`. The stored conversation stays as it is, so each turn marks
  its own newest message.

  Args:
    messages: The conversation so far, in OpenAI chat format.

  Returns:
    `messages` itself for other models, or the marked copy for Claude.
  """
  if not CACHE_MARK:
    return messages
  out = list(messages)
  for i in {0, len(out) - 1}:
    m, content = out[i], out[i].get("content")
    if isinstance(content, str) and content:
      out[i] = {
          **m,
          "content": [{
              "type": "text",
              "text": content,
              "cache_control": {"type": "ephemeral"},
          }],
      }
    else:
      out[i] = {**m, "cache_control": {"type": "ephemeral"}}
  return out


async def run_tool(session, fn):
  """Executes a single tool call in the task container and returns its output.

  Rewrites `fn["arguments"]` as canonical JSON (or `"{}"` if invalid) so the
  conversation history remains valid for strict endpoints. Any tool failure is
  returned as an error string for the model.

  Args:
    session: Active `taskbridge.Session` connected to the task container.
    fn: Function-call dictionary containing `name` and `arguments`.

  Returns:
    The clipped tool output or error string to send back to the model.
  """
  name = fn["name"]
  try:
    args = json.loads(fn["arguments"] or "{}")
    if not isinstance(args, dict):
      raise TypeError("not a JSON object")
    fn["arguments"] = json.dumps(args, allow_nan=False)
  except (TypeError, ValueError) as exc:
    fn["arguments"] = "{}"
    return (
        f"error: the arguments are not a JSON object ({exc}); send the call"
        " again"
    )
  try:
    if name == "bash":
      result = await session.exec(args["command"])
      return clip(f"exit code {result['exit_code']}\n{result['output']}")
    if name == "read_file":
      return clip(
          (await session.read_file(args["path"])).decode("utf-8", "replace")
      )
    if name == "write_file":
      return (
          f"wrote {await session.write_file(args['path'], args['content'])}"
          f" bytes to {args['path']}"
      )
    return f"error: there is no tool named {name}"
  # ValueError: content with an unpaired surrogate
  except (taskbridge.TaskError, KeyError, ValueError) as exc:
    return f"error: {type(exc).__name__}: {exc}"


async def recover(session, messages, exc, waits, until):
  """Handles a failed model call and returns whether to retry.

  On a 400 context-overflow error, shortens the oldest tool results. On a rate
  limit, server error, or connection drop, sleeps for the next interval in
  `RETRY_WAITS` (capped at the wrap-up deadline). Errors are logged rather than
  added to the conversation.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: The conversation, shortened in place when the context is full.
    exc: The `openai.APIError` the call raised.
    waits: Iterator over the remaining intervals in `RETRY_WAITS`.
    until: Seconds before the time limit at which the wrap-up starts.

  Returns:
    `True` to retry the call, `False` to re-raise.
  """
  status = getattr(exc, "status_code", None)  # None: no HTTP answer
  if status == 400 and CONTEXT_FULL.search(str(exc)):
    return await shorten(session, messages)
  if status == 429 and DAILY_LIMIT.search(str(exc)):
    # A spent daily quota never recovers today: retrying burns the run's
    # remaining time on calls that will all fail the same way.
    await session.log("model call failed: the daily model quota is spent")
    return False
  wait = next(waits, None)
  if wait is None or not (
      status is None or status in (408, 409, 429) or status >= 500
  ):
    return False
  left = session.time_left()
  if left is not None and left > until:
    wait = min(wait, left - until)
  await session.log(
      f"model call failed ({str(exc)[:200]}); again in {wait:.0f} s"
  )
  await asyncio.sleep(wait)
  return True


async def shorten(session, messages):
  """Truncates the older half of long tool results to `CUT_KEEP` characters.

  Repeated calls truncate progressively newer results. Claude's thinking blocks
  are kept intact because each block's signature is self-contained and Claude
  rejects tool loops if the latest assistant turn loses its thinking block.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: The conversation, modified in place.

  Returns:
    Whether any tool result was truncated.
  """
  long = [
      n
      for n, m in enumerate(messages)
      if m["role"] == "tool" and len(m["content"]) > 2 * CUT_KEEP
  ]
  if not long:
    return False
  cut = long[: max(1, len(long) // 2)]
  for n in cut:
    messages[n]["content"] = (
        messages[n]["content"][:CUT_KEEP]
        + "\n[... cut to save context; run the tool again if needed]"
    )
  await session.log(f"context full: cut {len(cut)} old tool results")
  return True


async def solve(session):
  """Runs the minimal tool-calling loop on `session` until completion.

  Args:
    session: Active `taskbridge.Session` for the task.

  Returns:
    The final summary message from the model.
  """
  messages = [
      {"role": "system", "content": SYSTEM.format(workdir=session.workdir)},
      {"role": "user", "content": session.instruction},
  ]
  # The deliverable guard compares against the container as the run found it.
  start_epoch = time.time()
  start_inventory = await _inventory(session)
  # head: the start of an answer cut off at the output limit
  nudges, head, waits = 0, "", iter(RETRY_WAITS)
  deliverable_nudges = 0
  rounds, next_checkpoint = 0, CHECKPOINT_FIRST_ROUND
  while True:
    left = session.time_left()  # None when the task has no time limit
    wrap_up = left is not None and left < WRAP_UP_SEC
    if wrap_up and messages[-1:] != [WRAP_UP]:  # asked once, also on a retry
      messages.append(dict(WRAP_UP))
    try:  # a call still running when the wrap-up starts is stopped
      reply = await asyncio.wait_for(
          client.chat.completions.create(
              model=MODEL,
              messages=marked(messages),
              tools=TOOLS,
              tool_choice="none" if wrap_up else "auto",
              reasoning_effort=EFFORT,
              max_completion_tokens=MAX_OUTPUT_TOKENS,
          ),
          None if left is None or wrap_up else left - WRAP_UP_SEC,
      )
    except asyncio.TimeoutError:
      continue
    except openai.APIError as exc:
      if await recover(session, messages, exc, waits, WRAP_UP_SEC):
        continue
      raise
    waits = iter(RETRY_WAITS)
    if reply.usage:
      details = getattr(reply.usage, "prompt_tokens_details", None)
      await session.usage(
          MODEL,
          reply.usage.prompt_tokens,
          reply.usage.completion_tokens,
          getattr(details, "cached_tokens", 0) or 0,
      )
    message, cut = (
        reply.choices[0].message,
        reply.choices[0].finish_reason == "length",
    )
    text = head + (message.content or "")
    if text:
      # what taskbridge sends if the time runs out first
      session.final_message = text
    # Send the message back as the endpoint returned it: Gemini refuses a tool
    # call that comes back without its thought signature, which is one of the
    # endpoint's own fields.
    turn = message.model_dump(exclude_none=True)
    if wrap_up or not message.tool_calls:
      if (
          not wrap_up
          and not message.tool_calls
          and deliverable_nudges < MAX_DELIVERABLE_NUDGES
          and not await deliverables_present(
              session, session.instruction, start_epoch, start_inventory
          )
      ):
        # The model is about to finish with nothing on disk to grade. Ask it
        # to write the deliverable first; the check runs again if it tries to
        # finish a second time. At wrap-up there is no time left to write, so
        # the guard does not run there.
        deliverable_nudges += 1
        if message.content:  # keep the model's text in the conversation
          messages.append(turn)
        messages.append({"role": "user", "content": DELIVERABLE_NUDGE})
        await session.log(
            "no deliverable file on disk yet; asked the model to write it"
        )
        continue
      if wrap_up or (message.content and not cut) or nudges == MAX_NUDGES:
        return session.final_message
      nudges, head = nudges + 1, text
      if message.content:  # cut off at the output limit
        messages.append(turn)
      messages.append({
          "role": "user",
          "content": (
              "Your reply was cut off; continue where it stopped."
              if cut
              else "Continue the task, and finish with your summary."
          ),
      })
      continue
    nudges, head = 0, ""
    messages.append(turn)
    for call in turn["tool_calls"]:
      await session.log(
          f"{call['function']['name']} {call['function']['arguments'][:200]}"
      )
      content = await run_tool(session, call["function"])
      messages.append(
          {"role": "tool", "tool_call_id": call["id"], "content": content}
      )
    if left is not None:
      messages[-1][
          "content"
      ] += f"\n[{int(session.time_left() // 60)} minutes left]"
    rounds += 1
    if rounds >= next_checkpoint:
      # At wrap-up there is no time left to write, so the checkpoint does
      # not run there; the finish-time guard already covers that case.
      if not wrap_up and not await deliverables_present(
          session, session.instruction, start_epoch, start_inventory
      ):
        paths = deliverable_paths(session.instruction)
        named = (
            ", ".join(
                _resolve_candidates(
                    paths, session.workdir, session.instruction
                )
            )
            if paths
            else "the deliverable file(s) the task asks for"
        )
        messages.append({
            "role": "user",
            "content": CHECKPOINT_NUDGE.format(paths=named),
        })
        await session.log(
            f"round {rounds}: no deliverable on disk yet; forced a "
            "checkpoint write"
        )
      next_checkpoint = rounds + CHECKPOINT_EVERY_ROUNDS


if __name__ == "__main__":
  taskbridge.run(solve)
