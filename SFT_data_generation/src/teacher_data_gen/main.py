from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from .provider import FatalProviderError, TeacherProvider, build_response_text, parse_response_json
from .utils import (
    append_jsonl,
    build_bad_record,
    build_success_record,
    classify_output,
    compute_run_dir,
    init_stats,
    load_config,
    make_example_id,
    scan_existing_outputs,
    setup_logger,
    update_stats,
    write_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher data from GSM8K questions.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def load_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_cfg = config["dataset"]
    dataset = load_dataset(
        dataset_cfg["name"],
        dataset_cfg["config"],
        split=dataset_cfg["split"],
    )
    limit = int(dataset_cfg["limit"])
    records: list[dict[str, Any]] = []
    for idx in range(min(limit, len(dataset))):
        row = dataset[idx]
        records.append({"id": make_example_id(idx), "question": row["question"]})
    return records


async def run_pipeline(config: dict[str, Any], records: list[dict[str, Any]], provider: Any, run_dir: Path, logger: Any) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    success_path = run_dir / "success.jsonl"
    bad_path = run_dir / "bad.jsonl"
    stats_path = run_dir / "run_stats.json"

    if config["run"]["resume"]:
        done_ids, seen_texts = scan_existing_outputs(run_dir)
    else:
        done_ids, seen_texts = set(), set()
    logger.info("Resume scan complete. done_ids=%s", len(done_ids))

    total_target = min(int(config["dataset"]["limit"]), len(records))
    stats = init_stats(total_target)
    stats["resume_skipped_count"] = 0
    teacher_model = config["teacher"]["model"]
    final_answer_tag = config["output"]["final_answer_tag"]
    max_concurrency = int(config["teacher"]["max_concurrency"])
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_one(item: dict[str, str]) -> dict[str, Any]:
        async with semaphore:
            result = await provider.generate(item["question"])
            return {"item": item, "provider_result": result}

    tasks = []
    written_ids = set(done_ids)
    for item in records[:total_target]:
        if item["id"] in done_ids:
            stats["resume_skipped_count"] += 1
            update_stats(stats, "skipped", 0.0, None, 1)
            continue
        tasks.append(asyncio.create_task(process_one(item)))

    fatal_error = None
    for task in asyncio.as_completed(tasks):
        try:
            payload = await task
        except FatalProviderError as exc:
            fatal_error = str(exc)
            logger.error("Fatal provider error: %s", fatal_error)
            for pending in tasks:
                if not pending.done():
                    pending.cancel()
            break

        item = payload["item"]
        result = payload["provider_result"]
        record_id = item["id"]
        if record_id in written_ids:
            continue

        usage = result.usage
        if result.error is not None:
            bad_record = build_bad_record(
                record_id=record_id,
                question=item["question"],
                teacher_model=teacher_model,
                error=result.error,
                filter_reason="api_failed",
                usage=usage,
            )
            append_jsonl(bad_path, bad_record)
            written_ids.add(record_id)
            update_stats(stats, "failed", result.latency_sec, usage, result.attempt_count)
            write_stats(stats_path, stats)
            continue

        try:
            parsed_reasoning, answer = parse_response_json(result.content)
        except json.JSONDecodeError as exc:
            bad_record = build_bad_record(
                record_id=record_id,
                question=item["question"],
                teacher_model=teacher_model,
                error={"kind": "invalid_json", "message": str(exc)},
                filter_reason="invalid_json",
                usage=usage,
                finish_reason=result.finish_reason,
                content_preview=result.content[:200],
            )
            append_jsonl(bad_path, bad_record)
            written_ids.add(record_id)
            update_stats(stats, "filtered", result.latency_sec, usage, result.attempt_count)
            write_stats(stats_path, stats)
            continue
        except KeyError as exc:
            filter_reason = "missing_reasoning" if "reasoning" in str(exc) else "missing_answer"
            bad_record = build_bad_record(
                record_id=record_id,
                question=item["question"],
                teacher_model=teacher_model,
                error={"kind": filter_reason, "message": str(exc)},
                filter_reason=filter_reason,
                usage=usage,
                finish_reason=result.finish_reason,
                content_preview=result.content[:200],
            )
            append_jsonl(bad_path, bad_record)
            written_ids.add(record_id)
            update_stats(stats, "filtered", result.latency_sec, usage, result.attempt_count)
            write_stats(stats_path, stats)
            continue
        except Exception as exc:
            bad_record = build_bad_record(
                record_id=record_id,
                question=item["question"],
                teacher_model=teacher_model,
                error={"kind": "parse_error", "message": str(exc)},
                filter_reason="invalid_json",
                usage=usage,
                finish_reason=result.finish_reason,
                content_preview=result.content[:200],
            )
            append_jsonl(bad_path, bad_record)
            written_ids.add(record_id)
            update_stats(stats, "filtered", result.latency_sec, usage, result.attempt_count)
            write_stats(stats_path, stats)
            continue

        final_reasoning = parsed_reasoning or result.reasoning
        response_text = build_response_text(final_reasoning, answer, final_answer_tag)
        filter_reason = classify_output(final_reasoning, answer, result.finish_reason, response_text, seen_texts)
        if filter_reason is not None:
            bad_record = build_bad_record(
                record_id=record_id,
                question=item["question"],
                teacher_model=teacher_model,
                error=None,
                filter_reason=filter_reason,
                usage=usage,
                finish_reason=result.finish_reason,
                content_preview=result.content[:200],
            )
            append_jsonl(bad_path, bad_record)
            written_ids.add(record_id)
            update_stats(stats, "filtered", result.latency_sec, usage, result.attempt_count)
            write_stats(stats_path, stats)
            continue

        success_record = build_success_record(
            record_id=record_id,
            question=item["question"],
            reasoning=final_reasoning,
            answer=answer,
            response_text=response_text,
            teacher_model=teacher_model,
            usage=usage,
        )
        append_jsonl(success_path, success_record)
        seen_texts.add(response_text)
        written_ids.add(record_id)
        update_stats(stats, "success", result.latency_sec, usage, result.attempt_count)
        write_stats(stats_path, stats)

    if fatal_error:
        logger.error("Run stopped early because of a fatal provider error.")
    logger.info(
        "Run complete. success=%s failed=%s filtered=%s skipped=%s",
        stats["success_count"],
        stats["failed_count"],
        stats["filtered_count"],
        stats["skipped_count"],
    )
    write_stats(stats_path, stats)
    return stats


async def async_main(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    run_dir = compute_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir)
    logger.info("Starting run in %s", run_dir)
    logger.info("Config summary: dataset_limit=%s model=%s resume=%s concurrency=%s use_thinking=%s",
                config["dataset"]["limit"],
                config["teacher"]["model"],
                config["run"]["resume"],
                config["teacher"]["max_concurrency"],
                config["teacher"].get("use_thinking", False))
    records = load_records(config)
    provider = TeacherProvider(config)
    return await run_pipeline(config, records, provider, run_dir, logger)


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args.config))


if __name__ == "__main__":
    main()
