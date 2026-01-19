"""Async task scheduler using asyncio.

Provides scheduling for periodic tasks like:
- Market data collection
- Opportunity scanning
- Risk monitoring
- Report generation
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine

from ..core.utils import get_logger

logger = get_logger(__name__)


class JobStatus(Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobRun:
    """Record of a job execution.

    Attributes:
        started_at: When the run started.
        ended_at: When the run ended.
        status: Final status.
        error: Error message if failed.
        duration_seconds: Execution time.
    """

    started_at: datetime
    ended_at: datetime | None = None
    status: JobStatus = JobStatus.RUNNING
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        """Get execution duration."""
        if self.ended_at is None:
            return (datetime.now(timezone.utc) - self.started_at).total_seconds()
        return (self.ended_at - self.started_at).total_seconds()


@dataclass
class Job:
    """Scheduled job definition.

    Attributes:
        name: Job identifier.
        func: Async function to execute.
        interval_seconds: Seconds between executions.
        enabled: Whether job is active.
        max_retries: Max retries on failure.
        retry_delay_seconds: Delay between retries.
        run_immediately: Run once on scheduler start.
        last_run: Most recent run record.
        history: List of past runs.
    """

    name: str
    func: Callable[[], Coroutine[Any, Any, Any]]
    interval_seconds: float
    enabled: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    run_immediately: bool = False

    # Runtime state
    last_run: JobRun | None = field(default=None, repr=False)
    history: list[JobRun] = field(default_factory=list, repr=False)
    _task: asyncio.Task | None = field(default=None, repr=False)
    _consecutive_failures: int = field(default=0, repr=False)

    @property
    def next_run(self) -> datetime | None:
        """Estimate next run time."""
        if not self.enabled:
            return None

        if self.last_run is None:
            return datetime.now(timezone.utc)

        if self.last_run.ended_at is None:
            return None  # Currently running

        return self.last_run.ended_at + timedelta(seconds=self.interval_seconds)

    @property
    def success_rate(self) -> float:
        """Get success rate from history."""
        if not self.history:
            return 1.0

        successes = sum(
            1 for run in self.history if run.status == JobStatus.COMPLETED
        )
        return successes / len(self.history)

    @property
    def average_duration(self) -> float:
        """Get average execution duration."""
        completed = [
            run for run in self.history
            if run.status == JobStatus.COMPLETED and run.ended_at
        ]
        if not completed:
            return 0.0

        return sum(run.duration_seconds for run in completed) / len(completed)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "interval_seconds": self.interval_seconds,
            "enabled": self.enabled,
            "success_rate": self.success_rate,
            "average_duration": self.average_duration,
            "consecutive_failures": self._consecutive_failures,
            "last_run": {
                "started_at": self.last_run.started_at.isoformat(),
                "status": self.last_run.status.value,
                "duration": self.last_run.duration_seconds,
            } if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
        }


class Scheduler:
    """Async task scheduler.

    Usage:
        scheduler = Scheduler()

        # Add jobs
        scheduler.add_job(
            name="collect_markets",
            func=collector.collect,
            interval_seconds=300,  # 5 minutes
        )

        scheduler.add_job(
            name="scan_opportunities",
            func=scanner.scan,
            interval_seconds=60,  # 1 minute
            run_immediately=True,
        )

        # Run scheduler
        await scheduler.start()

        # Later...
        await scheduler.stop()
    """

    def __init__(
        self,
        max_history_per_job: int = 100,
        error_callback: Callable[[str, Exception], Coroutine[Any, Any, None]] | None = None,
    ):
        """Initialize scheduler.

        Args:
            max_history_per_job: Max runs to keep in history per job.
            error_callback: Async callback for job errors.
        """
        self._jobs: dict[str, Job] = {}
        self._running = False
        self._max_history = max_history_per_job
        self._error_callback = error_callback
        self._main_task: asyncio.Task | None = None

    def add_job(
        self,
        name: str,
        func: Callable[[], Coroutine[Any, Any, Any]],
        interval_seconds: float,
        enabled: bool = True,
        max_retries: int = 3,
        run_immediately: bool = False,
    ) -> Job:
        """Add a job to the scheduler.

        Args:
            name: Unique job name.
            func: Async function to execute.
            interval_seconds: Seconds between executions.
            enabled: Whether job starts enabled.
            max_retries: Max retries on failure.
            run_immediately: Run once on start.

        Returns:
            The created Job.
        """
        if name in self._jobs:
            raise ValueError(f"Job '{name}' already exists")

        job = Job(
            name=name,
            func=func,
            interval_seconds=interval_seconds,
            enabled=enabled,
            max_retries=max_retries,
            run_immediately=run_immediately,
        )

        self._jobs[name] = job
        logger.info(f"Added job '{name}' with interval {interval_seconds}s")

        return job

    def remove_job(self, name: str) -> None:
        """Remove a job.

        Args:
            name: Job name to remove.
        """
        if name not in self._jobs:
            return

        job = self._jobs[name]
        if job._task and not job._task.done():
            job._task.cancel()

        del self._jobs[name]
        logger.info(f"Removed job '{name}'")

    def enable_job(self, name: str) -> None:
        """Enable a job.

        Args:
            name: Job name to enable.
        """
        if name in self._jobs:
            self._jobs[name].enabled = True
            logger.info(f"Enabled job '{name}'")

    def disable_job(self, name: str) -> None:
        """Disable a job.

        Args:
            name: Job name to disable.
        """
        if name in self._jobs:
            self._jobs[name].enabled = False
            logger.info(f"Disabled job '{name}'")

    def get_job(self, name: str) -> Job | None:
        """Get a job by name.

        Args:
            name: Job name.

        Returns:
            Job or None if not found.
        """
        return self._jobs.get(name)

    def get_all_jobs(self) -> list[Job]:
        """Get all jobs.

        Returns:
            List of all jobs.
        """
        return list(self._jobs.values())

    async def run_job_once(self, name: str) -> bool:
        """Run a job immediately (one-shot).

        Args:
            name: Job name to run.

        Returns:
            True if job completed successfully.
        """
        job = self._jobs.get(name)
        if not job:
            logger.error(f"Job '{name}' not found")
            return False

        return await self._execute_job(job)

    async def _execute_job(self, job: Job) -> bool:
        """Execute a single job with retry logic.

        Args:
            job: Job to execute.

        Returns:
            True if successful.
        """
        run = JobRun(started_at=datetime.now(timezone.utc))
        job.last_run = run

        for attempt in range(job.max_retries + 1):
            try:
                logger.debug(f"Running job '{job.name}' (attempt {attempt + 1})")
                await job.func()

                run.ended_at = datetime.now(timezone.utc)
                run.status = JobStatus.COMPLETED
                job._consecutive_failures = 0

                logger.debug(
                    f"Job '{job.name}' completed in {run.duration_seconds:.2f}s"
                )

                # Update history
                job.history.append(run)
                if len(job.history) > self._max_history:
                    job.history = job.history[-self._max_history:]

                return True

            except asyncio.CancelledError:
                run.ended_at = datetime.now(timezone.utc)
                run.status = JobStatus.CANCELLED
                logger.info(f"Job '{job.name}' cancelled")
                raise

            except Exception as e:
                logger.warning(
                    f"Job '{job.name}' failed (attempt {attempt + 1}): {e}"
                )

                if attempt < job.max_retries:
                    await asyncio.sleep(job.retry_delay_seconds)
                else:
                    # Final failure
                    run.ended_at = datetime.now(timezone.utc)
                    run.status = JobStatus.FAILED
                    run.error = str(e)
                    job._consecutive_failures += 1

                    job.history.append(run)
                    if len(job.history) > self._max_history:
                        job.history = job.history[-self._max_history:]

                    # Error callback
                    if self._error_callback:
                        try:
                            await self._error_callback(job.name, e)
                        except Exception as cb_error:
                            logger.error(f"Error callback failed: {cb_error}")

                    return False

        return False

    async def _job_loop(self, job: Job) -> None:
        """Main loop for a single job.

        Args:
            job: Job to run in loop.
        """
        # Run immediately if configured
        if job.run_immediately:
            await self._execute_job(job)

        while self._running:
            try:
                # Wait for next interval
                await asyncio.sleep(job.interval_seconds)

                if not self._running or not job.enabled:
                    continue

                await self._execute_job(job)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in job loop '{job.name}': {e}")

    async def start(self) -> None:
        """Start the scheduler.

        Runs all enabled jobs in their own tasks.
        """
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        logger.info(f"Starting scheduler with {len(self._jobs)} jobs")

        # Start job loops
        for job in self._jobs.values():
            if job.enabled:
                job._task = asyncio.create_task(self._job_loop(job))

    async def stop(self, timeout: float = 10.0) -> None:
        """Stop the scheduler.

        Args:
            timeout: Max seconds to wait for jobs to finish.
        """
        if not self._running:
            return

        self._running = False
        logger.info("Stopping scheduler...")

        # Cancel all job tasks
        tasks = [
            job._task for job in self._jobs.values()
            if job._task and not job._task.done()
        ]

        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.wait(tasks, timeout=timeout)

        logger.info("Scheduler stopped")

    async def wait(self) -> None:
        """Wait for scheduler to be stopped."""
        while self._running:
            await asyncio.sleep(1.0)

    def get_status(self) -> dict:
        """Get scheduler status.

        Returns:
            Dictionary with scheduler metrics.
        """
        return {
            "running": self._running,
            "jobs": [job.to_dict() for job in self._jobs.values()],
            "total_jobs": len(self._jobs),
            "enabled_jobs": sum(1 for j in self._jobs.values() if j.enabled),
        }


async def create_default_scheduler(
    market_collector=None,
    opportunity_scanner=None,
    risk_checker=None,
) -> Scheduler:
    """Create scheduler with default jobs.

    Args:
        market_collector: Market collection function.
        opportunity_scanner: Opportunity scan function.
        risk_checker: Risk check function.

    Returns:
        Configured Scheduler.
    """
    scheduler = Scheduler()

    if market_collector:
        scheduler.add_job(
            name="collect_markets",
            func=market_collector,
            interval_seconds=300,  # 5 minutes
        )

    if opportunity_scanner:
        scheduler.add_job(
            name="scan_opportunities",
            func=opportunity_scanner,
            interval_seconds=60,  # 1 minute
            run_immediately=True,
        )

    if risk_checker:
        scheduler.add_job(
            name="check_risk",
            func=risk_checker,
            interval_seconds=60,  # 1 minute
        )

    return scheduler
